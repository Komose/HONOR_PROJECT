from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional

import torch

from .base import AttackBase, AttackOutput
from .utils import clamp_like_input_domain, project_linf, delta_norms, multi_label_success

Tensor = torch.Tensor


@dataclass
class FGSMLinf(AttackBase):
    """FGSM (Linf) attack.

    Untargeted: maximize loss wrt true labels.
    Targeted:   minimize loss wrt target labels.
    """
    epsilon: float = 8.0  # default in pixel domain [0,255]
    # If your input domain is [0,1], pass epsilon=8/255 instead.
    loss_reduction: str = "mean"

    def perturb(self, x: Tensor, y: Tensor, wrapper) -> AttackOutput:
        x0 = x.detach().clone()
        x_adv = x0.detach().clone().requires_grad_(True)

        logits_nat = wrapper.forward_logits(x0).detach()

        logits = wrapper.forward_logits(x_adv)
        loss = wrapper.loss_fn(logits, y, reduction=self.loss_reduction)
        # untargeted: ascend, targeted: descend
        obj = -loss if self.targeted else loss

        grad = torch.autograd.grad(obj, x_adv, retain_graph=False, create_graph=False)[0]
        x_adv = x_adv.detach() + self.epsilon * torch.sign(grad.detach())

        # project + clamp (in input domain)
        x_adv = project_linf(x_adv, x0, self.epsilon)
        x_adv = clamp_like_input_domain(x_adv, wrapper)

        logits_adv = wrapper.forward_logits(x_adv).detach()
        success = multi_label_success(logits_nat, logits_adv)

        meta = {
            "attack": "FGSM_Linf",
            "epsilon": float(self.epsilon),
            "targeted": bool(self.targeted),
            "norms": {k: v.detach().cpu() for k, v in delta_norms(x0, x_adv).items()},
            "success": success.detach().cpu(),
        }
        return {"x_adv": x_adv, "delta": (x_adv - x0), "meta": meta}
