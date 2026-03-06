from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional

import torch

from .base import AttackBase, AttackOutput
from .utils import clamp_like_input_domain, project_linf, delta_norms, multi_label_success

Tensor = torch.Tensor


@dataclass
class PGDLinf(AttackBase):
    """Projected Gradient Descent (Linf) attack.

    Untargeted: maximize loss wrt true labels.
    Targeted:   minimize loss wrt target labels.

    Operates in wrapper's input domain (default: pixel [0,255]).
    """
    epsilon: float = 8.0
    step_size: float = 2.0
    steps: int = 10
    random_start: bool = True
    loss_reduction: str = "mean"

    def perturb(self, x: Tensor, y: Tensor, wrapper) -> AttackOutput:
        x0 = x.detach().clone()
        logits_nat = wrapper.forward_logits(x0).detach()

        # init
        if self.random_start:
            # uniform noise in Linf ball
            noise = torch.empty_like(x0).uniform_(-self.epsilon, self.epsilon)
            x_adv = x0 + noise
            x_adv = clamp_like_input_domain(x_adv, wrapper)
        else:
            x_adv = x0.clone()

        losses = []

        for _ in range(int(self.steps)):
            x_adv = x_adv.detach().clone().requires_grad_(True)

            logits = wrapper.forward_logits(x_adv)
            loss = wrapper.loss_fn(logits, y, reduction=self.loss_reduction)
            losses.append(loss.detach())

            obj = -loss if self.targeted else loss
            grad = torch.autograd.grad(obj, x_adv, retain_graph=False, create_graph=False)[0]

            x_adv = x_adv.detach() + self.step_size * torch.sign(grad.detach())

            # project + clamp
            x_adv = project_linf(x_adv, x0, self.epsilon)
            x_adv = clamp_like_input_domain(x_adv, wrapper)

        logits_adv = wrapper.forward_logits(x_adv).detach()
        success = multi_label_success(logits_nat, logits_adv)

        meta = {
            "attack": "PGD_Linf",
            "epsilon": float(self.epsilon),
            "step_size": float(self.step_size),
            "steps": int(self.steps),
            "random_start": bool(self.random_start),
            "targeted": bool(self.targeted),
            "loss_curve": torch.stack(losses).detach().cpu() if len(losses) else None,
            "norms": {k: v.detach().cpu() for k, v in delta_norms(x0, x_adv).items()},
            "success": success.detach().cpu(),
        }
        return {"x_adv": x_adv, "delta": (x_adv - x0), "meta": meta}
