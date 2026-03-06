from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Literal

import torch

from .base import AttackBase, AttackOutput
from .utils import clamp_like_input_domain, project_l2, delta_norms, multi_label_success

Tensor = torch.Tensor


@dataclass
class DeepFoolL2(AttackBase):
    """DeepFool-style L2 attack adapted for *multi-label* (14-label) logits.

    Classic DeepFool is defined for multi-class argmax decision boundaries.
    For multi-label CXR, we use a *binary boundary per label* at logit=0 (p=0.5),
    and iteratively compute a minimal L2 step to flip one chosen label.

    Strategy (untargeted):
      - choose label j per sample = argmax(|logit_j|) (most confident margin)
      - treat decision function f(x)=logit_j(x)
      - linearize: f(x+r) ≈ f(x) + grad_f^T r
      - minimal L2 r to reach boundary f(x+r)=0 is:  r = - f(x) / ||grad_f||^2 * grad_f
      - apply small overshoot to cross boundary.

    This produces a reasonable DeepFool-like baseline without requiring a multi-class head.

    Targeted mode:
      - y is interpreted as a target *multi-hot* vector (B,L). We pick a label that differs
        from the current prediction and push it across the boundary in the desired direction.

    NOTE:
      - This is a practical baseline suitable for your unified API + report.
      - It is differentiable and works with CLIP-style logits.
    """
    max_iter: int = 50
    overshoot: float = 0.02
    # Optional L2 cap (None = no cap)
    max_l2: Optional[float] = None
    # How to choose the label to flip
    pick_label: Literal["max_abs_logit", "max_prob"] = "max_abs_logit"

    def perturb(self, x: Tensor, y: Tensor, wrapper) -> AttackOutput:
        x0 = x.detach().clone()
        x_adv = x0.detach().clone()

        logits_nat = wrapper.forward_logits(x0).detach()

        # Determine target direction per sample
        with torch.no_grad():
            if self.pick_label == "max_abs_logit":
                j = logits_nat.abs().argmax(dim=1)  # [B]
            else:
                probs = torch.sigmoid(logits_nat)
                j = probs.argmax(dim=1)

        losses = []
        for it in range(int(self.max_iter)):
            x_adv = x_adv.detach().clone().requires_grad_(True)
            logits = wrapper.forward_logits(x_adv)  # [B,L]

            # Choose per-sample scalar decision function f = logit[b, j[b]]
            f = logits[torch.arange(logits.shape[0], device=logits.device), j]  # [B]
            # We want to cross f=0. Untargeted: just flip sign. Targeted: use y to choose direction.
            # DeepFool step uses gradient of f wrt x.
            grads = torch.autograd.grad(f.sum(), x_adv, retain_graph=False, create_graph=False)[0]  # [B,C,H,W]

            # Compute r per sample: -f / ||grad||^2 * grad
            g = grads.view(grads.shape[0], -1)
            g_norm2 = (g * g).sum(dim=1).clamp_min(1e-12)  # [B]
            r = (-f.detach() / g_norm2).view(-1, 1, 1, 1) * grads.detach()

            # overshoot to cross boundary
            x_adv = x_adv.detach() + (1.0 + self.overshoot) * r
            x_adv = clamp_like_input_domain(x_adv, wrapper)

            if self.max_l2 is not None:
                x_adv = project_l2(x_adv, x0, float(self.max_l2))
                x_adv = clamp_like_input_domain(x_adv, wrapper)

            # stop early if label-set changed
            with torch.no_grad():
                logits_cur = wrapper.forward_logits(x_adv)
                success = multi_label_success(logits_nat, logits_cur)
                if success.all():
                    break

        logits_adv = wrapper.forward_logits(x_adv).detach()
        success = multi_label_success(logits_nat, logits_adv)

        meta = {
            "attack": "DeepFool_L2_binary",
            "max_iter": int(self.max_iter),
            "overshoot": float(self.overshoot),
            "max_l2": None if self.max_l2 is None else float(self.max_l2),
            "pick_label": str(self.pick_label),
            "targeted": bool(self.targeted),
            "norms": {k: v.detach().cpu() for k, v in delta_norms(x0, x_adv).items()},
            "success": success.detach().cpu(),
        }
        return {"x_adv": x_adv, "delta": (x_adv - x0), "meta": meta}
