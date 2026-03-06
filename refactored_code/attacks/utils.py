from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

Tensor = torch.Tensor


def clamp_like_input_domain(x: Tensor, wrapper) -> Tensor:
    """Clamp x in the wrapper's declared *input domain* (not normalized space)."""
    if not getattr(wrapper, "clamp", True):
        return x
    dom = getattr(wrapper, "input_domain", None)
    if dom is None:
        # default to [0,1] if unknown
        return x.clamp(0.0, 1.0)
    lo, hi = dom.value_range
    return x.clamp(float(lo), float(hi))


def project_linf(x_adv: Tensor, x0: Tensor, eps: float) -> Tensor:
    """Project onto Linf ball around x0 with radius eps (same domain as x0)."""
    return torch.max(torch.min(x_adv, x0 + eps), x0 - eps)


def project_l2(x_adv: Tensor, x0: Tensor, eps_l2: float) -> Tensor:
    """Project onto L2 ball around x0 with radius eps_l2."""
    delta = (x_adv - x0).view(x0.shape[0], -1)
    norm = torch.norm(delta, p=2, dim=1, keepdim=True).clamp_min(1e-12)
    factor = torch.minimum(torch.ones_like(norm), eps_l2 / norm)
    delta = delta * factor
    return (x0 + delta.view_as(x0))


def delta_norms(x0: Tensor, x_adv: Tensor) -> Dict[str, Tensor]:
    """Per-sample norms of delta in the *input domain*. Returns tensors of shape [N]."""
    delta = (x_adv - x0).view(x0.shape[0], -1)
    linf = delta.abs().max(dim=1).values
    l2 = torch.norm(delta, p=2, dim=1)
    l0 = (delta.abs() > 0).float().sum(dim=1)
    return {"linf": linf, "l2": l2, "l0": l0}


def sigmoid_probs_from_logits(logits: Tensor) -> Tensor:
    """Utility for multi-label logits -> probabilities."""
    return torch.sigmoid(logits)


def multi_label_success(
    logits_nat: Tensor,
    logits_adv: Tensor,
    *,
    threshold: float = 0.0,
) -> Tensor:
    """Define 'success' as: predicted label-set changes (using logit threshold).
    Returns bool tensor [N].
    """
    nat = (logits_nat > threshold)
    adv = (logits_adv > threshold)
    return (nat != adv).any(dim=1)
