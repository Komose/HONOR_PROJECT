"""Unified adversarial attack API package (PyTorch).

All attacks implement the same signature:

    attack = SomeAttack(...)
    out = attack.perturb(x, y, wrapper)

Where:
  - x: torch.Tensor, NCHW, in the wrapper's declared input domain (default: pixel [0,255])
  - y: torch.Tensor, shape [N, L] (multi-label) or [N] (single-label), depending on wrapper/loss
  - wrapper: exposes forward_logits(x)->logits and loss_fn(logits,y)->scalar

Returns:
  dict with at least:
    - "x_adv": adversarial tensor (same shape as x)
    - "delta": x_adv - x
    - "meta": dictionary (loss curve, norms, success flags if provided)

Masked attacks additionally accept a `mask` parameter:
    out = attack.perturb(x, y, wrapper, mask=lesion_mask)

where mask is [B, 1, H, W] or [B, C, H, W] binary tensor.
"""

from .base import AttackBase, AttackOutput
from .fgsm_linf import FGSMLinf
from .pgd_linf import PGDLinf
from .deepfool_l2 import DeepFoolL2
from .cw_l2 import CWL2, MaskedCWL2
from .masked_attacks import MaskedPGDLinf, MaskedFGSMLinf, MaskedDeepFoolL2, generate_and_attack

__all__ = [
    "AttackBase", "AttackOutput",
    # Standard attacks
    "FGSMLinf", "PGDLinf", "DeepFoolL2", "CWL2",
    # Lesion-aware (masked) attacks
    "MaskedPGDLinf", "MaskedFGSMLinf", "MaskedDeepFoolL2", "MaskedCWL2",
    # Utilities
    "generate_and_attack",
]
