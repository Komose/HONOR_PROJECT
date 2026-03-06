from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, TypedDict

import torch

Tensor = torch.Tensor


class WrapperProtocol(Protocol):
    def forward_logits(self, x: Tensor) -> Tensor: ...
    @staticmethod
    def loss_fn(logits: Tensor, y: Tensor, **kwargs) -> Tensor: ...


class AttackOutput(TypedDict, total=False):
    x_adv: Tensor
    delta: Tensor
    meta: Dict[str, Any]


@dataclass
class AttackBase:
    """Base class for unified attacks.

    Implementations MUST implement:
        perturb(x, y, wrapper) -> AttackOutput

    Conventions:
      - x: NCHW float tensor in wrapper's *input domain* (default in your pipeline: [0,255])
      - y: labels tensor
      - wrapper: provides forward_logits + loss_fn
    """
    targeted: bool = False  # if True, y is treated as target label(s)

    def perturb(self, x: Tensor, y: Tensor, wrapper: WrapperProtocol) -> AttackOutput:
        raise NotImplementedError
