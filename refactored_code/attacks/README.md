# attacks/ (Unified Attack API)

This package provides three attacks with a unified signature:

    out = attack.perturb(x, y, wrapper)

- x: torch.Tensor (N,C,H,W) in wrapper's input domain (default your pipeline: pixel [0,255])
- y: labels tensor:
    - multi-label: shape (N, L) with 0/1
- wrapper: your FMAttackWrapper exposing:
    - forward_logits(x) -> logits (N, L)
    - loss_fn(logits, y) -> scalar

Implemented:
  - FGSMLinf: 1-step sign-gradient attack under Linf budget.
  - PGDLinf: iterative sign-gradient + projection under Linf budget.
  - DeepFoolL2: practical DeepFool-style L2 attack adapted to multi-label logits by flipping
    the most confident label across the logit=0 boundary.

Notes on epsilon:
  - If x is in [0,255], use epsilon like 8.0 and step_size like 2.0.
  - If x is in [0,1], use epsilon like 8/255 and step_size like 2/255.
