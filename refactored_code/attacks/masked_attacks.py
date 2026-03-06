#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
masked_attacks.py

Lesion-aware (masked) adversarial attacks for medical imaging.
Constrains perturbations to specific spatial regions (lesion masks).

Key Innovation:
    - Traditional attacks: δ constrained globally (||δ||_∞ ≤ ε)
    - Lesion-aware attacks: δ constrained spatially (δ = δ ⊙ M, ||δ||_∞ ≤ ε)
      where M is a binary lesion mask derived from Grad-CAM/attention

References:
    - Li et al. (2025): LatAtk - Lesion-focused attacks with high transferability
    - Madry et al. (2018): PGD attack baseline

Author: Generated for HONER_PROJECT
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Union

import torch
import torch.nn.functional as F

from .base import AttackBase, AttackOutput
from .utils import clamp_like_input_domain, project_linf, delta_norms, multi_label_success

Tensor = torch.Tensor


@dataclass
class MaskedPGDLinf(AttackBase):
    """
    Lesion-aware PGD attack with spatial masking.

    Perturbations are constrained to lesion regions:
        δ_masked = δ ⊙ M
    where M is a binary spatial mask [B, 1, H, W] or [B, C, H, W].

    Args:
        epsilon: L∞ constraint (pixel domain)
        step_size: Step size per iteration
        steps: Number of PGD iterations
        random_start: Random initialization within ε-ball
        mask_mode: How to apply mask
            - "multiply": δ = δ * M (strict masking, zero outside)
            - "soft": δ = δ * (M + λ) (allow small perturbations outside)
        loss_reduction: "mean" or "sum"
    """
    epsilon: float = 8.0
    step_size: float = 2.0
    steps: int = 10
    random_start: bool = True
    mask_mode: str = "multiply"  # "multiply" or "soft"
    soft_lambda: float = 0.1  # For mask_mode="soft"
    loss_reduction: str = "mean"

    def perturb(
        self,
        x: Tensor,
        y: Tensor,
        wrapper,
        mask: Optional[Tensor] = None,
    ) -> AttackOutput:
        """
        Generate lesion-aware adversarial examples.

        Args:
            x: Input images [B, C, H, W] in wrapper's input domain
            y: Labels [B, L]
            wrapper: Model wrapper (forward_logits, loss_fn)
            mask: Binary lesion mask [B, 1, H, W] or [B, C, H, W]
                  If None, falls back to standard PGD (no masking)

        Returns:
            AttackOutput with x_adv, delta, and metadata
        """
        x0 = x.detach().clone()
        logits_nat = wrapper.forward_logits(x0).detach()

        # Prepare mask
        if mask is not None:
            mask = mask.to(x.device).float()
            # Broadcast to match image channels if needed
            if mask.shape[1] == 1 and x.shape[1] > 1:
                mask = mask.repeat(1, x.shape[1], 1, 1)
            mask = mask.clamp(0, 1)  # Ensure binary [0, 1]

            # Apply soft masking if specified
            if self.mask_mode == "soft":
                mask = mask + self.soft_lambda
        else:
            # No mask: full image attack
            mask = torch.ones_like(x0)

        # Initialize perturbation
        if self.random_start:
            # Random init in ε-ball, then mask
            noise = torch.empty_like(x0).uniform_(-self.epsilon, self.epsilon)
            delta = noise * mask  # Apply mask immediately
            x_adv = x0 + delta
            x_adv = clamp_like_input_domain(x_adv, wrapper)
        else:
            x_adv = x0.clone()

        losses = []

        for step in range(int(self.steps)):
            x_adv = x_adv.detach().clone().requires_grad_(True)

            logits = wrapper.forward_logits(x_adv)
            loss = wrapper.loss_fn(logits, y, reduction=self.loss_reduction)
            losses.append(loss.detach())

            # Objective: maximize loss (untargeted) or minimize (targeted)
            obj = -loss if self.targeted else loss
            grad = torch.autograd.grad(obj, x_adv, retain_graph=False, create_graph=False)[0]

            # Mask the gradient: only update in lesion regions
            grad_masked = grad * mask

            # PGD step
            x_adv = x_adv.detach() + self.step_size * torch.sign(grad_masked.detach())

            # Project onto L∞ ball
            delta = x_adv - x0
            delta = project_linf(delta, torch.zeros_like(delta), self.epsilon)

            # Re-apply spatial mask (critical for lesion-aware constraint)
            delta = delta * mask

            x_adv = x0 + delta
            x_adv = clamp_like_input_domain(x_adv, wrapper)

        # Final evaluation
        logits_adv = wrapper.forward_logits(x_adv).detach()
        success = multi_label_success(logits_nat, logits_adv)

        # Compute norms
        delta_final = x_adv - x0
        norms = delta_norms(x0, x_adv)

        # Additional metrics: lesion vs non-lesion perturbation
        if mask is not None:
            delta_in_mask = (delta_final * mask).abs().flatten(1).max(dim=1).values
            delta_out_mask = (delta_final * (1 - mask)).abs().flatten(1).max(dim=1).values
        else:
            delta_in_mask = norms["linf"]
            delta_out_mask = torch.zeros_like(norms["linf"])

        meta = {
            "attack": "MaskedPGD_Linf",
            "epsilon": float(self.epsilon),
            "step_size": float(self.step_size),
            "steps": int(self.steps),
            "random_start": bool(self.random_start),
            "mask_mode": str(self.mask_mode),
            "targeted": bool(self.targeted),
            "loss_curve": torch.stack(losses).detach().cpu() if len(losses) else None,
            "norms": {k: v.detach().cpu() for k, v in norms.items()},
            "norms_in_mask": delta_in_mask.detach().cpu(),
            "norms_out_mask": delta_out_mask.detach().cpu(),
            "success": success.detach().cpu(),
            "mask_coverage": mask.flatten(1).mean(dim=1).detach().cpu() if mask is not None else None,
        }

        return {"x_adv": x_adv, "delta": delta_final, "meta": meta}


@dataclass
class MaskedFGSMLinf(AttackBase):
    """
    Lesion-aware FGSM attack (single-step masked PGD).

    Args:
        epsilon: L∞ constraint
        mask_mode: "multiply" or "soft"
        soft_lambda: For soft masking
        loss_reduction: "mean" or "sum"
    """
    epsilon: float = 8.0
    mask_mode: str = "multiply"
    soft_lambda: float = 0.1
    loss_reduction: str = "mean"

    def perturb(
        self,
        x: Tensor,
        y: Tensor,
        wrapper,
        mask: Optional[Tensor] = None,
    ) -> AttackOutput:
        """Generate lesion-aware FGSM adversarial examples."""
        x0 = x.detach().clone()
        x_adv = x0.detach().clone().requires_grad_(True)

        logits_nat = wrapper.forward_logits(x0).detach()

        # Prepare mask
        if mask is not None:
            mask = mask.to(x.device).float()
            if mask.shape[1] == 1 and x.shape[1] > 1:
                mask = mask.repeat(1, x.shape[1], 1, 1)
            mask = mask.clamp(0, 1)
            if self.mask_mode == "soft":
                mask = mask + self.soft_lambda
        else:
            mask = torch.ones_like(x0)

        # Forward pass
        logits = wrapper.forward_logits(x_adv)
        loss = wrapper.loss_fn(logits, y, reduction=self.loss_reduction)

        obj = -loss if self.targeted else loss
        grad = torch.autograd.grad(obj, x_adv, retain_graph=False, create_graph=False)[0]

        # Mask gradient
        grad_masked = grad * mask

        # FGSM step
        delta = self.epsilon * torch.sign(grad_masked.detach())

        # Apply spatial mask
        delta = delta * mask

        x_adv = x0 + delta
        x_adv = clamp_like_input_domain(x_adv, wrapper)

        # Evaluate
        logits_adv = wrapper.forward_logits(x_adv).detach()
        success = multi_label_success(logits_nat, logits_adv)

        norms = delta_norms(x0, x_adv)

        meta = {
            "attack": "MaskedFGSM_Linf",
            "epsilon": float(self.epsilon),
            "mask_mode": str(self.mask_mode),
            "targeted": bool(self.targeted),
            "norms": {k: v.detach().cpu() for k, v in norms.items()},
            "success": success.detach().cpu(),
        }

        return {"x_adv": x_adv, "delta": delta, "meta": meta}


@dataclass
class MaskedDeepFoolL2(AttackBase):
    """
    Lesion-aware DeepFool attack.

    Iteratively finds minimal L2 perturbation to cross decision boundary,
    constrained to lesion regions.

    Args:
        max_iter: Maximum iterations
        overshoot: Overshoot factor
        max_l2: Optional L2 cap
        mask_mode: "multiply" or "soft"
    """
    max_iter: int = 50
    overshoot: float = 0.02
    max_l2: Optional[float] = None
    mask_mode: str = "multiply"
    soft_lambda: float = 0.1
    pick_label: str = "max_abs_logit"

    def perturb(
        self,
        x: Tensor,
        y: Tensor,
        wrapper,
        mask: Optional[Tensor] = None,
    ) -> AttackOutput:
        """Generate lesion-aware DeepFool adversarial examples."""
        x0 = x.detach().clone()
        x_adv = x0.detach().clone()

        logits_nat = wrapper.forward_logits(x0).detach()

        # Prepare mask
        if mask is not None:
            mask = mask.to(x.device).float()
            if mask.shape[1] == 1 and x.shape[1] > 1:
                mask = mask.repeat(1, x.shape[1], 1, 1)
            mask = mask.clamp(0, 1)
            if self.mask_mode == "soft":
                mask = mask + self.soft_lambda
        else:
            mask = torch.ones_like(x0)

        # Select target label per sample
        with torch.no_grad():
            if self.pick_label == "max_abs_logit":
                j = logits_nat.abs().argmax(dim=1)
            else:
                probs = torch.sigmoid(logits_nat)
                j = probs.argmax(dim=1)

        for it in range(int(self.max_iter)):
            x_adv = x_adv.detach().clone().requires_grad_(True)
            logits = wrapper.forward_logits(x_adv)

            # Per-sample decision function
            f = logits[torch.arange(logits.shape[0], device=logits.device), j]
            grads = torch.autograd.grad(f.sum(), x_adv, retain_graph=False, create_graph=False)[0]

            # Mask gradients
            grads_masked = grads * mask

            # DeepFool step: r = -f / ||grad||^2 * grad
            g = grads_masked.view(grads_masked.shape[0], -1)
            g_norm2 = (g * g).sum(dim=1).clamp_min(1e-12)
            r = (-f.detach() / g_norm2).view(-1, 1, 1, 1) * grads_masked.detach()

            # Apply mask to perturbation
            r = r * mask

            x_adv = x_adv.detach() + (1.0 + self.overshoot) * r
            x_adv = clamp_like_input_domain(x_adv, wrapper)

            # Optional L2 projection
            if self.max_l2 is not None:
                delta = x_adv - x0
                delta_norm = delta.view(delta.shape[0], -1).norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
                scale = torch.minimum(torch.ones_like(delta_norm), self.max_l2 / delta_norm)
                delta = delta * scale.view(-1, 1, 1, 1)
                # Re-apply mask
                delta = delta * mask
                x_adv = x0 + delta
                x_adv = clamp_like_input_domain(x_adv, wrapper)

            # Check convergence
            with torch.no_grad():
                logits_cur = wrapper.forward_logits(x_adv)
                success = multi_label_success(logits_nat, logits_cur)
                if success.all():
                    break

        logits_adv = wrapper.forward_logits(x_adv).detach()
        success = multi_label_success(logits_nat, logits_adv)

        norms = delta_norms(x0, x_adv)

        meta = {
            "attack": "MaskedDeepFool_L2",
            "max_iter": int(self.max_iter),
            "overshoot": float(self.overshoot),
            "max_l2": None if self.max_l2 is None else float(self.max_l2),
            "mask_mode": str(self.mask_mode),
            "targeted": bool(self.targeted),
            "norms": {k: v.detach().cpu() for k, v in norms.items()},
            "success": success.detach().cpu(),
        }

        return {"x_adv": x_adv, "delta": (x_adv - x0), "meta": meta}


# ============================================================================
# Utility: Automatic mask generation wrapper
# ============================================================================

def generate_and_attack(
    x: Tensor,
    y: Tensor,
    wrapper,
    attack: AttackBase,
    mask_generator: callable,
    mask_threshold: float = 0.5,
) -> AttackOutput:
    """
    End-to-end: Generate lesion mask and perform masked attack.

    Args:
        x: Input images [B, C, H, W]
        y: Labels [B, L]
        wrapper: Model wrapper
        attack: Masked attack instance (MaskedPGDLinf, etc.)
        mask_generator: Function that takes (x, y, wrapper) and returns mask [B, 1, H, W]
        mask_threshold: Threshold for mask binarization

    Returns:
        AttackOutput with x_adv, delta, meta
    """
    # Generate lesion mask
    mask = mask_generator(x, y, wrapper)

    # Binarize
    mask = (mask >= mask_threshold).float()

    # Perform attack
    return attack.perturb(x, y, wrapper, mask=mask)
