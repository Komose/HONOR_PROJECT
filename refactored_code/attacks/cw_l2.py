#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cw_l2.py

Carlini & Wagner (C&W) L2 attack - PyTorch implementation.
Optimizes for minimal L2 perturbation while ensuring misclassification.

Reference:
    Carlini & Wagner (2017): "Towards Evaluating the Robustness of Neural Networks"
    https://arxiv.org/abs/1608.04644

Key differences from PGD:
    - Optimization-based (not gradient-based iterative)
    - Minimizes ||δ||_2 + c · f(x+δ) where f is a loss function
    - Uses tanh transformation for box constraints
    - Binary search over constant c to find optimal perturbation

Adapted for multi-label medical imaging (CheXpert 14-label).

Author: Generated for HONER_PROJECT
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Literal

import torch
import torch.nn as nn
import torch.optim as optim

from .base import AttackBase, AttackOutput
from .utils import delta_norms, multi_label_success

Tensor = torch.Tensor


def atanh(x: Tensor, eps: float = 1e-6) -> Tensor:
    """
    Numerically stable arctanh.

    Args:
        x: Input tensor in (-1, 1)
        eps: Small constant for numerical stability

    Returns:
        arctanh(x)
    """
    x = x.clamp(-1 + eps, 1 - eps)
    return 0.5 * torch.log((1 + x) / (1 - x))


@dataclass
class CWL2(AttackBase):
    """
    Carlini & Wagner L2 attack (PyTorch implementation).

    Solves:
        min ||δ||_2^2 + c · loss(x+δ, y)

    where loss encourages misclassification.

    For multi-label (CheXpert):
        - Untargeted: Maximize prediction error on true labels
        - Targeted: Minimize distance to target label set

    Args:
        confidence: Confidence margin (κ in paper)
        c_init: Initial value of constant c
        c_range: (c_min, c_max) for binary search
        binary_search_steps: Number of binary search iterations
        max_iterations: Maximum optimization iterations per c
        learning_rate: Adam learning rate
        abort_early: Stop if attack succeeds early
        clip_min: Minimum pixel value (input domain)
        clip_max: Maximum pixel value (input domain)
        loss_mode: Loss function type
            - "margin": Use margin loss (default for classification)
            - "bce": Binary cross-entropy (for multi-label)
    """
    confidence: float = 0.0  # κ: margin for successful attack
    c_init: float = 1.0
    c_range: tuple = (0.0, 10.0)
    binary_search_steps: int = 9
    max_iterations: int = 1000
    learning_rate: float = 0.01
    abort_early: bool = True
    clip_min: float = 0.0
    clip_max: float = 255.0
    loss_mode: Literal["margin", "bce"] = "bce"  # "bce" better for multi-label

    def perturb(self, x: Tensor, y: Tensor, wrapper) -> AttackOutput:
        """
        Generate C&W L2 adversarial examples.

        Args:
            x: Input images [B, C, H, W] in [clip_min, clip_max]
            y: Labels [B, L] (multi-label binary)
            wrapper: Model wrapper (forward_logits, loss_fn)

        Returns:
            AttackOutput with x_adv, delta, meta
        """
        batch_size = x.shape[0]
        x0 = x.detach().clone()

        # Get natural predictions
        with torch.no_grad():
            logits_nat = wrapper.forward_logits(x0)

        # Initialize best adversarial examples
        best_adv = x0.clone()
        best_l2 = torch.full((batch_size,), float('inf'), device=x.device)
        best_success = torch.zeros(batch_size, dtype=torch.bool, device=x.device)

        # Binary search over c
        c_lower = torch.full((batch_size,), self.c_range[0], device=x.device)
        c_upper = torch.full((batch_size,), self.c_range[1], device=x.device)
        c = torch.full((batch_size,), self.c_init, device=x.device)

        for binary_step in range(self.binary_search_steps):
            # Initialize adversarial variable in tanh space
            # x = (clip_max - clip_min)/2 * tanh(w) + (clip_max + clip_min)/2
            # => w = atanh((2*x - clip_max - clip_min) / (clip_max - clip_min))

            scale = (self.clip_max - self.clip_min) / 2.0
            offset = (self.clip_max + self.clip_min) / 2.0

            w = atanh((x0 - offset) / scale)
            w = w.detach().clone().requires_grad_(True)

            optimizer = optim.Adam([w], lr=self.learning_rate)

            prev_loss = float('inf')
            for iteration in range(self.max_iterations):
                optimizer.zero_grad()

                # Transform w to valid input range
                x_adv = scale * torch.tanh(w) + offset

                # Get model predictions
                logits_adv = wrapper.forward_logits(x_adv)

                # Compute loss components
                l2_loss = torch.sum((x_adv - x0) ** 2, dim=[1, 2, 3])  # [B]

                if self.loss_mode == "bce":
                    # Multi-label BCE loss
                    # Untargeted: maximize loss on true labels
                    f_loss = wrapper.loss_fn(logits_adv, y, reduction='none')  # [B, L] or [B]
                    if f_loss.dim() > 1:
                        f_loss = f_loss.sum(dim=1)  # [B]

                    if not self.targeted:
                        f_loss = -f_loss  # Maximize loss (make predictions worse)

                elif self.loss_mode == "margin":
                    # Margin-based loss (similar to original C&W)
                    # For multi-label: push predictions away from targets
                    probs_adv = torch.sigmoid(logits_adv)  # [B, L]

                    if not self.targeted:
                        # Untargeted: reduce confidence on positive labels
                        correct_prob = (y * probs_adv + (1 - y) * (1 - probs_adv)).sum(dim=1)
                        f_loss = -correct_prob + self.confidence
                    else:
                        # Targeted: increase confidence on target labels
                        target_prob = (y * probs_adv + (1 - y) * (1 - probs_adv)).sum(dim=1)
                        f_loss = -(target_prob - self.confidence)

                else:
                    raise ValueError(f"Unknown loss_mode: {self.loss_mode}")

                # Total loss: L2 + c * f
                total_loss = l2_loss + c * f_loss
                loss = total_loss.sum()

                loss.backward()
                optimizer.step()

                # Early abort if loss not decreasing
                if self.abort_early and iteration % 100 == 0:
                    if loss.item() >= prev_loss * 0.9999:
                        break
                    prev_loss = loss.item()

            # Final evaluation for this c value
            with torch.no_grad():
                x_adv = scale * torch.tanh(w) + offset
                logits_adv = wrapper.forward_logits(x_adv)

                # Check if attack succeeded
                success = multi_label_success(logits_nat, logits_adv)

                # Compute L2 distance
                l2_dist = torch.sqrt(torch.sum((x_adv - x0) ** 2, dim=[1, 2, 3]))

                # Update best adversarial examples
                for i in range(batch_size):
                    if success[i] and l2_dist[i] < best_l2[i]:
                        best_adv[i] = x_adv[i]
                        best_l2[i] = l2_dist[i]
                        best_success[i] = True

                # Binary search update
                for i in range(batch_size):
                    if success[i]:
                        # Attack succeeded, try smaller c
                        c_upper[i] = c[i]
                    else:
                        # Attack failed, try larger c
                        c_lower[i] = c[i]

                    # Update c for next iteration
                    if c_upper[i] < self.c_range[1]:
                        c[i] = (c_lower[i] + c_upper[i]) / 2.0
                    else:
                        c[i] = c[i] * 10.0

        # Final check
        with torch.no_grad():
            logits_adv_final = wrapper.forward_logits(best_adv)
            success_final = multi_label_success(logits_nat, logits_adv_final)

        norms = delta_norms(x0, best_adv)

        meta = {
            "attack": "CW_L2",
            "confidence": float(self.confidence),
            "binary_search_steps": int(self.binary_search_steps),
            "max_iterations": int(self.max_iterations),
            "loss_mode": str(self.loss_mode),
            "targeted": bool(self.targeted),
            "norms": {k: v.detach().cpu() for k, v in norms.items()},
            "success": success_final.detach().cpu(),
            "best_l2": best_l2.detach().cpu(),
        }

        return {"x_adv": best_adv, "delta": (best_adv - x0), "meta": meta}


@dataclass
class MaskedCWL2(CWL2):
    """
    Lesion-aware C&W L2 attack with spatial masking.

    Constrains perturbations to lesion regions:
        min ||δ ⊙ M||_2^2 + c · loss(x + δ ⊙ M, y)

    where M is a binary spatial mask.

    Args:
        Same as CWL2, plus:
        mask_mode: "multiply" or "soft"
        soft_lambda: For soft masking
    """
    mask_mode: str = "multiply"
    soft_lambda: float = 0.1

    def perturb(
        self,
        x: Tensor,
        y: Tensor,
        wrapper,
        mask: Optional[Tensor] = None,
    ) -> AttackOutput:
        """
        Generate lesion-aware C&W L2 adversarial examples.

        Args:
            x: Input images [B, C, H, W]
            y: Labels [B, L]
            wrapper: Model wrapper
            mask: Binary lesion mask [B, 1, H, W] or [B, C, H, W]

        Returns:
            AttackOutput with x_adv, delta, meta
        """
        batch_size = x.shape[0]
        x0 = x.detach().clone()

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

        with torch.no_grad():
            logits_nat = wrapper.forward_logits(x0)

        best_adv = x0.clone()
        best_l2 = torch.full((batch_size,), float('inf'), device=x.device)
        best_success = torch.zeros(batch_size, dtype=torch.bool, device=x.device)

        c_lower = torch.full((batch_size,), self.c_range[0], device=x.device)
        c_upper = torch.full((batch_size,), self.c_range[1], device=x.device)
        c = torch.full((batch_size,), self.c_init, device=x.device)

        for binary_step in range(self.binary_search_steps):
            # Initialize perturbation in tanh space
            scale = (self.clip_max - self.clip_min) / 2.0
            offset = (self.clip_max + self.clip_min) / 2.0

            # Initialize w for perturbation (not full image)
            delta_init = torch.zeros_like(x0)
            w = atanh(delta_init / (scale * 2))  # Small init
            w = w.detach().clone().requires_grad_(True)

            optimizer = optim.Adam([w], lr=self.learning_rate)

            prev_loss = float('inf')
            for iteration in range(self.max_iterations):
                optimizer.zero_grad()

                # Transform w to perturbation
                delta = scale * torch.tanh(w)

                # Apply mask to perturbation
                delta_masked = delta * mask

                # Generate adversarial example
                x_adv = (x0 + delta_masked).clamp(self.clip_min, self.clip_max)

                logits_adv = wrapper.forward_logits(x_adv)

                # L2 loss only on masked region
                l2_loss = torch.sum((delta_masked) ** 2, dim=[1, 2, 3])

                # Classification loss
                if self.loss_mode == "bce":
                    f_loss = wrapper.loss_fn(logits_adv, y, reduction='none')
                    if f_loss.dim() > 1:
                        f_loss = f_loss.sum(dim=1)
                    if not self.targeted:
                        f_loss = -f_loss
                elif self.loss_mode == "margin":
                    probs_adv = torch.sigmoid(logits_adv)
                    if not self.targeted:
                        correct_prob = (y * probs_adv + (1 - y) * (1 - probs_adv)).sum(dim=1)
                        f_loss = -correct_prob + self.confidence
                    else:
                        target_prob = (y * probs_adv + (1 - y) * (1 - probs_adv)).sum(dim=1)
                        f_loss = -(target_prob - self.confidence)
                else:
                    raise ValueError(f"Unknown loss_mode: {self.loss_mode}")

                total_loss = l2_loss + c * f_loss
                loss = total_loss.sum()

                loss.backward()
                optimizer.step()

                if self.abort_early and iteration % 100 == 0:
                    if loss.item() >= prev_loss * 0.9999:
                        break
                    prev_loss = loss.item()

            # Evaluation
            with torch.no_grad():
                delta = scale * torch.tanh(w)
                delta_masked = delta * mask
                x_adv = (x0 + delta_masked).clamp(self.clip_min, self.clip_max)
                logits_adv = wrapper.forward_logits(x_adv)

                success = multi_label_success(logits_nat, logits_adv)
                l2_dist = torch.sqrt(torch.sum((x_adv - x0) ** 2, dim=[1, 2, 3]))

                for i in range(batch_size):
                    if success[i] and l2_dist[i] < best_l2[i]:
                        best_adv[i] = x_adv[i]
                        best_l2[i] = l2_dist[i]
                        best_success[i] = True

                for i in range(batch_size):
                    if success[i]:
                        c_upper[i] = c[i]
                    else:
                        c_lower[i] = c[i]

                    if c_upper[i] < self.c_range[1]:
                        c[i] = (c_lower[i] + c_upper[i]) / 2.0
                    else:
                        c[i] = c[i] * 10.0

        with torch.no_grad():
            logits_adv_final = wrapper.forward_logits(best_adv)
            success_final = multi_label_success(logits_nat, logits_adv_final)

        norms = delta_norms(x0, best_adv)

        meta = {
            "attack": "MaskedCW_L2",
            "confidence": float(self.confidence),
            "mask_mode": str(self.mask_mode),
            "targeted": bool(self.targeted),
            "norms": {k: v.detach().cpu() for k, v in norms.items()},
            "success": success_final.detach().cpu(),
            "best_l2": best_l2.detach().cpu(),
        }

        return {"x_adv": best_adv, "delta": (best_adv - x0), "meta": meta}
