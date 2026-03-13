"""
Unified Multi-Algorithm Attack Framework
=========================================

Integrates multiple attack algorithms (FGSM, PGD, C&W, DeepFool)
with multi-metric constraints (L∞, L0, L2).

Key Features:
- Unified interface for all attacks
- Support for lesion-targeted and full-image attacks
- Compatible with torchattacks library
- Mask-based attack region control

Author: Multi-Metric Analysis Framework
Date: March 2026
"""

import torch
import torch.nn as nn
import torchattacks
from typing import Tuple, Optional, Dict
import numpy as np


class CheXzeroForAttack(nn.Module):
    """
    Wrapper that adapts CheXzeroWrapper output for torchattacks compatibility.

    torchattacks expects: logits (B, num_classes)
    CheXzeroWrapper outputs: probability (B,)

    This wrapper converts probability to logits for binary classification.
    """

    def __init__(self, chexzero_model):
        super().__init__()
        self.chexzero = chexzero_model

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) images

        Returns:
            logits: (B, 2) tensor
                logits[:, 0] = negative class (Normal)
                logits[:, 1] = positive class (Pneumonia)
        """
        # Get pneumonia probability from CheXzero
        prob = self.chexzero(x)  # (B,)

        # Convert to logits for binary classification
        # Using inverse sigmoid (logit function)
        # prob = sigmoid(logit) => logit = log(prob / (1 - prob))
        eps = 1e-7  # For numerical stability
        prob = torch.clamp(prob, eps, 1 - eps)

        pos_logit = torch.log(prob / (1 - prob))  # Positive class (Pneumonia)
        neg_logit = -pos_logit  # Negative class (Normal)

        logits = torch.stack([neg_logit, pos_logit], dim=1)  # (B, 2)

        return logits


def apply_mask_to_perturbation(
    perturbation: torch.Tensor,
    mask: torch.Tensor,
    attack_mode: str
) -> torch.Tensor:
    """
    Apply mask to perturbation based on attack mode.

    Args:
        perturbation: (B, C, H, W) perturbation tensor
        mask: (B, C, H, W) binary mask (1=attack region, 0=keep)
        attack_mode: 'lesion', 'random_patch', or 'full'

    Returns:
        masked_perturbation: (B, C, H, W)
    """
    if attack_mode == 'full':
        return perturbation
    else:
        # Lesion or random patch: only perturb masked region
        return perturbation * mask


class MaskedAttackWrapper:
    """
    Wrapper to apply mask constraints to torchattacks attacks.

    This enables lesion-targeted attacks with torchattacks library.
    """

    def __init__(self, attack_fn, mask: torch.Tensor, attack_mode: str = 'lesion'):
        """
        Args:
            attack_fn: torchattacks attack instance
            mask: (B, C, H, W) binary mask
            attack_mode: 'lesion', 'random_patch', or 'full'
        """
        self.attack_fn = attack_fn
        self.mask = mask
        self.attack_mode = attack_mode

    def __call__(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Generate adversarial examples with mask constraint.

        Args:
            images: (B, C, H, W) clean images
            labels: (B,) target labels

        Returns:
            adv_images: (B, C, H, W) adversarial images
        """
        # Generate unconstrained adversarial examples
        adv_images_full = self.attack_fn(images, labels)

        # Compute perturbation
        perturbation = adv_images_full - images

        # Apply mask
        masked_perturbation = apply_mask_to_perturbation(
            perturbation, self.mask, self.attack_mode
        )

        # Create masked adversarial images
        adv_images = images + masked_perturbation
        adv_images = torch.clamp(adv_images, 0, 1)

        return adv_images


def fgsm_attack_unified(
    model,
    images: torch.Tensor,
    masks: torch.Tensor,
    epsilon: float,
    attack_mode: str = 'lesion',
    use_torchattacks: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    FGSM attack with unified interface.

    Args:
        model: CheXzeroWrapper or CheXzeroForAttack
        images: (B, C, H, W) clean images
        masks: (B, C, H, W) binary masks
        epsilon: L∞ perturbation budget
        attack_mode: 'lesion', 'random_patch', or 'full'
        use_torchattacks: whether to use torchattacks library

    Returns:
        adv_images: (B, C, H, W) adversarial images
        perturbations: (B, C, H, W) perturbations
    """
    if use_torchattacks:
        # Use torchattacks.FGSM
        attack_model = CheXzeroForAttack(model)
        fgsm = torchattacks.FGSM(attack_model, eps=epsilon)

        # Labels: 1 for pneumonia (we want to minimize this)
        labels = torch.ones(images.size(0), dtype=torch.long, device=images.device)

        if attack_mode != 'full':
            # Apply mask constraint
            wrapper = MaskedAttackWrapper(fgsm, masks, attack_mode)
            adv_images = wrapper(images, labels)
        else:
            adv_images = fgsm(images, labels)

    else:
        # Use original implementation
        from rsna_attack_framework import fgsm_attack
        adv_images, _ = fgsm_attack(model, images, masks, epsilon, False, attack_mode)

    perturbations = adv_images - images
    return adv_images, perturbations


def pgd_attack_unified(
    model,
    images: torch.Tensor,
    masks: torch.Tensor,
    epsilon: float,
    alpha: float,
    num_steps: int,
    attack_mode: str = 'lesion',
    use_torchattacks: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    PGD attack with unified interface.

    Args:
        model: CheXzeroWrapper or CheXzeroForAttack
        images: (B, C, H, W) clean images
        masks: (B, C, H, W) binary masks
        epsilon: L∞ perturbation budget
        alpha: step size
        num_steps: number of iterations
        attack_mode: 'lesion', 'random_patch', or 'full'
        use_torchattacks: whether to use torchattacks library

    Returns:
        adv_images: (B, C, H, W) adversarial images
        perturbations: (B, C, H, W) perturbations
    """
    if use_torchattacks:
        # Use torchattacks.PGD
        attack_model = CheXzeroForAttack(model)
        pgd = torchattacks.PGD(attack_model, eps=epsilon, alpha=alpha, steps=num_steps)

        labels = torch.ones(images.size(0), dtype=torch.long, device=images.device)

        if attack_mode != 'full':
            wrapper = MaskedAttackWrapper(pgd, masks, attack_mode)
            adv_images = wrapper(images, labels)
        else:
            adv_images = pgd(images, labels)

    else:
        # Use original implementation (recommended for mask support)
        from rsna_attack_framework import pgd_attack
        adv_images, _ = pgd_attack(model, images, masks, epsilon, alpha, num_steps, False, attack_mode)

    perturbations = adv_images - images
    return adv_images, perturbations


def cw_attack_unified(
    model,
    images: torch.Tensor,
    masks: torch.Tensor,
    c: float = 1.0,
    kappa: float = 0,
    steps: int = 100,
    lr: float = 0.01,
    attack_mode: str = 'lesion'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Carlini-Wagner L2 attack with unified interface.

    Args:
        model: CheXzeroWrapper
        images: (B, C, H, W) clean images
        masks: (B, C, H, W) binary masks
        c: confidence parameter
        kappa: margin parameter
        steps: number of optimization steps
        lr: learning rate
        attack_mode: 'lesion', 'random_patch', or 'full'

    Returns:
        adv_images: (B, C, H, W) adversarial images
        perturbations: (B, C, H, W) perturbations
    """
    attack_model = CheXzeroForAttack(model)
    cw = torchattacks.CW(attack_model, c=c, kappa=kappa, steps=steps, lr=lr)

    # Labels: 1 for pneumonia
    labels = torch.ones(images.size(0), dtype=torch.long, device=images.device)

    if attack_mode != 'full':
        wrapper = MaskedAttackWrapper(cw, masks, attack_mode)
        adv_images = wrapper(images, labels)
    else:
        adv_images = cw(images, labels)

    perturbations = adv_images - images
    return adv_images, perturbations


def deepfool_attack_unified(
    model,
    images: torch.Tensor,
    masks: torch.Tensor,
    steps: int = 50,
    overshoot: float = 0.02,
    attack_mode: str = 'lesion'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    DeepFool attack with unified interface.

    Args:
        model: CheXzeroWrapper
        images: (B, C, H, W) clean images
        masks: (B, C, H, W) binary masks
        steps: maximum number of iterations
        overshoot: overshoot parameter
        attack_mode: 'lesion', 'random_patch', or 'full'

    Returns:
        adv_images: (B, C, H, W) adversarial images
        perturbations: (B, C, H, W) perturbations
    """
    attack_model = CheXzeroForAttack(model)
    deepfool = torchattacks.DeepFool(attack_model, steps=steps, overshoot=overshoot)

    # DeepFool doesn't use labels, but torchattacks requires them
    labels = torch.ones(images.size(0), dtype=torch.long, device=images.device)

    if attack_mode != 'full':
        wrapper = MaskedAttackWrapper(deepfool, masks, attack_mode)
        adv_images = wrapper(images, labels)
    else:
        adv_images = deepfool(images, labels)

    perturbations = adv_images - images
    return adv_images, perturbations


# Dictionary of available attacks
ATTACK_REGISTRY = {
    'fgsm': fgsm_attack_unified,
    'pgd': pgd_attack_unified,
    'cw': cw_attack_unified,
    'deepfool': deepfool_attack_unified,
}


def get_attack_function(attack_name: str):
    """
    Get attack function by name.

    Args:
        attack_name: 'fgsm', 'pgd', 'cw', or 'deepfool'

    Returns:
        attack_fn: attack function
    """
    attack_name = attack_name.lower()
    if attack_name not in ATTACK_REGISTRY:
        raise ValueError(f"Unknown attack: {attack_name}. Available: {list(ATTACK_REGISTRY.keys())}")
    return ATTACK_REGISTRY[attack_name]


def compute_metrics(
    clean_images: torch.Tensor,
    adv_images: torch.Tensor,
    clean_probs: torch.Tensor,
    adv_probs: torch.Tensor,
    perturbations: torch.Tensor,
    success_threshold: float = 0.5
) -> Dict:
    """
    Compute comprehensive attack metrics.

    Args:
        clean_images: (B, C, H, W)
        adv_images: (B, C, H, W)
        clean_probs: (B,) clean pneumonia probabilities
        adv_probs: (B,) adversarial pneumonia probabilities
        perturbations: (B, C, H, W)
        success_threshold: threshold for attack success

    Returns:
        metrics: dict with various metrics
    """
    B = clean_images.size(0)

    # Move all tensors to CPU for metric computation
    perturbations = perturbations.cpu()
    clean_probs = clean_probs.cpu()
    adv_probs = adv_probs.cpu()

    # Compute norms
    l2_norms = torch.norm(perturbations.view(B, -1), p=2, dim=1)
    linf_norms = torch.abs(perturbations).view(B, -1).max(dim=1)[0]
    l0_norms = (torch.abs(perturbations) > 1e-6).sum(dim=[1,2,3])

    # Success metric
    success = (adv_probs < success_threshold).float()

    # Confidence drop
    conf_drop = clean_probs - adv_probs

    # Attack efficiency
    efficiency = conf_drop / (l2_norms + 1e-8)

    metrics = {
        'l2_norm': l2_norms.cpu().numpy(),
        'linf_norm': linf_norms.cpu().numpy(),
        'l0_norm': l0_norms.cpu().numpy(),
        'success': success.cpu().numpy(),
        'clean_prob': clean_probs.cpu().numpy(),
        'adv_prob': adv_probs.cpu().numpy(),
        'confidence_drop': conf_drop.cpu().numpy(),
        'efficiency': efficiency.cpu().numpy(),
    }

    return metrics


if __name__ == "__main__":
    print("Unified Attack Framework Loaded")
    print("=" * 80)
    print("\nAvailable Attacks:")
    for i, (name, fn) in enumerate(ATTACK_REGISTRY.items(), 1):
        print(f"  {i}. {name.upper()}: {fn.__name__}")
    print("\nAll attacks support:")
    print("  - Lesion-targeted attacks")
    print("  - Random patch attacks")
    print("  - Full-image attacks")
    print("\nReady for multi-algorithm experiments!")
