"""
Multi-Metric Attack Framework - Core Utilities
===============================================

This module provides core utility functions for adversarial attack experiments:

Core Functions:
- extract_lung_region_mask(): Extract lung regions from chest X-rays
- generate_equivalent_random_mask(): Generate random patches using RIGID TRANSLATION
- scale_to_l2_norm(): Scale perturbations to target L2 norm with <1% error
- compute_attack_efficiency(): Compute attack efficiency metrics

Key Features:
- Rigid translation for topology-preserving random patch generation
- Tissue intensity validation to avoid black background
- L0 norm exact alignment (zero area error)
- Automated lung region extraction

Author: HONER Project
Date: March 2026
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import h5py
import json
from tqdm import tqdm
import pandas as pd
from typing import Dict, List, Tuple, Optional
import cv2
from scipy import ndimage

# Add CheXzero to path
sys.path.insert(0, 'CheXzero')
from model import build_model
import clip


# ============================================================================
# Core Utility Functions
# ============================================================================

def extract_lung_region_mask(image: torch.Tensor, threshold_method='otsu') -> torch.Tensor:
    """
    Extract lung region from chest X-ray using thresholding.

    Args:
        image: (C, H, W) tensor, normalized chest X-ray image
        threshold_method: 'otsu' or 'fixed'

    Returns:
        lung_mask: (H, W) binary mask, 1 for lung region, 0 for background

    Note:
        This ensures random patches are placed within anatomically relevant regions.
    """
    # Convert to grayscale (average across RGB channels)
    gray = image.mean(dim=0).cpu().numpy()  # (H, W)

    # Denormalize to [0, 255] for thresholding
    # Assuming CLIP normalization: mean=[0.48, 0.46, 0.41], std=[0.27, 0.26, 0.28]
    gray = (gray * 0.27 + 0.48) * 255
    gray = np.clip(gray, 0, 255).astype(np.uint8)

    if threshold_method == 'otsu':
        # Otsu's method for automatic thresholding
        _, binary = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        # Fixed threshold (empirically chosen for chest X-rays)
        _, binary = cv2.threshold(gray, 30, 1, cv2.THRESH_BINARY)

    # Remove small noise regions (morphological opening)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    lung_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Fill holes (morphological closing)
    lung_mask = cv2.morphologyEx(lung_mask, cv2.MORPH_CLOSE, kernel)

    return torch.from_numpy(lung_mask).float()


def generate_equivalent_random_mask(
    lesion_mask: torch.Tensor,
    lung_mask: torch.Tensor,
    image: Optional[torch.Tensor] = None,
    image_size: Tuple[int, int] = (224, 224),
    max_attempts: int = 1000,
    overlap_threshold: float = 0.1,
    tissue_intensity_threshold: float = -1.0
) -> Tuple[torch.Tensor, Dict]:
    """
    Generate random mask using RIGID TRANSLATION method to preserve lesion topology.

    This function extracts the original lesion shape (including its spatial structure
    and internal gaps) and translates it as a rigid body to a new location. This
    ensures EXACT area matching and eliminates "shape concentration" as a confounding
    variable.

    CRITICAL for scientific rigor:
    - Bilateral pneumonia with two separate patches → translated as two separate patches
    - Single compact lesion → translated as single compact lesion
    - NO shape distortion, NO topology change, ONLY translation

    Args:
        lesion_mask: (C, H, W) or (H, W) binary mask of lesion regions
        lung_mask: (H, W) binary mask of lung regions (1=lung, 0=background)
        image: (C, H, W) original clean image for tissue intensity validation
        image_size: (height, width) of image
        max_attempts: maximum number of attempts to find valid placement
        overlap_threshold: maximum allowed overlap with lesion (as fraction)
        tissue_intensity_threshold: minimum mean pixel intensity to ensure patch
                                   is on actual tissue (not black background)
                                   Default -1.0 for CLIP normalized images

    Returns:
        random_mask: (C, H, W) binary mask with EXACTLY same area as lesion_mask
        metadata: dict containing placement info and statistics
                  - 'method': 'rigid_translation'
                  - 'area': exactly equals 'lesion_area' (zero error)

    Raises:
        ValueError: If cannot find valid placement after max_attempts

    Example (Bilateral Pneumonia):
        >>> # Two separate lesion patches (simulating bilateral infection)
        >>> lesion_mask = torch.zeros(3, 224, 224)
        >>> lesion_mask[:, 50:70, 30:50] = 1    # Left lung: 20×20 = 400 pixels
        >>> lesion_mask[:, 50:70, 180:200] = 1  # Right lung: 20×20 = 400 pixels
        >>> # Total: 800 pixels, but bounding box is 20×170 = 3400 pixels!
        >>> lung_mask = torch.ones(224, 224)
        >>> image = torch.randn(3, 224, 224)
        >>> random_mask, info = generate_equivalent_random_mask(lesion_mask, lung_mask, image)
        >>> print(info['area'], info['lesion_area'])  # Both exactly 800
        >>> print(info['method'])  # 'rigid_translation'
    """
    H, W = image_size

    # Handle different input shapes
    if lesion_mask.dim() == 3:
        lesion_2d = lesion_mask[0]  # Take first channel
    else:
        lesion_2d = lesion_mask

    # Calculate lesion area
    lesion_pixels = (lesion_2d > 0).sum().item()

    if lesion_pixels == 0:
        raise ValueError("Lesion mask is empty!")

    # RIGID TRANSLATION METHOD: Extract lesion shape and translate as a whole
    # Step 1: Find the minimal bounding box containing all non-zero pixels
    lesion_indices = torch.nonzero(lesion_2d, as_tuple=False)
    y_min, x_min = lesion_indices.min(dim=0)[0]
    y_max, x_max = lesion_indices.max(dim=0)[0]

    # Step 2: Extract the lesion shape slice (preserves internal topology and gaps)
    # This is the KEY step: we keep the original spatial structure intact
    lesion_shape = lesion_2d[y_min:y_max+1, x_min:x_max+1].clone()

    bbox_height = lesion_shape.shape[0]
    bbox_width = lesion_shape.shape[1]

    # Verify that extracted shape has exactly the same area
    extracted_pixels = (lesion_shape > 0).sum().item()
    assert extracted_pixels == lesion_pixels, \
        f"Shape extraction error: extracted {extracted_pixels} != original {lesion_pixels}"

    # Create valid placement region (lung - lesion)
    valid_region = (lung_mask > 0) & (lesion_2d == 0)
    valid_indices = torch.nonzero(valid_region, as_tuple=False)

    if len(valid_indices) == 0:
        raise ValueError("No valid non-lesion lung regions available!")

    # Try to find a valid placement for rigid translation
    for attempt in range(max_attempts):
        # Randomly select a new top-left corner for translation
        new_y = np.random.randint(0, max(1, H - bbox_height))
        new_x = np.random.randint(0, max(1, W - bbox_width))

        # Step 3: Create candidate mask by rigidly translating the lesion shape
        # This is CRITICAL: we paste the original shape with all its internal structure
        candidate_mask = torch.zeros_like(lesion_2d)
        candidate_mask[new_y:new_y+bbox_height, new_x:new_x+bbox_width] = lesion_shape

        # Check constraints
        # 1. Must be mostly in lung region
        candidate_pixels = (candidate_mask > 0).sum()
        if candidate_pixels == 0:
            continue
        in_lung_ratio = ((candidate_mask > 0) & (lung_mask > 0)).sum() / candidate_pixels

        # 2. Minimal overlap with lesion
        overlap = ((candidate_mask > 0) & (lesion_2d > 0)).sum().item()
        overlap_ratio = overlap / lesion_pixels

        # 3. CRITICAL: Tissue validity check - ensure patch is on actual tissue, not black background
        tissue_valid = True
        mean_intensity = 0.0
        if image is not None:
            # Extract region from original image
            if image.dim() == 3:
                region = image[:, new_y:new_y+bbox_height, new_x:new_x+bbox_width]
            else:
                region = image[new_y:new_y+bbox_height, new_x:new_x+bbox_width]

            # Compute mean pixel intensity only for non-zero mask pixels (tissue check)
            # This ensures we check actual tissue, not the gaps within the bounding box
            mask_region = candidate_mask[new_y:new_y+bbox_height, new_x:new_x+bbox_width]
            if (mask_region > 0).sum() > 0:
                mean_intensity = region[:, mask_region > 0].mean().item() if image.dim() == 3 else region[mask_region > 0].mean().item()
            else:
                mean_intensity = 0.0

            # Check if intensity is above threshold (not pure black background)
            tissue_valid = mean_intensity > tissue_intensity_threshold

        if in_lung_ratio > 0.8 and overlap_ratio < overlap_threshold and tissue_valid:
            # Valid placement found!
            # Expand to 3 channels
            random_mask = candidate_mask.unsqueeze(0).expand(3, -1, -1)

            # CRITICAL: Assert L0 alignment to ensure scientific rigor
            # With rigid translation, area MUST be exactly equal (no rounding error)
            random_area = int((random_mask[0] > 0).sum().item())
            area_error = abs(random_area - lesion_pixels)
            area_error_ratio = area_error / lesion_pixels if lesion_pixels > 0 else 0

            # With rigid translation, area_error_ratio should be exactly 0
            assert area_error_ratio < 0.001, \
                f"L0 ALIGNMENT FAILED! Random area={random_area}, Lesion area={lesion_pixels}, Error={area_error_ratio:.1%}"

            metadata = {
                'area': random_area,
                'width': bbox_width,
                'height': bbox_height,
                'x': int(new_x),
                'y': int(new_y),
                'lesion_area': int(lesion_pixels),
                'area_match': area_error == 0,  # Should be exactly 0 with rigid translation
                'area_error_ratio': float(area_error_ratio),
                'overlap_ratio': float(overlap_ratio),
                'in_lung_ratio': float(in_lung_ratio),
                'mean_intensity': float(mean_intensity),
                'tissue_valid': tissue_valid,
                'tissue_threshold': tissue_intensity_threshold,
                'attempts': attempt + 1,
                'method': 'rigid_translation'  # Document the method used
            }

            return random_mask, metadata

    # Failed to find valid placement
    raise ValueError(
        f"Could not find valid random patch placement after {max_attempts} attempts. "
        f"Lesion area: {lesion_pixels} pixels, Bbox: {bbox_height}×{bbox_width}"
    )


def scale_to_l2_norm(
    perturbation: torch.Tensor,
    target_l2: float,
    mask: Optional[torch.Tensor] = None,
    max_iterations: int = 10,
    tolerance: float = 0.01
) -> Tuple[torch.Tensor, Dict]:
    """
    Scale perturbation to match target L2 norm with <1% error.

    Args:
        perturbation: (C, H, W) perturbation tensor
        target_l2: target L2 norm value
        mask: optional (C, H, W) mask to compute norm only in masked region
        max_iterations: maximum refinement iterations
        tolerance: acceptable relative error (default 0.01 = 1%)

    Returns:
        scaled_perturbation: perturbation scaled to target L2
        metadata: dict with scaling statistics

    Algorithm:
        1. Compute current L2 norm (with mask if provided)
        2. Initial scaling: δ_scaled = δ * (target_L2 / current_L2)
        3. Iterative refinement if error > tolerance

    Example:
        >>> delta = torch.randn(3, 224, 224) * 0.05
        >>> scaled_delta, info = scale_to_l2_norm(delta, target_l2=10.0, tolerance=0.01)
        >>> print(f"Error: {info['final_error']:.4f}")  # < 0.01
    """
    if mask is not None:
        # Compute L2 norm only in masked region
        masked_pert = perturbation * mask
        current_l2 = torch.norm(masked_pert).item()
    else:
        current_l2 = torch.norm(perturbation).item()

    if current_l2 == 0:
        raise ValueError("Cannot scale zero perturbation!")

    # Initial scaling factor
    scale_factor = target_l2 / current_l2
    scaled_pert = perturbation * scale_factor

    # Iterative refinement to ensure <1% error
    iteration_history = [{'iteration': 0, 'l2': current_l2 * scale_factor, 'error': abs(1 - scale_factor)}]

    for iteration in range(1, max_iterations + 1):
        # Recompute L2
        if mask is not None:
            current_l2_scaled = torch.norm(scaled_pert * mask).item()
        else:
            current_l2_scaled = torch.norm(scaled_pert).item()

        relative_error = abs(current_l2_scaled - target_l2) / target_l2
        iteration_history.append({'iteration': iteration, 'l2': current_l2_scaled, 'error': relative_error})

        if relative_error < tolerance:
            # Converged!
            metadata = {
                'initial_l2': current_l2,
                'target_l2': target_l2,
                'final_l2': current_l2_scaled,
                'scale_factor': scale_factor,
                'final_error': relative_error,
                'iterations': iteration,
                'converged': True,
                'history': iteration_history
            }
            return scaled_pert, metadata

        # Refine scale factor
        correction = target_l2 / current_l2_scaled
        scaled_pert = scaled_pert * correction
        scale_factor *= correction

    # Max iterations reached
    if mask is not None:
        final_l2 = torch.norm(scaled_pert * mask).item()
    else:
        final_l2 = torch.norm(scaled_pert).item()

    final_error = abs(final_l2 - target_l2) / target_l2

    metadata = {
        'initial_l2': current_l2,
        'target_l2': target_l2,
        'final_l2': final_l2,
        'scale_factor': scale_factor,
        'final_error': final_error,
        'iterations': max_iterations,
        'converged': False,
        'history': iteration_history
    }

    if final_error >= tolerance:
        print(f"WARNING: L2 scaling did not converge within tolerance ({final_error:.4f} > {tolerance})")

    return scaled_pert, metadata


def compute_attack_efficiency(
    clean_prob: float,
    adv_prob: float,
    l2_norm: float,
    success_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute attack efficiency metrics.

    Metrics:
        - Confidence Drop: clean_prob - adv_prob
        - Attack Efficiency: (Confidence Drop) / L2 norm
        - Success: whether adv_prob < success_threshold

    Args:
        clean_prob: model confidence on clean image
        adv_prob: model confidence on adversarial image
        l2_norm: L2 norm of perturbation
        success_threshold: threshold for attack success (default 0.5)

    Returns:
        metrics: dict with efficiency metrics

    Example:
        >>> efficiency = compute_attack_efficiency(
        ...     clean_prob=0.85, adv_prob=0.12, l2_norm=5.0
        ... )
        >>> print(efficiency)
        {'confidence_drop': 0.73, 'efficiency': 0.146, 'success': True}
    """
    confidence_drop = clean_prob - adv_prob

    if l2_norm > 0:
        efficiency = confidence_drop / l2_norm
    else:
        efficiency = 0.0

    success = adv_prob < success_threshold

    return {
        'confidence_drop': float(confidence_drop),
        'efficiency': float(efficiency),
        'efficiency_if_success': float(efficiency if success else 0.0),
        'success': bool(success),
        'clean_prob': float(clean_prob),
        'adv_prob': float(adv_prob),
        'l2_norm': float(l2_norm)
    }
