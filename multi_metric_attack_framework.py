"""
Multi-Metric Sensitivity Analysis Framework
============================================

This module implements a comprehensive framework for evaluating adversarial
attack sensitivity across three key metrics:
- Experiment A: Fixed L∞ (Intensity Alignment)
- Experiment B: Fixed L0 (Area Alignment)
- Experiment C: Fixed L2 (Energy Alignment)

Key Features:
- Random non-lesion patch generation with lung region constraints
- L2-norm scaling with <1% error tolerance
- Attack efficiency metrics
- Automated reporting and visualization

Author: Multi-Metric Analysis Framework
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
    Generate a random rectangular mask with area equal to lesion_mask,
    placed in non-lesion lung regions WITH TISSUE VALIDITY CHECK.

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
                                   (empirically validated on RSNA dataset)

    Returns:
        random_mask: (C, H, W) binary mask of random patch
        metadata: dict containing placement info and statistics

    Raises:
        ValueError: If cannot find valid placement after max_attempts

    Example:
        >>> lesion_mask = torch.zeros(3, 224, 224)
        >>> lesion_mask[:, 50:100, 80:150] = 1  # 50×70 = 3500 pixels
        >>> lung_mask = torch.ones(224, 224)  # Simplified full lung
        >>> image = torch.randn(3, 224, 224)  # Clean image
        >>> random_mask, info = generate_equivalent_random_mask(lesion_mask, lung_mask, image)
        >>> print(info)
        {'area': 3500, 'width': 70, 'height': 50, 'x': 120, 'y': 85,
         'mean_intensity': 0.15, 'tissue_valid': True, ...}
    """
    H, W = image_size

    # Handle different input shapes
    if lesion_mask.dim() == 3:
        lesion_2d = lesion_mask[0]  # Take first channel
    else:
        lesion_2d = lesion_mask

    # Calculate lesion area and bounding box dimensions
    lesion_pixels = (lesion_2d > 0).sum().item()

    if lesion_pixels == 0:
        raise ValueError("Lesion mask is empty!")

    # Find the actual bounding box of lesions to match dimensions
    lesion_indices = torch.nonzero(lesion_2d, as_tuple=False)
    y_min, x_min = lesion_indices.min(dim=0)[0]
    y_max, x_max = lesion_indices.max(dim=0)[0]

    bbox_height = int(y_max - y_min + 1)
    bbox_width = int(x_max - x_min + 1)

    # Create valid placement region (lung - lesion)
    valid_region = (lung_mask > 0) & (lesion_2d == 0)
    valid_indices = torch.nonzero(valid_region, as_tuple=False)

    if len(valid_indices) == 0:
        raise ValueError("No valid non-lesion lung regions available!")

    # Try to find a valid placement
    for attempt in range(max_attempts):
        # Randomly select a top-left corner
        y = np.random.randint(0, max(1, H - bbox_height))
        x = np.random.randint(0, max(1, W - bbox_width))

        # Create candidate mask
        candidate_mask = torch.zeros_like(lesion_2d)
        candidate_mask[y:y+bbox_height, x:x+bbox_width] = 1

        # Check constraints
        # 1. Must be mostly in lung region
        in_lung_ratio = ((candidate_mask > 0) & (lung_mask > 0)).sum() / (candidate_mask > 0).sum()

        # 2. Minimal overlap with lesion
        overlap = ((candidate_mask > 0) & (lesion_2d > 0)).sum().item()
        overlap_ratio = overlap / lesion_pixels

        # 3. CRITICAL: Tissue validity check - ensure patch is on actual tissue, not black background
        tissue_valid = True
        mean_intensity = 0.0
        if image is not None:
            # Extract region from original image
            if image.dim() == 3:
                region = image[:, y:y+bbox_height, x:x+bbox_width]
            else:
                region = image[y:y+bbox_height, x:x+bbox_width]

            # Compute mean pixel intensity in this region
            mean_intensity = region.mean().item()

            # Check if intensity is above threshold (not pure black background)
            tissue_valid = mean_intensity > tissue_intensity_threshold

        if in_lung_ratio > 0.8 and overlap_ratio < overlap_threshold and tissue_valid:
            # Valid placement found!
            # Expand to 3 channels
            random_mask = candidate_mask.unsqueeze(0).expand(3, -1, -1)

            metadata = {
                'area': int((random_mask[0] > 0).sum().item()),
                'width': bbox_width,
                'height': bbox_height,
                'x': int(x),
                'y': int(y),
                'lesion_area': int(lesion_pixels),
                'area_match': abs((random_mask[0] > 0).sum().item() - lesion_pixels) < 10,
                'overlap_ratio': float(overlap_ratio),
                'in_lung_ratio': float(in_lung_ratio),
                'mean_intensity': float(mean_intensity),
                'tissue_valid': tissue_valid,
                'tissue_threshold': tissue_intensity_threshold,
                'attempts': attempt + 1
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


# ============================================================================
# Experiment-Specific Functions
# ============================================================================

def run_experiment_a_fixed_linf(
    model,
    dataloader,
    attack_fn,
    epsilon_values: List[float],
    device,
    output_dir: str,
    attack_name: str = "PGD"
) -> pd.DataFrame:
    """
    Experiment A: Fixed L∞ (Intensity Alignment)

    Compare lesion attack vs full attack with identical ε constraints.

    Args:
        model: CheXzeroWrapper instance
        dataloader: DataLoader for RSNA dataset
        attack_fn: attack function (e.g., pgd_attack)
        epsilon_values: list of ε values to test (e.g., [4/255, 8/255])
        device: torch device
        output_dir: directory to save results
        attack_name: name of attack (for logging)

    Returns:
        results_df: DataFrame with comparative results
    """
    print("\n" + "="*80)
    print("EXPERIMENT A: Fixed L∞ (Intensity Alignment)")
    print("="*80)
    print(f"Attack: {attack_name}")
    print(f"Epsilon values: {epsilon_values}")
    print()

    all_results = []

    for eps in epsilon_values:
        print(f"\n--- Testing ε = {eps:.4f} ({eps*255:.1f}/255) ---\n")

        for attack_mode in ['lesion', 'full']:
            mode_results = []

            for batch in tqdm(dataloader, desc=f"{attack_name}-{attack_mode}-eps{eps:.4f}"):
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                patient_ids = batch['patient_id']

                # Run attack
                adv_images, perturbations = attack_fn(
                    model=model,
                    images=images,
                    masks=masks,
                    attack_mode=attack_mode,
                    epsilon=eps,
                    num_steps=40,  # Use PGD-40 for best results
                    alpha=eps/4
                )

                # Evaluate
                with torch.no_grad():
                    clean_probs = model(images).cpu().numpy()
                    adv_probs = model(adv_images).cpu().numpy()

                # Compute metrics for each sample
                for i in range(len(patient_ids)):
                    pert = perturbations[i]
                    mask = masks[i]

                    # Compute norms
                    l2_norm = torch.norm(pert).item()
                    linf_norm = torch.abs(pert).max().item()
                    l0_norm = (torch.abs(pert) > 1e-6).sum().item()

                    # Compute efficiency
                    efficiency_metrics = compute_attack_efficiency(
                        clean_prob=clean_probs[i],
                        adv_prob=adv_probs[i],
                        l2_norm=l2_norm
                    )

                    mode_results.append({
                        'experiment': 'A_Fixed_Linf',
                        'patient_id': patient_ids[i],
                        'attack': attack_name,
                        'mode': attack_mode,
                        'epsilon': eps,
                        **efficiency_metrics,
                        'l0_norm': l0_norm,
                        'linf_norm': linf_norm
                    })

            all_results.extend(mode_results)

    results_df = pd.DataFrame(all_results)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(os.path.join(output_dir, 'experiment_a_fixed_linf.csv'), index=False)

    # Print summary
    print("\n" + "="*80)
    print("EXPERIMENT A SUMMARY")
    print("="*80)
    summary = results_df.groupby(['epsilon', 'mode']).agg({
        'success': 'mean',
        'confidence_drop': 'mean',
        'efficiency': 'mean',
        'l2_norm': 'mean',
        'linf_norm': 'mean'
    }).round(4)
    print(summary)
    print()

    return results_df


def run_experiment_b_fixed_l0(
    model,
    dataloader,
    attack_fn,
    device,
    output_dir: str,
    attack_name: str = "PGD",
    epsilon: float = 8/255
) -> pd.DataFrame:
    """
    Experiment B: Fixed L0 (Area Alignment)

    Compare three attack modes with identical number of modified pixels:
    1. Lesion Attack: real lesion bounding boxes
    2. Random Patch Attack: equal-area random non-lesion patches
    3. Full Attack: entire image (for reference)

    This experiment isolates the effect of "semantic specificity" by controlling
    for the number of pixels modified.

    Args:
        model: CheXzeroWrapper instance
        dataloader: DataLoader for RSNA dataset
        attack_fn: attack function (e.g., pgd_attack)
        device: torch device
        output_dir: directory to save results
        attack_name: name of attack
        epsilon: L∞ constraint for attack

    Returns:
        results_df: DataFrame with comparative results
    """
    print("\n" + "="*80)
    print("EXPERIMENT B: Fixed L0 (Area Alignment)")
    print("="*80)
    print(f"Attack: {attack_name}, ε = {epsilon:.4f}")
    print()

    all_results = []
    random_mask_failures = 0

    for batch in tqdm(dataloader, desc="Exp-B"):
        images = batch['image'].to(device)
        lesion_masks = batch['mask'].to(device)
        patient_ids = batch['patient_id']

        batch_size = images.size(0)

        for i in range(batch_size):
            image = images[i:i+1]
            lesion_mask = lesion_masks[i:i+1]
            patient_id = patient_ids[i]

            # Extract lung region for this image
            lung_mask = extract_lung_region_mask(image[0])
            lung_mask = lung_mask.to(device)

            # Generate random patch with equal area
            try:
                random_mask, random_info = generate_equivalent_random_mask(
                    lesion_mask=lesion_mask[0],
                    lung_mask=lung_mask,
                    image=image[0],  # CRITICAL: Pass original image for tissue validation
                    max_attempts=500
                )
                random_mask = random_mask.unsqueeze(0).to(device)

                # Verify tissue validity
                if not random_info['tissue_valid']:
                    print(f"WARNING: Random patch tissue validation failed for {patient_id} "
                          f"(intensity={random_info['mean_intensity']:.3f})")
            except ValueError as e:
                print(f"WARNING: Failed to generate random mask for {patient_id}: {e}")
                random_mask_failures += 1
                continue

            # Run three attacks: lesion, random patch, full
            attack_configs = [
                ('lesion', lesion_mask),
                ('random_patch', random_mask),
                ('full', torch.ones_like(lesion_mask))
            ]

            results_for_sample = {}

            for mode_name, mask in attack_configs:
                # Determine attack mode
                if mode_name == 'full':
                    attack_mode = 'full'
                else:
                    attack_mode = 'lesion'  # Use lesion attack logic with custom mask

                # Run attack
                adv_image, perturbation = attack_fn(
                    model=model,
                    images=image,
                    masks=mask,
                    attack_mode=attack_mode,
                    epsilon=epsilon,
                    num_steps=40,
                    alpha=epsilon/4
                )

                # Evaluate
                with torch.no_grad():
                    clean_prob = model(image).item()
                    adv_prob = model(adv_image).item()

                # Compute metrics
                pert = perturbation[0]
                l2_norm = torch.norm(pert).item()
                linf_norm = torch.abs(pert).max().item()
                l0_norm = (torch.abs(pert) > 1e-6).sum().item()

                efficiency_metrics = compute_attack_efficiency(
                    clean_prob=clean_prob,
                    adv_prob=adv_prob,
                    l2_norm=l2_norm
                )

                results_for_sample[mode_name] = {
                    'experiment': 'B_Fixed_L0',
                    'patient_id': patient_id,
                    'attack': attack_name,
                    'mode': mode_name,
                    'epsilon': epsilon,
                    **efficiency_metrics,
                    'l0_norm': l0_norm,
                    'linf_norm': linf_norm,
                    'mask_area': (mask[0, 0] > 0).sum().item()
                }

                if mode_name == 'random_patch':
                    results_for_sample[mode_name]['random_mask_info'] = str(random_info)

            # Verify L0 matching between lesion and random patch
            lesion_l0 = results_for_sample['lesion']['mask_area']
            random_l0 = results_for_sample['random_patch']['mask_area']
            l0_match = abs(lesion_l0 - random_l0) / lesion_l0 < 0.1  # Within 10%

            for mode_name in results_for_sample:
                results_for_sample[mode_name]['l0_area_match'] = l0_match
                all_results.append(results_for_sample[mode_name])

    results_df = pd.DataFrame(all_results)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(os.path.join(output_dir, 'experiment_b_fixed_l0.csv'), index=False)

    # Print summary
    print("\n" + "="*80)
    print("EXPERIMENT B SUMMARY")
    print("="*80)
    print(f"Random mask generation failures: {random_mask_failures}")
    summary = results_df.groupby('mode').agg({
        'success': 'mean',
        'confidence_drop': 'mean',
        'efficiency': 'mean',
        'l2_norm': 'mean',
        'l0_norm': 'mean',
        'mask_area': 'mean'
    }).round(4)
    print(summary)
    print()

    return results_df


def run_experiment_c_fixed_l2(
    model,
    dataloader,
    attack_fn,
    device,
    output_dir: str,
    attack_name: str = "PGD",
    epsilon: float = 8/255
) -> pd.DataFrame:
    """
    Experiment C: Fixed L2 (Energy Alignment)

    Compare lesion attack vs full attack with identical L2 perturbation energy:
    1. Run lesion attack → record L2 norm
    2. Run full attack → scale perturbation to match lesion L2
    3. Compare effectiveness under equal L2 budget

    Args:
        model: CheXzeroWrapper instance
        dataloader: DataLoader for RSNA dataset
        attack_fn: attack function (e.g., pgd_attack)
        device: torch device
        output_dir: directory to save results
        attack_name: name of attack
        epsilon: L∞ constraint for initial attack generation

    Returns:
        results_df: DataFrame with comparative results
    """
    print("\n" + "="*80)
    print("EXPERIMENT C: Fixed L2 (Energy Alignment)")
    print("="*80)
    print(f"Attack: {attack_name}, ε = {epsilon:.4f}")
    print()

    all_results = []
    scaling_errors = []

    for batch in tqdm(dataloader, desc="Exp-C"):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        patient_ids = batch['patient_id']

        batch_size = images.size(0)

        for i in range(batch_size):
            image = images[i:i+1]
            mask = masks[i:i+1]
            patient_id = patient_ids[i]

            # Step 1: Run lesion attack
            adv_lesion, pert_lesion = attack_fn(
                model=model,
                images=image,
                masks=mask,
                attack_mode='lesion',
                epsilon=epsilon,
                num_steps=40,
                alpha=epsilon/4
            )

            lesion_l2 = torch.norm(pert_lesion[0]).item()

            with torch.no_grad():
                clean_prob = model(image).item()
                adv_lesion_prob = model(adv_lesion).item()

            lesion_metrics = compute_attack_efficiency(
                clean_prob=clean_prob,
                adv_prob=adv_lesion_prob,
                l2_norm=lesion_l2
            )

            all_results.append({
                'experiment': 'C_Fixed_L2',
                'patient_id': patient_id,
                'attack': attack_name,
                'mode': 'lesion',
                'epsilon': epsilon,
                **lesion_metrics,
                'l0_norm': (torch.abs(pert_lesion[0]) > 1e-6).sum().item(),
                'linf_norm': torch.abs(pert_lesion[0]).max().item(),
                'target_l2': lesion_l2,
                'l2_scaling_error': 0.0,
                'l2_converged': True
            })

            # Step 2: Run full attack
            adv_full, pert_full = attack_fn(
                model=model,
                images=image,
                masks=torch.ones_like(mask),
                attack_mode='full',
                epsilon=epsilon,
                num_steps=40,
                alpha=epsilon/4
            )

            full_l2_before = torch.norm(pert_full[0]).item()

            # Step 3: Scale full attack perturbation to match lesion L2
            pert_full_scaled, scaling_info = scale_to_l2_norm(
                perturbation=pert_full[0],
                target_l2=lesion_l2,
                tolerance=0.01,
                max_iterations=10
            )

            scaling_errors.append(scaling_info['final_error'])

            # Create scaled adversarial image
            adv_full_scaled = image + pert_full_scaled.unsqueeze(0)
            adv_full_scaled = torch.clamp(adv_full_scaled, 0, 1)

            with torch.no_grad():
                adv_full_scaled_prob = model(adv_full_scaled).item()

            full_metrics = compute_attack_efficiency(
                clean_prob=clean_prob,
                adv_prob=adv_full_scaled_prob,
                l2_norm=scaling_info['final_l2']
            )

            all_results.append({
                'experiment': 'C_Fixed_L2',
                'patient_id': patient_id,
                'attack': attack_name,
                'mode': 'full_scaled',
                'epsilon': epsilon,
                **full_metrics,
                'l0_norm': (torch.abs(pert_full_scaled) > 1e-6).sum().item(),
                'linf_norm': torch.abs(pert_full_scaled).max().item(),
                'target_l2': lesion_l2,
                'l2_before_scaling': full_l2_before,
                'l2_scaling_error': scaling_info['final_error'],
                'l2_converged': scaling_info['converged'],
                'scaling_iterations': scaling_info['iterations']
            })

    results_df = pd.DataFrame(all_results)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(os.path.join(output_dir, 'experiment_c_fixed_l2.csv'), index=False)

    # Print summary
    print("\n" + "="*80)
    print("EXPERIMENT C SUMMARY")
    print("="*80)
    print(f"Average L2 scaling error: {np.mean(scaling_errors):.6f}")
    print(f"Max L2 scaling error: {np.max(scaling_errors):.6f}")
    print(f"Convergence rate: {results_df['l2_converged'].mean()*100:.1f}%")
    print()

    summary = results_df.groupby('mode').agg({
        'success': 'mean',
        'confidence_drop': 'mean',
        'efficiency': 'mean',
        'l2_norm': 'mean',
        'linf_norm': 'mean'
    }).round(4)
    print(summary)
    print()

    return results_df


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("Multi-Metric Attack Framework Module Loaded")
    print("=" * 80)
    print("\nKey Functions:")
    print("  - extract_lung_region_mask()")
    print("  - generate_equivalent_random_mask()")
    print("  - scale_to_l2_norm()")
    print("  - compute_attack_efficiency()")
    print("  - run_experiment_a_fixed_linf()")
    print("  - run_experiment_b_fixed_l0()")
    print("  - run_experiment_c_fixed_l2()")
    print("\nReady for multi-metric sensitivity analysis!")
