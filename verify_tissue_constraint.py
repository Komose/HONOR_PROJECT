"""
Verify Tissue Constraint in Random Patch Generation
====================================================

This script tests the newly added tissue validity check to ensure
random patches are NOT placed on pure black background regions.

Author: Scientific Validation Test
Date: 2026-03-06
"""

import torch
from torch.utils.data import DataLoader
from rsna_attack_framework import RSNADataset
from multi_metric_attack_framework import extract_lung_region_mask, generate_equivalent_random_mask
import matplotlib.pyplot as plt
import numpy as np

print("="*80)
print("TISSUE CONSTRAINT VERIFICATION TEST")
print("="*80)

# Load dataset
dataset = RSNADataset(
    h5_path='dataset/rsna/rsna_200_samples.h5',
    lesion_info_path='dataset/rsna/rsna_200_lesion_info.json'
)

# Test on 10 samples
num_test_samples = 10
success_count = 0
intensity_stats = []

for i in range(num_test_samples):
    sample = dataset[i]
    image = sample['image']  # (C, H, W)
    lesion_mask = sample['mask']  # (C, H, W)
    patient_id = sample['patient_id']

    print(f"\n--- Sample {i+1}/{num_test_samples}: {patient_id} ---")

    # Extract lung mask
    lung_mask = extract_lung_region_mask(image)

    # Generate random patch WITH tissue constraint
    try:
        random_mask, random_info = generate_equivalent_random_mask(
            lesion_mask=lesion_mask,
            lung_mask=lung_mask,
            image=image,  # Pass image for tissue validation
            max_attempts=500
        )

        success_count += 1
        intensity = random_info['mean_intensity']
        intensity_stats.append(intensity)

        print(f"[OK] Random patch generated successfully")
        print(f"  Mean Intensity: {intensity:.4f}")
        print(f"  Tissue Valid: {random_info['tissue_valid']}")
        print(f"  In Lung Ratio: {random_info['in_lung_ratio']:.2%}")
        print(f"  Overlap Ratio: {random_info['overlap_ratio']:.2%}")
        print(f"  Attempts: {random_info['attempts']}")

        # Verify patch is NOT on black background
        if intensity > -0.5:
            print(f"  [OK] PASS: Intensity above threshold (-0.5)")
        else:
            print(f"  [FAIL] FAIL: Intensity below threshold!")

        # Extract actual pixel values in random patch region
        if random_mask.dim() == 3:
            patch_region = image * random_mask
        else:
            patch_region = image * random_mask.unsqueeze(0)

        patch_pixels = patch_region[patch_region != 0]
        if len(patch_pixels) > 0:
            print(f"  Patch pixel stats: min={patch_pixels.min():.3f}, "
                  f"max={patch_pixels.max():.3f}, mean={patch_pixels.mean():.3f}")

    except ValueError as e:
        print(f"[FAIL] Failed: {e}")

print("\n" + "="*80)
print("VERIFICATION SUMMARY")
print("="*80)
print(f"Success Rate: {success_count}/{num_test_samples} ({success_count/num_test_samples*100:.1f}%)")

if intensity_stats:
    print(f"\nIntensity Statistics:")
    print(f"  Mean: {np.mean(intensity_stats):.4f}")
    print(f"  Std: {np.std(intensity_stats):.4f}")
    print(f"  Min: {np.min(intensity_stats):.4f}")
    print(f"  Max: {np.max(intensity_stats):.4f}")
    print(f"  All above threshold (-0.5): {all(x > -0.5 for x in intensity_stats)}")

    if all(x > -0.5 for x in intensity_stats):
        print("\n[OK][OK][OK] TISSUE CONSTRAINT WORKING CORRECTLY! [OK][OK][OK]")
    else:
        print("\n[FAIL][FAIL][FAIL] WARNING: Some patches have low intensity! [FAIL][FAIL][FAIL]")

dataset.close()
print("\n" + "="*80)
