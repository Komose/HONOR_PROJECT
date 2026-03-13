"""
Diagnose Appropriate Tissue Intensity Threshold
================================================

Check actual pixel intensity distribution in random patches
to determine appropriate threshold value.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from rsna_attack_framework import RSNADataset
from multi_metric_attack_framework import extract_lung_region_mask, generate_equivalent_random_mask

print("="*80)
print("DIAGNOSING TISSUE INTENSITY THRESHOLD")
print("="*80)

# Load dataset
dataset = RSNADataset(
    h5_path='dataset/rsna/rsna_200_samples.h5',
    lesion_info_path='dataset/rsna/rsna_200_lesion_info.json'
)

# Collect intensity statistics WITHOUT tissue constraint
intensities_no_constraint = []
intensities_with_constraint = []

num_samples = 20

print("\n[Phase 1] Testing WITHOUT tissue constraint (baseline)")
print("-" * 80)

for i in range(num_samples):
    sample = dataset[i]
    image = sample['image']
    lesion_mask = sample['mask']
    patient_id = sample['patient_id']

    lung_mask = extract_lung_region_mask(image)

    try:
        # Generate random patch WITHOUT tissue constraint (pass None for image)
        random_mask, info = generate_equivalent_random_mask(
            lesion_mask=lesion_mask,
            lung_mask=lung_mask,
            image=None,  # No tissue validation
            max_attempts=500
        )

        # Manually compute intensity in this region
        if random_mask.dim() == 3:
            patch_region = image * random_mask
        else:
            patch_region = image * random_mask.unsqueeze(0)

        patch_pixels = patch_region[patch_region != 0]
        if len(patch_pixels) > 0:
            mean_intensity = patch_pixels.mean().item()
            intensities_no_constraint.append(mean_intensity)
            print(f"Sample {i+1}: intensity={mean_intensity:.4f}, "
                  f"min={patch_pixels.min():.3f}, max={patch_pixels.max():.3f}")
    except Exception as e:
        print(f"Sample {i+1}: FAILED - {e}")

print("\n" + "="*80)
print("INTENSITY STATISTICS (No Constraint)")
print("="*80)
if intensities_no_constraint:
    arr = np.array(intensities_no_constraint)
    print(f"Mean: {arr.mean():.4f}")
    print(f"Std:  {arr.std():.4f}")
    print(f"Min:  {arr.min():.4f}")
    print(f"Max:  {arr.max():.4f}")
    print(f"Q25:  {np.percentile(arr, 25):.4f}")
    print(f"Q50:  {np.percentile(arr, 50):.4f}")
    print(f"Q75:  {np.percentile(arr, 75):.4f}")

    # Suggest threshold
    threshold_candidate = arr.min() - 0.1
    print(f"\nSuggested threshold: {threshold_candidate:.4f}")
    print(f"(This ensures all current patches would pass)")

    # Test different thresholds
    print("\nDistribution by threshold:")
    for thresh in [-2.0, -1.5, -1.0, -0.5, 0.0]:
        count = (arr > thresh).sum()
        percent = count / len(arr) * 100
        print(f"  > {thresh:5.2f}: {count:2d}/{len(arr)} ({percent:5.1f}%)")

# Now test WITH tissue constraint at different thresholds
print("\n" + "="*80)
print("[Phase 2] Testing WITH tissue constraint")
print("="*80)

for threshold in [-2.0, -1.5, -1.0]:
    print(f"\n--- Testing threshold = {threshold} ---")
    success = 0
    fail = 0

    for i in range(min(10, num_samples)):
        sample = dataset[i]
        image = sample['image']
        lesion_mask = sample['mask']
        lung_mask = extract_lung_region_mask(image)

        try:
            random_mask, info = generate_equivalent_random_mask(
                lesion_mask=lesion_mask,
                lung_mask=lung_mask,
                image=image,
                max_attempts=500,
                tissue_intensity_threshold=threshold
            )
            success += 1
        except:
            fail += 1

    print(f"  Success: {success}/10, Fail: {fail}/10")

dataset.close()
print("\n" + "="*80)
print("DIAGNOSIS COMPLETE")
print("="*80)
