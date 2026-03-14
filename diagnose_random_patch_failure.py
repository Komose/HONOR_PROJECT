"""
Diagnose Random Patch Attack Failure
=====================================

Check why random_patch attacks are producing L2=0
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from rsna_attack_framework import RSNADataset, CheXzeroWrapper
from multi_metric_attack_framework import extract_lung_region_mask, generate_equivalent_random_mask

print("="*80)
print("DIAGNOSING RANDOM PATCH ATTACK FAILURE")
print("="*80)

# Load patient 2 (bf842c92-c6ed-4be4-b188-c05ec8bf8dc2)
h5_path = 'dataset/rsna/rsna_200_samples.h5'
lesion_info_path = 'dataset/rsna/rsna_200_lesion_info.json'
dataset = RSNADataset(h5_path, lesion_info_path)

# Get patient 2 (index 1)
sample = dataset[1]
patient_id = sample['patient_id']
image = sample['image']  # (3, 224, 224)
lesion_mask = sample['mask']  # (224, 224)

print(f"\nPatient ID: {patient_id}")
print(f"Image shape: {image.shape}")
print(f"Lesion mask shape: {lesion_mask.shape}")

# Check lesion mask
lesion_pixels = (lesion_mask > 0).sum().item()
print(f"\nLesion mask non-zero pixels: {lesion_pixels}")

# Generate lung mask
lung_mask = extract_lung_region_mask(image)
print(f"Lung mask non-zero pixels: {(lung_mask > 0).sum().item()}")

# Generate random patch
print("\nGenerating random patch...")
try:
    random_mask, info = generate_equivalent_random_mask(
        lesion_mask=lesion_mask,
        lung_mask=lung_mask,
        image=image,
        max_attempts=500
    )
    print("[OK] Random patch generated successfully!")
    print(f"Random mask shape: {random_mask.shape}")
    print(f"Random mask non-zero pixels: {(random_mask[0] > 0).sum().item()}")
    print(f"\nMetadata: {info}")

except Exception as e:
    print(f"[ERROR] Random patch generation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Check pixel intensities in each region
print("\n" + "="*80)
print("PIXEL INTENSITY ANALYSIS")
print("="*80)

# Lesion region (use first channel of mask)
lesion_mask_2d = lesion_mask[0] if lesion_mask.dim() == 3 else lesion_mask
lesion_region_pixels = image[:, lesion_mask_2d > 0]  # (3, N)
lesion_mean = lesion_region_pixels.mean().item()
lesion_std = lesion_region_pixels.std().item()
lesion_min = lesion_region_pixels.min().item()
lesion_max = lesion_region_pixels.max().item()

print(f"\nLesion region:")
print(f"  Mean intensity: {lesion_mean:.4f}")
print(f"  Std:            {lesion_std:.4f}")
print(f"  Min/Max:        {lesion_min:.4f} / {lesion_max:.4f}")

# Random patch region
random_region_pixels = image[:, random_mask[0] > 0]  # (3, N)
random_mean = random_region_pixels.mean().item()
random_std = random_region_pixels.std().item()
random_min = random_region_pixels.min().item()
random_max = random_region_pixels.max().item()

print(f"\nRandom patch region:")
print(f"  Mean intensity: {random_mean:.4f}")
print(f"  Std:            {random_std:.4f}")
print(f"  Min/Max:        {random_min:.4f} / {random_max:.4f}")

# Full image
full_mean = image.mean().item()
full_std = image.std().item()

print(f"\nFull image:")
print(f"  Mean intensity: {full_mean:.4f}")
print(f"  Std:            {full_std:.4f}")

# Check if random region is mostly black
if random_mean < 0.1:
    print(f"\n[WARNING]  WARNING: Random patch region is very dark (mean={random_mean:.4f})")
    print("   This could explain why attacks fail!")

# Visualize
print("\n" + "="*80)
print("GENERATING VISUALIZATION")
print("="*80)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Original image
axes[0, 0].imshow(image[0].numpy(), cmap='gray')
axes[0, 0].set_title('Original Image', fontweight='bold')
axes[0, 0].axis('off')

# Lesion mask
axes[0, 1].imshow(lesion_mask.numpy(), cmap='Reds', alpha=0.7)
axes[0, 1].set_title(f'Lesion Mask ({lesion_pixels} pixels)', fontweight='bold')
axes[0, 1].axis('off')

# Random mask
axes[0, 2].imshow(random_mask[0].numpy(), cmap='Blues', alpha=0.7)
axes[0, 2].set_title(f'Random Patch ({(random_mask[0]>0).sum().item()} pixels)', fontweight='bold')
axes[0, 2].axis('off')

# Image with lesion overlay
img_with_lesion = image[0].clone()
img_with_lesion_rgb = torch.stack([img_with_lesion, img_with_lesion, img_with_lesion], dim=0)
img_with_lesion_rgb[0, lesion_mask > 0] = 1.0  # Red
axes[1, 0].imshow(img_with_lesion_rgb.permute(1, 2, 0).numpy())
axes[1, 0].set_title('Image + Lesion (Red)', fontweight='bold')
axes[1, 0].axis('off')

# Image with random patch overlay
img_with_random = image[0].clone()
img_with_random_rgb = torch.stack([img_with_random, img_with_random, img_with_random], dim=0)
img_with_random_rgb[2, random_mask[0] > 0] = 1.0  # Blue
axes[1, 1].imshow(img_with_random_rgb.permute(1, 2, 0).numpy())
axes[1, 1].set_title('Image + Random Patch (Blue)', fontweight='bold')
axes[1, 1].axis('off')

# Both masks overlay
img_with_both = image[0].clone()
img_with_both_rgb = torch.stack([img_with_both, img_with_both, img_with_both], dim=0)
img_with_both_rgb[0, lesion_mask > 0] = 1.0  # Red
img_with_both_rgb[2, random_mask[0] > 0] = 1.0  # Blue
axes[1, 2].imshow(img_with_both_rgb.permute(1, 2, 0).numpy())
axes[1, 2].set_title('Both Masks (Red=Lesion, Blue=Random)', fontweight='bold')
axes[1, 2].axis('off')

plt.tight_layout()
save_path = 'results/random_patch_diagnosis.png'
plt.savefig(save_path, dpi=200, bbox_inches='tight')
print(f"[OK] Saved visualization: {save_path}")

# Check mask validity for C&W attack
print("\n" + "="*80)
print("MASK VALIDITY CHECK FOR C&W ATTACK")
print("="*80)

# Masks are already 3D
lesion_mask_3d = lesion_mask  # Already (3, 224, 224)
random_mask_3d = random_mask  # Already (3, 224, 224)

print(f"\nLesion mask 3D shape: {lesion_mask_3d.shape}")
print(f"Lesion mask 3D non-zero: {(lesion_mask_3d > 0).sum().item()}")

print(f"\nRandom mask 3D shape: {random_mask_3d.shape}")
print(f"Random mask 3D non-zero: {(random_mask_3d > 0).sum().item()}")

# Test if masks are all zeros (which would cause attack failure)
if (random_mask_3d > 0).sum() == 0:
    print("\n[ERROR] CRITICAL: Random mask is all zeros!")
else:
    print("\n[OK] Random mask has non-zero values")

# Check if random mask is being applied correctly
print("\n" + "="*80)
print("SIMULATING ATTACK MASK APPLICATION")
print("="*80)

# Simulate gradient masking (as done in C&W)
test_gradient = torch.randn_like(image)  # Random gradient
masked_gradient_lesion = test_gradient * lesion_mask_3d
masked_gradient_random = test_gradient * random_mask_3d

print(f"\nOriginal gradient norm: {test_gradient.norm().item():.4f}")
print(f"Lesion-masked gradient norm: {masked_gradient_lesion.norm().item():.4f}")
print(f"Random-masked gradient norm: {masked_gradient_random.norm().item():.4f}")

if masked_gradient_random.norm().item() == 0:
    print("\n[ERROR] CRITICAL: Random-masked gradient is zero!")
    print("   This would cause C&W optimization to fail completely.")
else:
    print(f"\n[OK] Random-masked gradient is non-zero")
    ratio = masked_gradient_random.norm().item() / masked_gradient_lesion.norm().item()
    print(f"   Random/Lesion gradient norm ratio: {ratio:.4f}")

print("\n" + "="*80)
print("DIAGNOSIS COMPLETE")
print("="*80)
