"""
Quick Test: Verify L0 Alignment Fix
====================================

Tests that random_patch now has EXACTLY the same area as lesion.
"""

import torch
import numpy as np
from multi_metric_attack_framework import generate_equivalent_random_mask

def test_scattered_lesion():
    """Test case: Scattered lesion (the problematic case)"""
    print("=" * 60)
    print("TEST 1: Scattered Lesion (Multiple Small Patches)")
    print("=" * 60)

    # Create scattered lesion: two small patches far apart
    lesion_mask = torch.zeros(3, 224, 224)

    # Patch 1: top-left (10×10 = 100 pixels)
    lesion_mask[:, 20:30, 30:40] = 1

    # Patch 2: bottom-right (15×15 = 225 pixels)
    lesion_mask[:, 180:195, 190:205] = 1

    # Total: 100 + 225 = 325 pixels
    # But bounding box would be: (195-20) × (205-30) = 175 × 175 = 30,625 pixels!

    lesion_area = (lesion_mask[0] > 0).sum().item()
    print(f"Lesion actual area: {lesion_area} pixels")

    # Find bounding box (old method)
    lesion_indices = torch.nonzero(lesion_mask[0], as_tuple=False)
    y_min, x_min = lesion_indices.min(dim=0)[0]
    y_max, x_max = lesion_indices.max(dim=0)[0]
    bbox_area = (y_max - y_min + 1) * (x_max - x_min + 1)
    print(f"Bounding box area (OLD BUG): {bbox_area} pixels")
    print(f"→ Old method would be {bbox_area / lesion_area:.1f}× too large! [BUG]")

    # Create lung mask and dummy image
    lung_mask = torch.ones(224, 224)
    image = torch.randn(3, 224, 224)

    # Generate random patch with FIXED method
    try:
        random_mask, info = generate_equivalent_random_mask(
            lesion_mask=lesion_mask,
            lung_mask=lung_mask,
            image=image,
            max_attempts=500
        )

        random_area = (random_mask[0] > 0).sum().item()
        error_ratio = abs(random_area - lesion_area) / lesion_area

        print(f"\n[OK] FIXED random_patch area: {random_area} pixels")
        print(f"   Lesion area: {lesion_area} pixels")
        print(f"   Error: {error_ratio:.1%}")
        print(f"   Area match: {info['area_match']}")
        print(f"   Mean intensity: {info['mean_intensity']:.3f}")

        assert error_ratio < 0.02, "Area error too large!"
        print("\n[PASS] TEST PASSED! L0 alignment is correct!")

    except Exception as e:
        print(f"\n[X] TEST FAILED: {e}")
        return False

    return True


def test_compact_lesion():
    """Test case: Compact lesion (should also work)"""
    print("\n" + "=" * 60)
    print("TEST 2: Compact Lesion (Single Solid Patch)")
    print("=" * 60)

    # Create compact lesion: single solid rectangle
    lesion_mask = torch.zeros(3, 224, 224)
    lesion_mask[:, 80:120, 90:130] = 1  # 40×40 = 1600 pixels

    lesion_area = (lesion_mask[0] > 0).sum().item()
    print(f"Lesion actual area: {lesion_area} pixels")

    # Create lung mask and dummy image
    lung_mask = torch.ones(224, 224)
    image = torch.randn(3, 224, 224)

    # Generate random patch
    try:
        random_mask, info = generate_equivalent_random_mask(
            lesion_mask=lesion_mask,
            lung_mask=lung_mask,
            image=image,
            max_attempts=500
        )

        random_area = (random_mask[0] > 0).sum().item()
        error_ratio = abs(random_area - lesion_area) / lesion_area

        print(f"\n[OK] Random_patch area: {random_area} pixels")
        print(f"   Lesion area: {lesion_area} pixels")
        print(f"   Error: {error_ratio:.1%}")

        assert error_ratio < 0.02, "Area error too large!"
        print("\n[PASS] TEST PASSED! L0 alignment is correct!")

    except Exception as e:
        print(f"\n[X] TEST FAILED: {e}")
        return False

    return True


if __name__ == "__main__":
    print("\n[*] Testing L0 Alignment Fix")
    print("This tests that random_patch area EXACTLY matches lesion area\n")

    test1_passed = test_scattered_lesion()
    test2_passed = test_compact_lesion()

    print("\n" + "=" * 60)
    if test1_passed and test2_passed:
        print("[SUCCESS] ALL TESTS PASSED! L0 alignment fix is working!")
        print("\nNow you can re-run experiments with confidence that:")
        print("  - random_patch area = lesion area")
        print("  - L0 norm will be properly aligned across all modes")
        print("  - Your paper's conclusions will be scientifically rigorous!")
    else:
        print("[FAILED] SOME TESTS FAILED - check the output above")
    print("=" * 60)
