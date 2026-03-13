"""
Verify that mask is correctly applied in gradient steps.
"""

import torch
import torch.nn as nn
from unified_attack_framework_fixed import CheXzeroForAttack, MaskedCWAttack, MaskedDeepFoolAttack


class DummyModel(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3*224*224, 2)

    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))


def verify_cw_mask():
    """Verify C&W applies mask in each gradient step."""
    print("=" * 80)
    print("Verifying C&W Mask Application")
    print("=" * 80)

    # Create dummy data
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image = torch.rand(1, 3, 224, 224).to(device)

    # Create a small mask in the center
    mask = torch.zeros_like(image)
    mask[:, :, 100:124, 100:124] = 1.0  # 24x24 patch in center

    model = DummyModel().to(device)
    attack_model = CheXzeroForAttack(model)

    # Run C&W for just 5 steps
    cw = MaskedCWAttack(
        model=attack_model,
        mask=mask,
        c=1.0,
        kappa=10.0,
        steps=5,  # Just 5 steps for testing
        lr=0.01,
        attack_mode='lesion'
    )

    labels = torch.ones(1, dtype=torch.long, device=device)
    adv_image = cw(image, labels)

    # Check that only masked region was modified
    perturbation = adv_image - image

    # Non-masked region perturbation
    non_masked_region = perturbation * (1 - mask)
    non_masked_norm = torch.norm(non_masked_region)

    # Masked region perturbation
    masked_region = perturbation * mask
    masked_norm = torch.norm(masked_region)

    print(f"Non-masked region perturbation norm: {non_masked_norm.item():.6f}")
    print(f"Masked region perturbation norm: {masked_norm.item():.6f}")

    if non_masked_norm < 1e-5:
        print("[OK] SUCCESS: Non-masked region unchanged!")
    else:
        print("[FAIL] FAILED: Non-masked region was modified!")

    print()
    return non_masked_norm < 1e-5


def verify_deepfool_mask():
    """Verify DeepFool applies mask in each iteration."""
    print("=" * 80)
    print("Verifying DeepFool Mask Application")
    print("=" * 80)

    # Create dummy data
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image = torch.rand(1, 3, 224, 224).to(device)

    # Create a small mask in the center
    mask = torch.zeros_like(image)
    mask[:, :, 100:124, 100:124] = 1.0  # 24x24 patch in center

    model = DummyModel().to(device)
    attack_model = CheXzeroForAttack(model)

    # Run DeepFool for just 5 steps
    deepfool = MaskedDeepFoolAttack(
        model=attack_model,
        mask=mask,
        steps=5,  # Just 5 steps for testing
        overshoot=0.02,
        attack_mode='lesion'
    )

    labels = torch.ones(1, dtype=torch.long, device=device)
    adv_image = deepfool(image, labels)

    # Check that only masked region was modified
    perturbation = adv_image - image

    # Non-masked region perturbation
    non_masked_region = perturbation * (1 - mask)
    non_masked_norm = torch.norm(non_masked_region)

    # Masked region perturbation
    masked_region = perturbation * mask
    masked_norm = torch.norm(masked_region)

    print(f"Non-masked region perturbation norm: {non_masked_norm.item():.6f}")
    print(f"Masked region perturbation norm: {masked_norm.item():.6f}")

    if non_masked_norm < 1e-5:
        print("[OK] SUCCESS: Non-masked region unchanged!")
    else:
        print("[FAIL] FAILED: Non-masked region was modified!")

    print()
    return non_masked_norm < 1e-5


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("MASK APPLICATION VERIFICATION TEST")
    print("=" * 80 + "\n")

    cw_pass = verify_cw_mask()
    deepfool_pass = verify_deepfool_mask()

    print("=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"C&W: {'PASS' if cw_pass else 'FAIL'}")
    print(f"DeepFool: {'PASS' if deepfool_pass else 'FAIL'}")

    if cw_pass and deepfool_pass:
        print("\n[OK] ALL TESTS PASSED! Mask is correctly applied in gradient steps.")
    else:
        print("\n[FAIL] SOME TESTS FAILED! Check implementation.")
    print("=" * 80 + "\n")
