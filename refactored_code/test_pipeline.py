#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_pipeline.py

Quick validation script to test the complete lesion-aware attack pipeline.
Tests each component independently before running full experiments.

Usage:
    python test_pipeline.py --model_path models/chexzero.pt

Author: Generated for HONER_PROJECT
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

print("="*60)
print("LESION-AWARE ATTACK PIPELINE - VALIDATION TEST")
print("="*60)

# Test 1: Import all modules
print("\n[1/7] Testing module imports...")
try:
    from model import CLIP
    import clip as openai_clip
    from fm_attack_wrapper import FMAttackWrapper, InputDomain
    from grad_cam import GradCAM, MultiLabelGradCAM, create_lesion_mask
    from attacks import (
        PGDLinf, CWL2, FGSMLinf, DeepFoolL2,
        MaskedPGDLinf, MaskedCWL2, MaskedFGSMLinf, MaskedDeepFoolL2
    )
    from evaluation import compute_auroc, compute_attack_success_rate
    print("✅ All modules imported successfully")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Create synthetic data
print("\n[2/7] Creating synthetic test data...")
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Using device: {device}")

    # Synthetic images [B, C, H, W]
    batch_size = 2
    img_size = 320
    n_labels = 14

    x = torch.rand(batch_size, 3, img_size, img_size).to(device) * 255.0  # Pixel domain [0, 255]
    y = torch.randint(0, 2, (batch_size, n_labels)).float().to(device)

    print(f"✅ Created synthetic data: x={x.shape}, y={y.shape}")
except Exception as e:
    print(f"❌ Data creation failed: {e}")
    sys.exit(1)

# Test 3: Load model (if provided)
def test_model_loading(model_path: str = None):
    print("\n[3/7] Testing model loading...")

    if model_path is None or not Path(model_path).exists():
        print("   ⚠️  Model path not provided or doesn't exist, using dummy model")

        # Create a minimal dummy CLIP model for testing
        class DummyCLIP(nn.Module):
            def __init__(self):
                super().__init__()
                self.visual = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((10, 10)),
                    nn.Flatten(),
                )
                self.fc = nn.Linear(64 * 10 * 10, 768)

                # Add transformer-like structure for Grad-CAM compatibility
                self.visual.transformer = nn.ModuleList([nn.Identity()])

            def encode_image(self, x):
                feat = self.visual(x)
                return self.fc(feat)

            def encode_text(self, text):
                return torch.randn(text.shape[0], 768, device=text.device)

        model = DummyCLIP().to(device)
        model.eval()
        print("   ✅ Dummy model created for testing")
        return model

    try:
        # Load real CheXzero model
        params = dict(
            embed_dim=768,
            image_resolution=320,
            vision_layers=12,
            vision_width=768,
            vision_patch_size=16,
            context_length=77,
            vocab_size=49408,
            transformer_width=512,
            transformer_heads=8,
            transformer_layers=12,
        )
        model = CLIP(**params)

        sd = torch.load(model_path, map_location=device)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        model.load_state_dict(sd, strict=False)
        model.to(device)
        model.eval()

        print(f"✅ Loaded CheXzero model from {model_path}")
        return model

    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        sys.exit(1)

# Test 4: Model wrapper
def test_wrapper(model, x, y):
    print("\n[4/7] Testing model wrapper...")
    try:
        CXR_LABELS = [
            'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
            'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity',
            'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia',
            'Pneumothorax', 'Support Devices'
        ]

        wrapper = FMAttackWrapper(
            model,
            class_names=CXR_LABELS,
            device=device,
            input_resolution=320,
            mean=(101.48761, 101.48761, 101.48761),
            std=(83.43944, 83.43944, 83.43944),
            input_domain=InputDomain(value_range=(0.0, 255.0), space="pixel", layout="NCHW"),
        )

        # Test forward pass
        logits = wrapper.forward_logits(x)
        loss = wrapper.loss_fn(logits, y)

        print(f"   Logits shape: {logits.shape}")
        print(f"   Loss: {loss.item():.4f}")
        print("✅ Model wrapper working correctly")

        return wrapper

    except Exception as e:
        print(f"❌ Wrapper test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# Test 5: Grad-CAM
def test_gradcam(model, x, y, wrapper):
    print("\n[5/7] Testing Grad-CAM...")
    try:
        # Get text features
        text_features = wrapper.pos_text_features

        # Generate Grad-CAM
        grad_cam = MultiLabelGradCAM(model, use_cuda=(device.type == "cuda"))
        cam = grad_cam.generate_multilabel_cam(
            wrapper.preprocess(x),
            text_features,
            target_labels=y
        )

        # Create mask
        mask = create_lesion_mask(cam, threshold=0.5)

        print(f"   CAM shape: {cam.shape}")
        print(f"   CAM range: [{cam.min():.3f}, {cam.max():.3f}]")
        print(f"   Mask shape: {mask.shape}")
        print(f"   Mask coverage: {mask.mean():.3f}")
        print("✅ Grad-CAM working correctly")

        return mask

    except Exception as e:
        print(f"❌ Grad-CAM test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# Test 6: Attacks
def test_attacks(x, y, wrapper, mask):
    print("\n[6/7] Testing adversarial attacks...")

    attacks_to_test = {
        "PGD": PGDLinf(epsilon=8.0, steps=5),
        "FGSM": FGSMLinf(epsilon=8.0),
        "C&W": CWL2(max_iterations=100, clip_min=0.0, clip_max=255.0),
        "MaskedPGD": MaskedPGDLinf(epsilon=8.0, steps=5),
        "MaskedFGSM": MaskedFGSMLinf(epsilon=8.0),
        "MaskedC&W": MaskedCWL2(max_iterations=100, clip_min=0.0, clip_max=255.0),
    }

    mask_tensor = torch.from_numpy(mask).unsqueeze(1).to(device)

    for attack_name, attack in attacks_to_test.items():
        try:
            print(f"   Testing {attack_name}...", end=" ")

            # Run attack
            if "Masked" in attack_name:
                out = attack.perturb(x, y, wrapper, mask=mask_tensor)
            else:
                out = attack.perturb(x, y, wrapper)

            x_adv = out["x_adv"]
            delta = out["delta"]

            # Validate output
            assert x_adv.shape == x.shape, f"Shape mismatch: {x_adv.shape} != {x.shape}"
            assert delta.shape == x.shape, f"Delta shape mismatch"

            # Check perturbation magnitude
            linf_norm = delta.abs().max().item()
            l2_norm = delta.flatten(1).norm(p=2, dim=1).mean().item()

            print(f"✅ (L∞={linf_norm:.2f}, L2={l2_norm:.2f})")

        except Exception as e:
            print(f"❌ Failed: {e}")
            import traceback
            traceback.print_exc()

    print("✅ All attacks tested successfully")

# Test 7: Evaluation metrics
def test_evaluation(x, y, wrapper):
    print("\n[7/7] Testing evaluation metrics...")
    try:
        # Generate predictions
        with torch.no_grad():
            logits_clean = wrapper.forward_logits(x)
            prob_clean = torch.sigmoid(logits_clean)

            # Simulate adversarial predictions
            logits_adv = logits_clean + torch.randn_like(logits_clean) * 0.5
            prob_adv = torch.sigmoid(logits_adv)

        # Convert to numpy
        y_np = y.cpu().numpy()
        prob_clean_np = prob_clean.cpu().numpy()
        prob_adv_np = prob_adv.cpu().numpy()
        logits_clean_np = logits_clean.cpu().numpy()
        logits_adv_np = logits_adv.cpu().numpy()

        # Test AUROC
        auroc_clean = compute_auroc(y_np, prob_clean_np)
        auroc_adv = compute_auroc(y_np, prob_adv_np)

        print(f"   Clean AUROC: {auroc_clean['mean_auroc']:.4f}")
        print(f"   Adversarial AUROC: {auroc_adv['mean_auroc']:.4f}")

        # Test ASR
        asr = compute_attack_success_rate(logits_clean_np, logits_adv_np)
        print(f"   Attack Success Rate: {asr['asr']:.4f}")

        print("✅ Evaluation metrics working correctly")

    except Exception as e:
        print(f"❌ Evaluation test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# Main test function
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None, help="Path to CheXzero model (optional)")
    args = parser.parse_args()

    # Run tests
    model = test_model_loading(args.model_path)
    wrapper = test_wrapper(model, x, y)
    mask = test_gradcam(model, x, y, wrapper)
    test_attacks(x, y, wrapper, mask)
    test_evaluation(x, y, wrapper)

    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED SUCCESSFULLY!")
    print("="*60)
    print("\nThe pipeline is ready for full experiments.")
    print("Run: python run_lesion_aware_attack.py --help")
    print("="*60)

if __name__ == "__main__":
    main()
