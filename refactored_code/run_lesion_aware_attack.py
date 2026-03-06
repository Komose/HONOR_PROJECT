#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_lesion_aware_attack.py

Unified runner for lesion-aware adversarial attack experiments.
Implements Objective 5 & 6 of the dissertation project.

Pipeline:
    1. Load CheXzero foundation model
    2. Load CheXpert test data
    3. Generate lesion masks using Grad-CAM
    4. Run lesion-aware attacks (MaskedPGD, MaskedCW)
    5. Run baseline (non-masked) attacks for comparison
    6. Evaluate robustness metrics (AUC drop, ASR, L2/Linf norms)
    7. Save results and visualizations

Example usage:
    python run_lesion_aware_attack.py \
        --model_path models/chexzero.pt \
        --h5_path data/test_cxr.h5 \
        --labels_csv data/final_paths.csv \
        --attack pgd \
        --epsilon 8 \
        --mask_threshold 0.5 \
        --out_dir outputs/lesion_aware_pgd_eps8

Author: Generated for HONER_PROJECT
"""
from __future__ import annotations

import sys
from pathlib import Path
# Add CheXzero to path for model imports
sys.path.insert(0, str(Path(__file__).parent.parent / "CheXzero"))
# Add custom library path for packages installed on C drive (due to disk space constraints)
sys.path.insert(0, "C:/temp/python_libs")

import argparse
import json
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Project modules
from model import CLIP, build_model
import clip as openai_clip

from fm_attack_wrapper import FMAttackWrapper, InputDomain
from grad_cam import MultiLabelGradCAM, create_lesion_mask

# Attacks
from attacks import (
    PGDLinf, CWL2, FGSMLinf, DeepFoolL2,
    MaskedPGDLinf, MaskedCWL2, MaskedFGSMLinf, MaskedDeepFoolL2
)


# 14-label CheXpert list
CXR_LABELS: List[str] = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity',
    'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia',
    'Pneumothorax', 'Support Devices'
]


class CXRH5Dataset(Dataset):
    """CheXpert dataset from HDF5 + CSV labels."""

    def __init__(self, h5_path: str, labels_csv: str, labels: List[str] = CXR_LABELS):
        super().__init__()
        self.h5_path = str(h5_path)
        self.labels_csv = str(labels_csv)
        self.labels = list(labels)

        # Load labels
        df = pd.read_csv(self.labels_csv)
        if all(l in df.columns for l in self.labels):
            y = df[self.labels].to_numpy()
        else:
            df2 = df.copy()
            df2.drop(df2.columns[0], axis=1, inplace=True)
            y = df2.to_numpy()

        self.y = y.astype(np.float32)

        self._h5 = None
        self._dset = None

    def _lazy_init(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
            self._dset = self._h5["cxr"]

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int):
        self._lazy_init()
        img = self._dset[idx]  # (H,W) grayscale

        img = np.asarray(img, dtype=np.float32)
        img = np.expand_dims(img, axis=0)
        img = np.repeat(img, 3, axis=0)
        x = torch.from_numpy(img)  # [3,H,W]

        y = torch.from_numpy(self.y[idx])  # [14]
        return x, y


def str2bool(x: str) -> bool:
    return x.lower() in ("1", "true", "t", "yes", "y")


def load_clip_model(model_path: str, pretrained: bool, context_length: int, device: torch.device) -> torch.nn.Module:
    """Load CheXzero CLIP model using build_model to infer architecture from checkpoint."""
    sd = torch.load(model_path, map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    # Use build_model to automatically infer architecture from state_dict
    model = build_model(sd)
    model.to(device)
    model.eval()
    # Keep requires_grad=True for Grad-CAM (we're in eval mode so no updates will happen)
    return model


def batch_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute mean AUROC across labels."""
    from sklearn.metrics import roc_auc_score
    aucs = []
    for j in range(y_true.shape[1]):
        col = y_true[:, j]
        if np.all(col == 0) or np.all(col == 1):
            continue
        aucs.append(roc_auc_score(col, y_prob[:, j]))
    return float(np.mean(aucs)) if len(aucs) else float("nan")


def create_attack(attack_name: str, epsilon: float, **kwargs):
    """Factory function to create attack instances."""
    if attack_name == "pgd":
        return PGDLinf(epsilon=epsilon, **kwargs)
    elif attack_name == "cw":
        return CWL2(**kwargs)
    elif attack_name == "fgsm":
        return FGSMLinf(epsilon=epsilon, **kwargs)
    elif attack_name == "deepfool":
        return DeepFoolL2(**kwargs)
    else:
        raise ValueError(f"Unknown attack: {attack_name}")


def create_masked_attack(attack_name: str, epsilon: float, mask_mode: str = "multiply", **kwargs):
    """Factory function to create masked attack instances."""
    if attack_name == "pgd":
        return MaskedPGDLinf(epsilon=epsilon, mask_mode=mask_mode, **kwargs)
    elif attack_name == "cw":
        return MaskedCWL2(mask_mode=mask_mode, **kwargs)
    elif attack_name == "fgsm":
        return MaskedFGSMLinf(epsilon=epsilon, mask_mode=mask_mode, **kwargs)
    elif attack_name == "deepfool":
        return MaskedDeepFoolL2(mask_mode=mask_mode, **kwargs)
    else:
        raise ValueError(f"Unknown attack: {attack_name}")


def save_visualization(
    x_clean: np.ndarray,
    x_adv: np.ndarray,
    mask: np.ndarray,
    cam: np.ndarray,
    save_path: Path,
    idx: int = 0
):
    """Save visualization of clean image, adversarial image, mask, and CAM."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Clean image (grayscale)
    img_clean = x_clean[idx, 0, :, :]  # First channel
    axes[0].imshow(img_clean, cmap='gray')
    axes[0].set_title('Clean Image')
    axes[0].axis('off')

    # Adversarial image
    img_adv = x_adv[idx, 0, :, :]
    axes[1].imshow(img_adv, cmap='gray')
    axes[1].set_title('Adversarial Image')
    axes[1].axis('off')

    # Grad-CAM heatmap
    axes[2].imshow(img_clean, cmap='gray')
    axes[2].imshow(cam[idx], cmap='jet', alpha=0.5)
    axes[2].set_title('Grad-CAM Heatmap')
    axes[2].axis('off')

    # Lesion mask
    axes[3].imshow(mask[idx], cmap='Reds')
    axes[3].set_title('Lesion Mask')
    axes[3].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    # Data & Model
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--h5_path", type=str, required=True)
    ap.add_argument("--labels_csv", type=str, required=True)
    ap.add_argument("--pretrained", type=str, default="false")
    ap.add_argument("--context_length", type=int, default=77)

    # Attack settings
    ap.add_argument("--attack", type=str, default="pgd", choices=["pgd", "cw", "fgsm", "deepfool"])
    ap.add_argument("--epsilon", type=float, default=8.0, help="Epsilon for Linf attacks (pixel domain)")
    ap.add_argument("--step_size", type=float, default=2.0, help="Step size for PGD")
    ap.add_argument("--steps", type=int, default=10, help="PGD iterations")
    ap.add_argument("--random_start", type=str, default="true")

    # C&W specific
    ap.add_argument("--cw_confidence", type=float, default=0.0)
    ap.add_argument("--cw_max_iter", type=int, default=1000)
    ap.add_argument("--cw_learning_rate", type=float, default=0.01)

    # Lesion mask settings
    ap.add_argument("--mask_threshold", type=float, default=0.5, help="Grad-CAM threshold for mask")
    ap.add_argument("--mask_mode", type=str, default="multiply", choices=["multiply", "soft"])
    ap.add_argument("--run_baseline", type=str, default="true", help="Also run non-masked baseline")

    # System
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--mean", type=float, nargs=3, default=(101.48761, 101.48761, 101.48761))
    ap.add_argument("--std", type=float, nargs=3, default=(83.43944, 83.43944, 83.43944))
    ap.add_argument("--input_resolution", type=int, default=320)

    # Output
    ap.add_argument("--out_dir", type=str, default="outputs/lesion_aware")
    ap.add_argument("--save_vis", type=str, default="true", help="Save visualizations")
    ap.add_argument("--num_vis", type=int, default=10, help="Number of visualizations to save")

    args = ap.parse_args()

    pretrained = str2bool(args.pretrained)
    random_start = str2bool(args.random_start)
    run_baseline = str2bool(args.run_baseline)
    save_vis = str2bool(args.save_vis)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if save_vis:
        vis_dir = out_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load model
    print("\n[1/6] Loading CheXzero model...")
    model = load_clip_model(args.model_path, pretrained, args.context_length, device)

    # 2. Build wrapper
    print("[2/6] Building model wrapper...")
    wrapper = FMAttackWrapper(
        model,
        class_names=CXR_LABELS,
        pos_template="{}",
        neg_template="no {}",
        context_length=args.context_length,
        device=device,
        input_resolution=args.input_resolution,
        mean=tuple(args.mean),
        std=tuple(args.std),
        input_domain=InputDomain(value_range=(0.0, 255.0), space="pixel", layout="NCHW"),
        clamp=True,
    )

    # Pre-compute text features for Grad-CAM
    text_features = wrapper.pos_text_features  # [14, D]

    # 3. Load data
    print("[3/6] Loading CheXpert data...")
    dset = CXRH5Dataset(args.h5_path, args.labels_csv, labels=CXR_LABELS)
    loader = DataLoader(dset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # 4. Create attacks
    print(f"[4/6] Creating {args.attack.upper()} attacks...")

    attack_kwargs = {}
    if args.attack == "pgd":
        attack_kwargs = {"step_size": args.step_size, "steps": args.steps, "random_start": random_start}
    elif args.attack == "cw":
        attack_kwargs = {
            "confidence": args.cw_confidence,
            "max_iterations": args.cw_max_iter,
            "learning_rate": args.cw_learning_rate,
            "clip_min": 0.0,
            "clip_max": 255.0,
        }

    masked_attack = create_masked_attack(args.attack, args.epsilon, args.mask_mode, **attack_kwargs)

    if run_baseline:
        baseline_attack = create_attack(args.attack, args.epsilon, **attack_kwargs)

    # 5. Run experiments
    print("[5/6] Running lesion-aware attack experiments...")

    # Storage
    results = {
        "masked": {"y": [], "prob_clean": [], "prob_adv": [], "success": [], "norms": [], "masks": [], "cams": []},
    }
    if run_baseline:
        results["baseline"] = {"y": [], "prob_clean": [], "prob_adv": [], "success": [], "norms": []}

    vis_samples = []

    for batch_idx, (x, y) in enumerate(tqdm(loader, desc="Processing batches")):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # Clean predictions
        with torch.no_grad():
            logits_clean = wrapper.forward_logits(x)
            prob_clean = torch.sigmoid(logits_clean)

        # Generate lesion masks using Grad-CAM
        grad_cam = MultiLabelGradCAM(model, use_cuda=(device.type == "cuda"))
        cam = grad_cam.generate_multilabel_cam(
            wrapper.preprocess(x),  # Preprocessed input for CLIP
            text_features,
            target_labels=y
        )
        mask = create_lesion_mask(cam, threshold=args.mask_threshold)
        mask_tensor = torch.from_numpy(mask).unsqueeze(1).to(device)  # [B, 1, H, W]

        # === Lesion-aware (masked) attack ===
        out_masked = masked_attack.perturb(x, y, wrapper, mask=mask_tensor)
        x_adv_masked = out_masked["x_adv"]

        with torch.no_grad():
            logits_adv_masked = wrapper.forward_logits(x_adv_masked)
            prob_adv_masked = torch.sigmoid(logits_adv_masked)

        # Store masked results
        results["masked"]["y"].append(y.cpu().numpy())
        results["masked"]["prob_clean"].append(prob_clean.cpu().numpy())
        results["masked"]["prob_adv"].append(prob_adv_masked.cpu().numpy())
        results["masked"]["success"].append(out_masked["meta"]["success"].numpy())
        results["masked"]["norms"].append({k: v.numpy() for k, v in out_masked["meta"]["norms"].items()})
        results["masked"]["masks"].append(mask)
        results["masked"]["cams"].append(cam)

        # === Baseline (non-masked) attack ===
        if run_baseline:
            out_baseline = baseline_attack.perturb(x, y, wrapper)
            x_adv_baseline = out_baseline["x_adv"]

            with torch.no_grad():
                logits_adv_baseline = wrapper.forward_logits(x_adv_baseline)
                prob_adv_baseline = torch.sigmoid(logits_adv_baseline)

            results["baseline"]["y"].append(y.cpu().numpy())
            results["baseline"]["prob_clean"].append(prob_clean.cpu().numpy())
            results["baseline"]["prob_adv"].append(prob_adv_baseline.cpu().numpy())
            results["baseline"]["success"].append(out_baseline["meta"]["success"].numpy())
            results["baseline"]["norms"].append({k: v.numpy() for k, v in out_baseline["meta"]["norms"].items()})

        # Save visualizations
        if save_vis and len(vis_samples) < args.num_vis:
            vis_samples.append({
                "x_clean": x.cpu().numpy(),
                "x_adv_masked": x_adv_masked.cpu().numpy(),
                "mask": mask,
                "cam": cam,
            })

    # 6. Evaluate and save results
    print("[6/6] Evaluating and saving results...")

    def aggregate_results(res_dict):
        y_all = np.concatenate(res_dict["y"], axis=0)
        prob_clean_all = np.concatenate(res_dict["prob_clean"], axis=0)
        prob_adv_all = np.concatenate(res_dict["prob_adv"], axis=0)
        success_all = np.concatenate(res_dict["success"], axis=0)

        auc_clean = batch_auc(y_all, prob_clean_all)
        auc_adv = batch_auc(y_all, prob_adv_all)
        auc_drop = auc_clean - auc_adv
        asr = float(np.mean(success_all))

        # Aggregate norms
        norms_list = res_dict["norms"]
        linf_all = np.concatenate([n["linf"] for n in norms_list])
        l2_all = np.concatenate([n["l2"] for n in norms_list])

        return {
            "auc_clean": float(auc_clean),
            "auc_adv": float(auc_adv),
            "auc_drop": float(auc_drop),
            "asr": float(asr),
            "mean_linf": float(np.mean(linf_all)),
            "mean_l2": float(np.mean(l2_all)),
            "n_samples": int(len(y_all)),
        }

    summary = {
        "attack": args.attack,
        "epsilon": args.epsilon,
        "mask_threshold": args.mask_threshold,
        "mask_mode": args.mask_mode,
        "masked_attack": aggregate_results(results["masked"]),
    }

    if run_baseline:
        summary["baseline_attack"] = aggregate_results(results["baseline"])

    # Additional metrics: mask coverage
    all_masks = np.concatenate(results["masked"]["masks"], axis=0)
    summary["mean_mask_coverage"] = float(np.mean(all_masks))

    # Save summary
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save raw results
    np.save(out_dir / "masked_probs_clean.npy", np.concatenate(results["masked"]["prob_clean"], axis=0))
    np.save(out_dir / "masked_probs_adv.npy", np.concatenate(results["masked"]["prob_adv"], axis=0))
    np.save(out_dir / "y_true.npy", np.concatenate(results["masked"]["y"], axis=0))

    if run_baseline:
        np.save(out_dir / "baseline_probs_adv.npy", np.concatenate(results["baseline"]["prob_adv"], axis=0))

    # Save visualizations
    if save_vis:
        print(f"Saving {len(vis_samples)} visualizations...")
        for i, sample in enumerate(vis_samples):
            save_visualization(
                sample["x_clean"],
                sample["x_adv_masked"],
                sample["mask"],
                sample["cam"],
                vis_dir / f"sample_{i:03d}.png",
                idx=0
            )

    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(f"Attack: {args.attack.upper()}")
    print(f"Epsilon: {args.epsilon}")
    print(f"Mask Threshold: {args.mask_threshold}")
    print(f"Samples: {summary['masked_attack']['n_samples']}")
    print("\n--- Lesion-Aware (Masked) Attack ---")
    for k, v in summary["masked_attack"].items():
        if k != "n_samples":
            print(f"  {k}: {v:.4f}")
    print(f"  Mean Mask Coverage: {summary['mean_mask_coverage']:.4f}")

    if run_baseline:
        print("\n--- Baseline (Non-Masked) Attack ---")
        for k, v in summary["baseline_attack"].items():
            if k != "n_samples":
                print(f"  {k}: {v:.4f}")

        print("\n--- Comparison ---")
        auc_diff = summary["baseline_attack"]["auc_drop"] - summary["masked_attack"]["auc_drop"]
        print(f"  AUC Drop Difference (Baseline - Masked): {auc_diff:.4f}")
        if auc_diff > 0:
            print("  → Masked attack is MORE effective (larger AUC drop)")
        else:
            print("  → Baseline attack is more effective")

    print(f"\nResults saved to: {out_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
