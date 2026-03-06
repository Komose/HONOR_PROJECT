#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluation.py

Comprehensive evaluation metrics for adversarial robustness assessment
in medical imaging foundation models.

Metrics implemented:
    - AUROC (Area Under ROC Curve) - classification performance
    - Attack Success Rate (ASR) - percentage of successful attacks
    - Perturbation norms (L0, L2, Linf) - perturbation magnitude
    - Lesion-specific metrics - perturbation analysis by region

Author: Generated for HONER_PROJECT
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix


def compute_auroc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    per_label: bool = False
) -> Dict[str, float]:
    """
    Compute AUROC for multi-label classification.

    Args:
        y_true: Ground truth labels [N, L]
        y_prob: Predicted probabilities [N, L]
        per_label: If True, return per-label scores

    Returns:
        Dictionary with mean AUROC and optionally per-label AUROCs
    """
    results = {}
    label_aucs = []

    for j in range(y_true.shape[1]):
        col_true = y_true[:, j]
        col_prob = y_prob[:, j]

        # Skip labels with all-0 or all-1 ground truth
        if np.all(col_true == 0) or np.all(col_true == 1):
            label_aucs.append(np.nan)
            continue

        try:
            auc_score = roc_auc_score(col_true, col_prob)
            label_aucs.append(auc_score)
        except Exception:
            label_aucs.append(np.nan)

    results["mean_auroc"] = float(np.nanmean(label_aucs))
    results["valid_labels"] = int(np.sum(~np.isnan(label_aucs)))

    if per_label:
        results["per_label_auroc"] = label_aucs

    return results


def compute_attack_success_rate(
    logits_clean: np.ndarray,
    logits_adv: np.ndarray,
    threshold: float = 0.0,
) -> Dict[str, float]:
    """
    Compute Attack Success Rate (ASR).

    Success = prediction label-set changes between clean and adversarial.

    Args:
        logits_clean: Clean logits [N, L]
        logits_adv: Adversarial logits [N, L]
        threshold: Decision threshold for binary classification

    Returns:
        Dictionary with ASR and related metrics
    """
    pred_clean = (logits_clean > threshold).astype(int)
    pred_adv = (logits_adv > threshold).astype(int)

    # Check if any label changed
    changed = (pred_clean != pred_adv).any(axis=1)
    asr = float(np.mean(changed))

    # Average number of labels flipped
    num_flipped = (pred_clean != pred_adv).sum(axis=1)
    mean_flipped = float(np.mean(num_flipped))

    return {
        "asr": asr,
        "mean_labels_flipped": mean_flipped,
        "total_successful": int(np.sum(changed)),
        "total_samples": int(len(changed)),
    }


def compute_perturbation_norms(
    x_clean: np.ndarray,
    x_adv: np.ndarray,
) -> Dict[str, float]:
    """
    Compute perturbation norms.

    Args:
        x_clean: Clean images [N, C, H, W]
        x_adv: Adversarial images [N, C, H, W]

    Returns:
        Dictionary with L0, L2, Linf norms (mean and std)
    """
    delta = x_adv - x_clean
    batch_size = delta.shape[0]

    # Flatten per sample
    delta_flat = delta.reshape(batch_size, -1)

    # L0: number of changed pixels
    l0 = np.sum(np.abs(delta_flat) > 1e-6, axis=1)

    # L2: Euclidean distance
    l2 = np.linalg.norm(delta_flat, ord=2, axis=1)

    # Linf: maximum absolute change
    linf = np.max(np.abs(delta_flat), axis=1)

    return {
        "mean_l0": float(np.mean(l0)),
        "std_l0": float(np.std(l0)),
        "mean_l2": float(np.mean(l2)),
        "std_l2": float(np.std(l2)),
        "mean_linf": float(np.mean(linf)),
        "std_linf": float(np.std(linf)),
    }


def compute_lesion_aware_metrics(
    x_clean: np.ndarray,
    x_adv: np.ndarray,
    masks: np.ndarray,
) -> Dict[str, float]:
    """
    Compute lesion-specific perturbation metrics.

    Args:
        x_clean: Clean images [N, C, H, W]
        x_adv: Adversarial images [N, C, H, W]
        masks: Binary lesion masks [N, H, W] or [N, 1, H, W]

    Returns:
        Dictionary with lesion-aware metrics
    """
    delta = x_adv - x_clean
    batch_size = delta.shape[0]

    # Ensure mask is [N, 1, H, W]
    if masks.ndim == 3:
        masks = masks[:, np.newaxis, :, :]
    elif masks.shape[1] != 1:
        masks = masks[:, 0:1, :, :]  # Take first channel

    # Broadcast mask to match delta channels
    masks = np.repeat(masks, delta.shape[1], axis=1)

    # Split perturbations: inside vs outside lesion
    delta_inside = delta * masks
    delta_outside = delta * (1 - masks)

    # Compute norms per region
    delta_inside_flat = delta_inside.reshape(batch_size, -1)
    delta_outside_flat = delta_outside.reshape(batch_size, -1)

    linf_inside = np.max(np.abs(delta_inside_flat), axis=1)
    linf_outside = np.max(np.abs(delta_outside_flat), axis=1)

    l2_inside = np.linalg.norm(delta_inside_flat, ord=2, axis=1)
    l2_outside = np.linalg.norm(delta_outside_flat, ord=2, axis=1)

    # Mask coverage
    mask_coverage = masks[:, 0, :, :].reshape(batch_size, -1).mean(axis=1)

    # Perturbation concentration: ratio of perturbation inside vs total
    total_pert = np.abs(delta).reshape(batch_size, -1).sum(axis=1)
    inside_pert = np.abs(delta_inside).reshape(batch_size, -1).sum(axis=1)
    concentration = inside_pert / (total_pert + 1e-8)

    return {
        "mean_linf_inside_lesion": float(np.mean(linf_inside)),
        "mean_linf_outside_lesion": float(np.mean(linf_outside)),
        "mean_l2_inside_lesion": float(np.mean(l2_inside)),
        "mean_l2_outside_lesion": float(np.mean(l2_outside)),
        "mean_mask_coverage": float(np.mean(mask_coverage)),
        "mean_perturbation_concentration": float(np.mean(concentration)),
    }


def compute_robustness_score(
    auc_clean: float,
    auc_adv: float,
    asr: float,
    mean_linf: float,
    max_epsilon: float = 8.0,
) -> Dict[str, float]:
    """
    Compute overall robustness score.

    Robustness Score = w1 * (1 - AUC_drop) + w2 * (1 - ASR) + w3 * (1 - norm_ratio)

    Args:
        auc_clean: Clean AUC
        auc_adv: Adversarial AUC
        asr: Attack success rate
        mean_linf: Mean Linf norm
        max_epsilon: Maximum expected epsilon (for normalization)

    Returns:
        Dictionary with robustness score and components
    """
    auc_drop = auc_clean - auc_adv
    auc_drop_norm = np.clip(auc_drop / auc_clean, 0, 1) if auc_clean > 0 else 1.0

    asr_norm = np.clip(asr, 0, 1)

    norm_ratio = np.clip(mean_linf / max_epsilon, 0, 1)

    # Weights (can be tuned)
    w1, w2, w3 = 0.4, 0.4, 0.2

    robustness = w1 * (1 - auc_drop_norm) + w2 * (1 - asr_norm) + w3 * (1 - norm_ratio)

    return {
        "robustness_score": float(robustness),
        "auc_drop_normalized": float(auc_drop_norm),
        "asr_normalized": float(asr_norm),
        "norm_ratio": float(norm_ratio),
    }


def create_comparison_table(
    results_dict: Dict[str, Dict[str, float]],
    metrics: List[str] = ["mean_auroc", "asr", "mean_linf", "mean_l2"],
) -> str:
    """
    Create a formatted comparison table for multiple experiments.

    Args:
        results_dict: Dictionary mapping experiment names to metric dicts
        metrics: List of metrics to include

    Returns:
        Formatted table string
    """
    # Get experiment names
    exp_names = list(results_dict.keys())

    # Build header
    header = f"{'Metric':<30} | " + " | ".join([f"{name:<15}" for name in exp_names])
    separator = "-" * len(header)

    lines = [header, separator]

    # Add metric rows
    for metric in metrics:
        values = []
        for exp_name in exp_names:
            if metric in results_dict[exp_name]:
                val = results_dict[exp_name][metric]
                values.append(f"{val:>15.4f}")
            else:
                values.append(f"{'N/A':>15}")

        row = f"{metric:<30} | " + " | ".join(values)
        lines.append(row)

    return "\n".join(lines)


def save_evaluation_report(
    results: Dict[str, any],
    save_path: str,
):
    """
    Save a comprehensive evaluation report as markdown.

    Args:
        results: Dictionary with all evaluation metrics
        save_path: Path to save markdown report
    """
    lines = [
        "# Adversarial Robustness Evaluation Report",
        "",
        "## Experiment Configuration",
        f"- Attack: {results.get('attack', 'N/A')}",
        f"- Epsilon: {results.get('epsilon', 'N/A')}",
        f"- Samples: {results.get('n_samples', 'N/A')}",
        "",
        "## Performance Metrics",
        "",
    ]

    # Add masked attack results
    if "masked_attack" in results:
        lines.extend([
            "### Lesion-Aware (Masked) Attack",
            "",
            f"- Clean AUC: {results['masked_attack'].get('auc_clean', 0):.4f}",
            f"- Adversarial AUC: {results['masked_attack'].get('auc_adv', 0):.4f}",
            f"- AUC Drop: {results['masked_attack'].get('auc_drop', 0):.4f}",
            f"- Attack Success Rate: {results['masked_attack'].get('asr', 0):.4f}",
            f"- Mean L∞ Norm: {results['masked_attack'].get('mean_linf', 0):.4f}",
            f"- Mean L2 Norm: {results['masked_attack'].get('mean_l2', 0):.4f}",
            "",
        ])

    # Add baseline results
    if "baseline_attack" in results:
        lines.extend([
            "### Baseline (Non-Masked) Attack",
            "",
            f"- Clean AUC: {results['baseline_attack'].get('auc_clean', 0):.4f}",
            f"- Adversarial AUC: {results['baseline_attack'].get('auc_adv', 0):.4f}",
            f"- AUC Drop: {results['baseline_attack'].get('auc_drop', 0):.4f}",
            f"- Attack Success Rate: {results['baseline_attack'].get('asr', 0):.4f}",
            f"- Mean L∞ Norm: {results['baseline_attack'].get('mean_linf', 0):.4f}",
            f"- Mean L2 Norm: {results['baseline_attack'].get('mean_l2', 0):.4f}",
            "",
        ])

    # Add comparison
    if "masked_attack" in results and "baseline_attack" in results:
        auc_diff = results["baseline_attack"]["auc_drop"] - results["masked_attack"]["auc_drop"]
        lines.extend([
            "## Comparison",
            "",
            f"- AUC Drop Difference (Baseline - Masked): {auc_diff:.4f}",
            "",
            "**Interpretation:**",
        ])

        if auc_diff > 0.01:
            lines.append("- ✅ Masked attack is MORE effective (larger AUC drop on adversarial examples)")
            lines.append("- Lesion-focused perturbations achieve better attack success")
        elif auc_diff < -0.01:
            lines.append("- ⚠️ Baseline attack is more effective")
            lines.append("- Global perturbations outperform lesion-specific perturbations")
        else:
            lines.append("- ≈ Both attacks have similar effectiveness")

    lines.append("")

    # Save
    with open(save_path, "w") as f:
        f.write("\n".join(lines))


# Example usage
if __name__ == "__main__":
    # Demo with synthetic data
    np.random.seed(42)

    # Simulate results
    n_samples = 100
    n_labels = 14

    y_true = np.random.randint(0, 2, (n_samples, n_labels)).astype(float)
    prob_clean = np.random.uniform(0.3, 0.7, (n_samples, n_labels))
    prob_adv = np.random.uniform(0.2, 0.6, (n_samples, n_labels))

    # Compute metrics
    auroc_results = compute_auroc(y_true, prob_clean)
    print("AUROC (clean):", auroc_results)

    auroc_adv = compute_auroc(y_true, prob_adv)
    print("AUROC (adversarial):", auroc_adv)

    # Synthetic logits
    logits_clean = np.random.randn(n_samples, n_labels)
    logits_adv = logits_clean + np.random.randn(n_samples, n_labels) * 0.5

    asr_results = compute_attack_success_rate(logits_clean, logits_adv)
    print("ASR:", asr_results)

    print("\nEvaluation module loaded successfully!")
