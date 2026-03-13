"""
Attack Visualization Module
============================

Visualize adversarial attacks for all four algorithms (FGSM, PGD, C&W, DeepFool).
Creates comparison images showing:
- Original clean image
- Adversarial image
- Amplified perturbation residual (×30 for visibility)

Usage:
    from attack_visualization import save_attack_visualization

    save_attack_visualization(
        clean_image=clean_img,
        adv_image=adv_img,
        patient_id=patient_id,
        algorithm=algo_name,
        mode=attack_mode,
        save_dir='results/visualizations'
    )

Author: HONER Project
Date: March 2026
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch
from typing import Optional, Union


def denormalize_clip_image(img_tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """
    Denormalize CLIP-normalized image for visualization.

    Args:
        img_tensor: (3, H, W) tensor or array, CLIP normalized

    Returns:
        img_gray: (H, W) grayscale image in [0, 1]
    """
    if isinstance(img_tensor, torch.Tensor):
        img_tensor = img_tensor.cpu().numpy()

    # CLIP normalization parameters
    mean = np.array([0.48145466, 0.4578275, 0.40821073]).reshape(3, 1, 1)
    std = np.array([0.26862954, 0.26130258, 0.27577711]).reshape(3, 1, 1)

    # Denormalize
    img = img_tensor * std + mean
    img = np.clip(img, 0, 1)

    # Convert to grayscale (average across channels)
    img_gray = img.mean(axis=0)

    return img_gray


def save_attack_visualization(
    clean_image: Union[torch.Tensor, np.ndarray],
    adv_image: Union[torch.Tensor, np.ndarray],
    patient_id: str,
    algorithm: str,
    mode: str,
    save_dir: str,
    clean_prob: Optional[float] = None,
    adv_prob: Optional[float] = None,
    epsilon: Optional[float] = None,
    mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    amplify_factor: int = 30
) -> str:
    """
    Save attack visualization showing clean, adversarial, and perturbation images.

    Args:
        clean_image: (3, H, W) clean image
        adv_image: (3, H, W) adversarial image
        patient_id: patient identifier
        algorithm: attack algorithm name (fgsm, pgd, cw, deepfool)
        mode: attack mode (lesion, random_patch, full)
        save_dir: directory to save visualization
        clean_prob: clean prediction probability (optional)
        adv_prob: adversarial prediction probability (optional)
        epsilon: epsilon value for L∞ attacks (optional)
        mask: (3, H, W) or (H, W) attack mask (optional)
        amplify_factor: amplification factor for perturbation visualization

    Returns:
        save_path: path to saved visualization
    """
    os.makedirs(save_dir, exist_ok=True)

    # Denormalize images
    clean_vis = denormalize_clip_image(clean_image)
    adv_vis = denormalize_clip_image(adv_image)

    # Compute perturbation
    diff = adv_vis - clean_vis

    # Amplified perturbation for visualization
    diff_amplified = np.clip(diff * amplify_factor + 0.5, 0, 1)

    # Compute norms
    l2_norm = np.linalg.norm(diff)
    linf_norm = np.max(np.abs(diff))

    # Create figure
    fig = plt.figure(figsize=(18, 6))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Title
    title_parts = [f'Patient: {patient_id[:24]}...',
                   f'{algorithm.upper()} - {mode.upper()}']
    if epsilon is not None:
        title_parts.append(f'ε={epsilon*255:.0f}/255')
    fig.suptitle(' | '.join(title_parts), fontsize=14, fontweight='bold')

    # Row 1: Images
    # Clean image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(clean_vis, cmap='gray', vmin=0, vmax=1)
    title1 = 'Clean Image'
    if clean_prob is not None:
        title1 += f'\nProb: {clean_prob:.4f}'
    ax1.set_title(title1, fontsize=11, fontweight='bold')
    ax1.axis('off')

    # Adversarial image
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(adv_vis, cmap='gray', vmin=0, vmax=1)
    title2 = 'Adversarial Image'
    if adv_prob is not None:
        success = (clean_prob is not None and clean_prob >= 0.5 and adv_prob < 0.5)
        color = 'green' if success else 'red'
        symbol = '✓' if success else '✗'
        title2 += f'\nProb: {adv_prob:.4f} {symbol}'
        ax2.set_title(title2, fontsize=11, fontweight='bold', color=color)
    else:
        ax2.set_title(title2, fontsize=11, fontweight='bold')
    ax2.axis('off')

    # Perturbation (amplified)
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(diff_amplified, cmap='hot', vmin=0, vmax=1)
    ax3.set_title(f'Perturbation (×{amplify_factor})', fontsize=11, fontweight='bold')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046)

    # Row 2: Detailed analysis
    # Absolute difference (not amplified)
    ax4 = fig.add_subplot(gs[1, 0])
    im4 = ax4.imshow(np.abs(diff), cmap='hot', vmin=0, vmax=0.1)
    ax4.set_title('Absolute Difference\n(True Scale)', fontsize=10)
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046)

    # Mask overlay (if provided)
    ax5 = fig.add_subplot(gs[1, 1])
    if mask is not None:
        # Process mask
        if isinstance(mask, torch.Tensor):
            mask_np = mask.cpu().numpy()
        else:
            mask_np = mask

        if mask_np.ndim == 3:
            mask_2d = mask_np[0]
        else:
            mask_2d = mask_np

        # Show clean image with mask overlay
        ax5.imshow(clean_vis, cmap='gray')
        overlay = np.zeros((*mask_2d.shape, 4))
        overlay[mask_2d > 0] = [1, 0, 0, 0.4]  # Red with 40% opacity
        ax5.imshow(overlay)
        ax5.set_title(f'Attack Mask\n{int(mask_2d.sum())} pixels', fontsize=10)
    else:
        ax5.imshow(clean_vis, cmap='gray')
        ax5.set_title('No Mask', fontsize=10)
    ax5.axis('off')

    # Statistics
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')

    stats_text = "PERTURBATION METRICS\n" + "="*25 + "\n\n"
    stats_text += f"L2 Norm:     {l2_norm:.4f}\n"
    stats_text += f"L∞ Norm:     {linf_norm:.6f}\n"
    if epsilon is not None:
        stats_text += f"ε constraint: {epsilon:.6f}\n"
    stats_text += f"\nMin change:  {diff.min():.6f}\n"
    stats_text += f"Max change:  {diff.max():.6f}\n"
    stats_text += f"Mean |Δ|:    {np.abs(diff).mean():.6f}\n"

    if clean_prob is not None and adv_prob is not None:
        stats_text += f"\nPROBABILITY CHANGE\n" + "="*25 + "\n"
        stats_text += f"Clean:       {clean_prob:.4f}\n"
        stats_text += f"Adversarial: {adv_prob:.4f}\n"
        stats_text += f"Drop:        {clean_prob - adv_prob:.4f}\n"

        if clean_prob >= 0.5:
            success = adv_prob < 0.5
            stats_text += f"\n{'SUCCESS ✓' if success else 'FAILED ✗'}"

    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.9)
    ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes,
             fontsize=9, verticalalignment='top', family='monospace',
             bbox=props)

    # Save
    safe_patient_id = patient_id[:12].replace('/', '-')
    if epsilon is not None:
        filename = f'{safe_patient_id}_{algorithm}_eps{int(epsilon*255)}_{mode}.png'
    else:
        filename = f'{safe_patient_id}_{algorithm}_{mode}.png'

    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return save_path


def save_multi_algorithm_comparison(
    clean_image: Union[torch.Tensor, np.ndarray],
    adv_images_dict: dict,
    patient_id: str,
    mode: str,
    save_dir: str,
    clean_prob: Optional[float] = None,
    adv_probs_dict: Optional[dict] = None,
    masks_dict: Optional[dict] = None
) -> str:
    """
    Save comparison visualization across multiple algorithms.

    Args:
        clean_image: (3, H, W) clean image
        adv_images_dict: dict mapping algorithm names to adversarial images
        patient_id: patient identifier
        mode: attack mode
        save_dir: directory to save
        clean_prob: clean prediction probability
        adv_probs_dict: dict mapping algorithm names to adversarial probabilities
        masks_dict: dict mapping algorithm names to masks

    Returns:
        save_path: path to saved visualization
    """
    os.makedirs(save_dir, exist_ok=True)

    algorithms = list(adv_images_dict.keys())
    n_algos = len(algorithms)

    # Create figure
    fig, axes = plt.subplots(n_algos, 4, figsize=(20, 5*n_algos))
    if n_algos == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(f'Patient: {patient_id[:24]}... | Mode: {mode.upper()}',
                 fontsize=14, fontweight='bold')

    clean_vis = denormalize_clip_image(clean_image)

    for i, algo in enumerate(algorithms):
        adv_image = adv_images_dict[algo]
        adv_vis = denormalize_clip_image(adv_image)
        diff = adv_vis - clean_vis
        diff_amp = np.clip(diff * 30 + 0.5, 0, 1)

        adv_prob = adv_probs_dict.get(algo) if adv_probs_dict else None

        # Clean
        axes[i, 0].imshow(clean_vis, cmap='gray')
        axes[i, 0].set_title('Clean' if i == 0 else '', fontsize=10)
        axes[i, 0].set_ylabel(algo.upper(), fontsize=11, fontweight='bold')
        axes[i, 0].axis('off')

        # Adversarial
        axes[i, 1].imshow(adv_vis, cmap='gray')
        title = 'Adversarial' if i == 0 else ''
        if adv_prob is not None:
            title += f'\n{adv_prob:.3f}'
        axes[i, 1].set_title(title, fontsize=10)
        axes[i, 1].axis('off')

        # Perturbation
        axes[i, 2].imshow(diff_amp, cmap='hot')
        axes[i, 2].set_title('Perturbation (×30)' if i == 0 else '', fontsize=10)
        axes[i, 2].axis('off')

        # Absolute diff
        axes[i, 3].imshow(np.abs(diff), cmap='hot', vmin=0, vmax=0.1)
        axes[i, 3].set_title('|Difference|' if i == 0 else '', fontsize=10)
        axes[i, 3].axis('off')

    plt.tight_layout()

    safe_patient_id = patient_id[:12].replace('/', '-')
    save_path = os.path.join(save_dir, f'{safe_patient_id}_comparison_{mode}.png')
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()

    return save_path


if __name__ == '__main__':
    print("Attack Visualization Module")
    print("="*50)
    print("\nThis module provides visualization functions for adversarial attacks.")
    print("\nMain functions:")
    print("  - save_attack_visualization(): Single attack visualization")
    print("  - save_multi_algorithm_comparison(): Multi-algorithm comparison")
    print("\nSee docstrings for usage examples.")
