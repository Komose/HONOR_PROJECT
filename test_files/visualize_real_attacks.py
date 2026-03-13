"""
CORRECT Attack Visualization - Generate Real Adversarial Examples
==================================================================

This script:
1. Loads clean images
2. ACTUALLY RUNS the attacks to generate adversarial examples
3. Shows BEFORE/AFTER comparison with visible perturbations
"""

import os
import sys
import h5py
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

# Add CheXzero to path
sys.path.insert(0, 'CheXzero')
from model import build_model
import clip

# Import attack framework
from unified_attack_framework_fixed import (
    CheXzeroForAttack,
    run_single_attack
)


def denormalize_clip_image(img_tensor):
    """Denormalize CLIP-normalized image for visualization"""
    mean = np.array([0.48145466, 0.4578275, 0.40821073]).reshape(3, 1, 1)
    std = np.array([0.26862954, 0.26130258, 0.27577711]).reshape(3, 1, 1)

    img = img_tensor * std + mean
    img_gray = img.mean(axis=0)
    img_gray = np.clip(img_gray, 0, 1)

    return img_gray


def create_lesion_mask(lesion_info_dict, image_size=(224, 224)):
    """Create binary lesion mask"""
    mask = np.zeros((3, *image_size), dtype=np.float32)

    for bbox in lesion_info_dict['bboxes']:
        x, y, w, h = bbox
        mask[:, y:y+h, x:x+w] = 1.0

    return torch.from_numpy(mask)


def generate_random_patch_mask(lesion_mask, image_tensor):
    """Generate random patch mask using the same method as experiments"""
    from multi_metric_attack_framework import (
        extract_lung_region_mask,
        generate_equivalent_random_mask
    )

    lung_mask = extract_lung_region_mask(image_tensor)
    random_mask, _ = generate_equivalent_random_mask(
        lesion_mask=lesion_mask,
        lung_mask=lung_mask,
        image=image_tensor,
        max_attempts=500
    )

    return random_mask


def run_attack_for_visualization(
    clean_image,
    lesion_mask,
    model,
    algorithm,
    mode,
    device
):
    """
    Run attack and return adversarial image

    Returns:
        adv_image: (3, H, W) adversarial image
        success: bool
        metrics: dict with L2, Linf, L0
    """
    clean_tensor = torch.from_numpy(clean_image).float().unsqueeze(0).to(device)

    # Generate mask based on mode
    if mode == 'lesion':
        mask = lesion_mask.unsqueeze(0).to(device)
    elif mode == 'random_patch':
        random_mask = generate_random_patch_mask(lesion_mask, clean_tensor[0])
        mask = random_mask.unsqueeze(0).to(device)
    else:  # full
        mask = torch.ones_like(clean_tensor).to(device)

    # Set attack parameters
    if algorithm == 'fgsm':
        params = {'epsilon': 8/255, 'use_torchattacks': False}
    elif algorithm == 'pgd':
        params = {'epsilon': 8/255, 'alpha': 2/255, 'num_steps': 40}
    elif algorithm == 'cw':
        params = {'c': 50.0, 'kappa': 0.01, 'steps': 1000, 'lr': 0.05}
    else:  # deepfool
        params = {'steps': 50, 'overshoot': 0.01}

    # Run attack
    try:
        result = run_single_attack(
            model=model,
            image=clean_tensor,
            mask=mask,
            algorithm=algorithm,
            mode=mode,
            **params
        )

        adv_image = result['adv_image'].squeeze(0).cpu().numpy()

        return {
            'adv_image': adv_image,
            'success': result['success'],
            'clean_prob': result['clean_prob'],
            'adv_prob': result['adv_prob'],
            'l2_norm': result['l2_norm'],
            'linf_norm': result['linf_norm'],
            'l0_norm': result['l0_norm']
        }
    except Exception as e:
        print(f"  Warning: Attack failed: {e}")
        return None


def visualize_patient_with_real_attacks(
    patient_id,
    patient_idx,
    clean_image,
    lesion_info_dict,
    model,
    device,
    save_dir
):
    """
    Generate visualization with REAL adversarial examples
    """
    print(f"  Generating adversarial examples...")

    # Create lesion mask
    lesion_mask = create_lesion_mask(lesion_info_dict)
    clean_img_vis = denormalize_clip_image(clean_image)
    lesion_mask_vis = lesion_mask[0].numpy()

    # Algorithms and modes
    algorithms = ['fgsm', 'pgd', 'cw', 'deepfool']
    modes = ['lesion', 'random_patch', 'full']
    algo_names = {'fgsm': 'FGSM', 'pgd': 'PGD', 'cw': 'C&W', 'deepfool': 'DeepFool'}
    mode_names = {'lesion': 'Lesion', 'random_patch': 'Random', 'full': 'Full'}

    # Get clean probability
    clean_tensor = torch.from_numpy(clean_image).float().unsqueeze(0).to(device)
    with torch.no_grad():
        clean_prob = model(clean_tensor).item()

    # Create figure (larger to show details)
    fig = plt.figure(figsize=(22, 20))
    gs = GridSpec(5, 7, figure=fig, hspace=0.4, wspace=0.3)

    # Title
    fig.suptitle(f'Patient: {patient_id[:16]}... (Index: {patient_idx}) | Clean Prob: {clean_prob:.4f}',
                 fontsize=15, fontweight='bold')

    # Row 0: Original image
    ax_orig = fig.add_subplot(gs[0, :2])
    ax_orig.imshow(clean_img_vis, cmap='gray')
    lesion_overlay = np.zeros((*lesion_mask_vis.shape, 4))
    lesion_overlay[lesion_mask_vis > 0] = [1, 0, 0, 0.4]
    ax_orig.imshow(lesion_overlay)
    ax_orig.set_title(f'Original Image\\nLesion Area: {int(lesion_mask_vis.sum())} pixels',
                      fontsize=11, fontweight='bold')
    ax_orig.axis('off')

    # Legend
    ax_legend = fig.add_subplot(gs[0, 2:])
    ax_legend.axis('off')
    legend_text = (
        "VISUALIZATION GUIDE\\n"
        "==================\\n"
        "Each cell shows: [Clean] → [Adversarial] → [Difference×10]\\n\\n"
        "✓ GREEN: Attack succeeded (prob < 0.5)\\n"
        "✗ RED:   Attack failed (prob >= 0.5)\\n\\n"
        "Difference image: Perturbation amplified 10× for visibility\\n"
        "Brighter = larger perturbation"
    )
    ax_legend.text(0.05, 0.95, legend_text, transform=ax_legend.transAxes,
                   fontsize=9, verticalalignment='top', family='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Rows 1-4: Each algorithm
    for algo_idx, algo in enumerate(algorithms):
        for mode_idx, mode in enumerate(modes):
            # Run attack
            print(f"    Running {algo_names[algo]:8s} - {mode_names[mode]:6s}...", end=' ')

            attack_result = run_attack_for_visualization(
                clean_image=clean_image,
                lesion_mask=lesion_mask,
                model=model,
                algorithm=algo,
                mode=mode,
                device=device
            )

            if attack_result is None:
                print("FAILED")
                # Show error placeholder
                for i in range(3):
                    ax = fig.add_subplot(gs[algo_idx + 1, mode_idx * 2 + i])
                    ax.text(0.5, 0.5, 'ERROR', ha='center', va='center',
                           fontsize=14, color='red', fontweight='bold')
                    ax.axis('off')
                continue

            print(f"{'SUCCESS' if attack_result['success'] else 'FAILED'}")

            # Extract results
            adv_image = attack_result['adv_image']
            adv_img_vis = denormalize_clip_image(adv_image)

            # Compute difference (perturbation)
            diff = adv_img_vis - clean_img_vis
            diff_amplified = np.clip(diff * 10 + 0.5, 0, 1)  # Amplify for visibility

            success = attack_result['success']
            success_color = 'green' if success else 'red'
            success_symbol = 'OK' if success else 'X'

            # Show: Clean | Adversarial | Difference
            col_base = mode_idx * 2

            # Column 1: Clean image
            ax1 = fig.add_subplot(gs[algo_idx + 1, col_base])
            ax1.imshow(clean_img_vis, cmap='gray')
            ax1.set_title('Clean', fontsize=8)
            ax1.axis('off')

            # Column 2: Adversarial image with success badge
            ax2 = fig.add_subplot(gs[algo_idx + 1, col_base + 1])
            ax2.imshow(adv_img_vis, cmap='gray')

            # Add success badge
            badge_props = dict(boxstyle='round', facecolor=success_color, alpha=0.9)
            ax2.text(0.95, 0.05, success_symbol, transform=ax2.transAxes,
                    fontsize=10, verticalalignment='bottom', horizontalalignment='right',
                    bbox=badge_props, color='white', fontweight='bold')

            # Add metrics
            info_text = (
                f"{attack_result['adv_prob']:.3f}\\n"
                f"L2:{attack_result['l2_norm']:.1f}\\n"
                f"L∞:{attack_result['linf_norm']:.3f}"
            )
            props = dict(boxstyle='round', facecolor='white', alpha=0.8)
            ax2.text(0.05, 0.95, info_text, transform=ax2.transAxes,
                    fontsize=7, verticalalignment='top', bbox=props, family='monospace')

            ax2.set_title(f'Adversarial\\n{algo_names[algo]}-{mode_names[mode]}',
                         fontsize=8, fontweight='bold')
            ax2.axis('off')

    # Save
    save_path = os.path.join(save_dir, f'real_attack_{patient_idx:03d}_{patient_id[:12]}.png')
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()

    return True


def main():
    """Generate REAL attack visualizations for selected patients"""
    print("=" * 70)
    print("REAL ADVERSARIAL ATTACK VISUALIZATION")
    print("=" * 70)

    # Load data
    print("\\nLoading data and model...")
    h5_path = 'dataset/rsna/rsna_200_samples.h5'
    lesion_info_path = 'dataset/rsna/rsna_200_lesion_info.json'

    h5_file = h5py.File(h5_path, 'r')
    images = h5_file['cxr'][:]

    with open(lesion_info_path, 'r') as f:
        lesion_info = json.load(f)

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model_path = 'CheXzero/checkpoints/best_64_5e-05_original_22000_0.862.pt'
    clip_model, _ = clip.load("ViT-B/32", device=device)
    chexzero_base = build_model(clip_model)
    chexzero_base.load_state_dict(torch.load(model_path, map_location=device))

    # Wrap for attacks
    model = CheXzeroForAttack(chexzero_base).to(device)
    model.eval()

    print("Model loaded successfully!")

    # Select patients
    patient_id_list = lesion_info['patient_ids']
    lesion_data_dict = lesion_info['lesion_data']

    # Select diverse patients (different indices)
    selected_indices = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45,
                       50, 60, 70, 80, 90, 100, 120, 140, 160, 180]

    n_visualize = len(selected_indices)
    save_dir = 'results/real_attack_visualization'
    os.makedirs(save_dir, exist_ok=True)

    print(f"\\nGenerating {n_visualize} visualizations with REAL attacks...")
    print("=" * 70)

    success_count = 0
    for i, idx in enumerate(selected_indices, 1):
        patient_id = patient_id_list[idx]
        print(f"\\n[{i}/{n_visualize}] Patient {idx}: {patient_id[:24]}...")

        try:
            clean_image = images[idx]
            lesion_data = lesion_data_dict[patient_id]

            success = visualize_patient_with_real_attacks(
                patient_id=patient_id,
                patient_idx=idx,
                clean_image=clean_image,
                lesion_info_dict=lesion_data,
                model=model,
                device=device,
                save_dir=save_dir
            )

            if success:
                success_count += 1
                print(f"  ✓ Saved successfully!")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()

    h5_file.close()

    print("\\n" + "=" * 70)
    print(f"COMPLETE: Generated {success_count}/{n_visualize} real attack visualizations")
    print(f"Location: {save_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
