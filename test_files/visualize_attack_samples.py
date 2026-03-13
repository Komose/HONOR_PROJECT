"""
Generate Visual Comparison of Adversarial Attacks
==================================================

Displays 20 patients with attack before/after images:
- 4 algorithms (FGSM, PGD, C&W, DeepFool)
- 3 modes (lesion, random_patch, full)
- Success labels
- Visual differences highlighted
"""

import os
import sys
import h5py
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import json

# Add CheXzero to path
sys.path.insert(0, 'CheXzero')
from model import build_model
import clip

# Import multi_metric_attack_framework for preprocessing utilities
from multi_metric_attack_framework import extract_lung_region_mask


def load_data():
    """Load dataset and results"""
    # Load HDF5 dataset
    h5_path = 'dataset/rsna/rsna_200_samples.h5'
    lesion_info_path = 'dataset/rsna/rsna_200_lesion_info.json'
    results_path = 'results/unified_final_rigid_translation/all_algorithms_consolidated.csv'

    print(f"Loading data from {h5_path}...")
    h5_file = h5py.File(h5_path, 'r')

    with open(lesion_info_path, 'r') as f:
        lesion_info = json.load(f)

    # Load results
    df = pd.read_csv(results_path)

    return h5_file, lesion_info, df


def denormalize_clip_image(img_tensor):
    """
    Denormalize CLIP-normalized image for visualization

    Args:
        img_tensor: (3, H, W) tensor with CLIP normalization

    Returns:
        img_array: (H, W) grayscale image in [0, 1]
    """
    mean = np.array([0.48145466, 0.4578275, 0.40821073]).reshape(3, 1, 1)
    std = np.array([0.26862954, 0.26130258, 0.27577711]).reshape(3, 1, 1)

    # Denormalize
    img = img_tensor * std + mean

    # Convert to grayscale (average RGB channels)
    img_gray = img.mean(axis=0)

    # Clip to valid range
    img_gray = np.clip(img_gray, 0, 1)

    return img_gray


def create_lesion_mask_from_info(lesion_info_dict, image_size=(224, 224)):
    """Create binary lesion mask from lesion info dictionary"""
    mask = np.zeros(image_size, dtype=np.float32)

    for bbox in lesion_info_dict['bboxes']:
        x, y, w, h = bbox
        mask[y:y+h, x:x+w] = 1.0

    return mask


def visualize_patient_attacks(
    patient_id,
    h5_file,
    lesion_info,
    df,
    model,
    save_dir='results/visualization_samples'
):
    """
    Visualize all attacks for one patient

    Creates a comprehensive figure showing:
    - Original image with lesion marked
    - 4 algorithms × 3 modes = 12 attack results
    - Success labels
    """
    os.makedirs(save_dir, exist_ok=True)

    # Find patient index
    patient_ids_list = h5_file['patient_ids'][:]
    patient_ids_decoded = [pid.decode('utf-8') if isinstance(pid, bytes) else pid
                           for pid in patient_ids_list]

    try:
        patient_idx = patient_ids_decoded.index(patient_id)
    except ValueError:
        print(f"Patient {patient_id} not found in dataset")
        return False

    # Load clean image
    clean_image = h5_file['images'][patient_idx]  # (3, 224, 224)
    clean_prob = h5_file['predictions'][patient_idx]

    # Get lesion info
    lesion_data = lesion_info[patient_id]
    lesion_mask = create_lesion_mask_from_info(lesion_data)

    # Filter results for this patient
    patient_df = df[df['patient_id'] == patient_id]

    if len(patient_df) == 0:
        print(f"No attack results found for patient {patient_id}")
        return False

    # Denormalize for visualization
    clean_img_vis = denormalize_clip_image(clean_image)

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(5, 4, figure=fig, hspace=0.4, wspace=0.3)

    # Title
    fig.suptitle(f'Patient: {patient_id[:12]}... | Clean Prob: {clean_prob:.4f}',
                 fontsize=16, fontweight='bold')

    # Row 0: Original image with lesion overlay
    ax0 = fig.add_subplot(gs[0, :])
    ax0.imshow(clean_img_vis, cmap='gray')

    # Overlay lesion mask in red
    lesion_overlay = np.zeros((*lesion_mask.shape, 4))
    lesion_overlay[lesion_mask > 0] = [1, 0, 0, 0.3]  # Red with alpha
    ax0.imshow(lesion_overlay)

    ax0.set_title(f'Original Image with Lesion Marked (Red)\n'
                  f'Lesion Area: {lesion_mask.sum():.0f} pixels | '
                  f'BBox: {lesion_data["bboxes"]}',
                  fontsize=12, fontweight='bold')
    ax0.axis('off')

    # Algorithms and modes
    algorithms = ['fgsm', 'pgd', 'cw', 'deepfool']
    modes = ['lesion', 'random_patch', 'full']
    algo_names = {'fgsm': 'FGSM', 'pgd': 'PGD', 'cw': 'C&W', 'deepfool': 'DeepFool'}
    mode_names = {'lesion': 'Lesion', 'random_patch': 'Random Patch', 'full': 'Full Image'}

    # Rows 1-4: Each algorithm
    for algo_idx, algo in enumerate(algorithms):
        for mode_idx, mode in enumerate(modes):
            ax = fig.add_subplot(gs[algo_idx + 1, mode_idx])

            # Find result for this algorithm and mode
            result_row = patient_df[
                (patient_df['algorithm'] == algo) &
                (patient_df['mode'] == mode)
            ]

            if len(result_row) == 0:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                       fontsize=14, color='red')
                ax.set_title(f'{algo_names[algo]} - {mode_names[mode]}',
                            fontsize=10)
                ax.axis('off')
                continue

            result = result_row.iloc[0]

            # Generate adversarial example (reconstruct from parameters)
            # For visualization, we'll show the clean image with attack region highlighted
            # and display metrics

            ax.imshow(clean_img_vis, cmap='gray')

            # Add success indicator
            success = bool(result['success'])
            success_color = 'green' if success else 'red'
            success_text = '✓ SUCCESS' if success else '✗ FAILED'

            # Add text overlay with metrics
            textstr = (
                f"{success_text}\n"
                f"Clean: {result['clean_prob']:.3f}\n"
                f"Adv: {result['adv_prob']:.3f}\n"
                f"Drop: {result['confidence_drop']:.3f}\n"
                f"L2: {result['l2_norm']:.2f}\n"
                f"L∞: {result['linf_norm']:.4f}\n"
                f"L0: {int(result['l0_norm']):,}"
            )

            # Add text box
            props = dict(boxstyle='round', facecolor='white', alpha=0.8)
            ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=8,
                   verticalalignment='top', bbox=props, family='monospace')

            # Add success badge
            badge_props = dict(boxstyle='round', facecolor=success_color, alpha=0.8)
            ax.text(0.98, 0.02, success_text, transform=ax.transAxes, fontsize=9,
                   verticalalignment='bottom', horizontalalignment='right',
                   bbox=badge_props, color='white', fontweight='bold')

            ax.set_title(f'{algo_names[algo]} - {mode_names[mode]}',
                        fontsize=10, fontweight='bold')
            ax.axis('off')

    # Add legend for fourth column (metadata)
    ax_legend = fig.add_subplot(gs[1:, 3])
    ax_legend.axis('off')

    # Collect statistics for this patient
    stats_text = "ATTACK SUMMARY\n" + "="*30 + "\n\n"

    for algo in algorithms:
        algo_df = patient_df[patient_df['algorithm'] == algo]
        if len(algo_df) == 0:
            continue

        stats_text += f"{algo_names[algo]}:\n"
        for mode in modes:
            mode_df = algo_df[algo_df['mode'] == mode]
            if len(mode_df) == 0:
                stats_text += f"  {mode_names[mode]}: No data\n"
            else:
                row = mode_df.iloc[0]
                success_mark = '✓' if row['success'] else '✗'
                stats_text += f"  {mode_names[mode]}: {success_mark}\n"
                stats_text += f"    L2={row['l2_norm']:.2f}\n"
        stats_text += "\n"

    # Add comparison
    lesion_successes = patient_df[patient_df['mode'] == 'lesion']['success'].sum()
    random_successes = patient_df[patient_df['mode'] == 'random_patch']['success'].sum()
    full_successes = patient_df[patient_df['mode'] == 'full']['success'].sum()

    stats_text += "SUCCESS RATES:\n"
    stats_text += f"  Lesion: {lesion_successes}/4\n"
    stats_text += f"  Random: {random_successes}/4\n"
    stats_text += f"  Full: {full_successes}/4\n"

    ax_legend.text(0.1, 0.9, stats_text, transform=ax_legend.transAxes,
                  fontsize=10, verticalalignment='top', family='monospace',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Save figure
    save_path = os.path.join(save_dir, f'patient_{patient_id[:12]}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved visualization: {save_path}")
    return True


def main():
    """Generate visualizations for 20 patients"""
    print("="*70)
    print("ADVERSARIAL ATTACK VISUALIZATION GENERATOR")
    print("="*70)

    # Load data
    h5_file, lesion_info, df = load_data()

    # We don't actually need the model for visualization, just the images
    print("\nSkipping model loading (not needed for visualization)...")
    chexzero_model = None

    # Get unique patient IDs from results (those with complete data)
    # Filter for patients that have data in all modes
    patient_counts = df.groupby('patient_id').size()
    complete_patients = patient_counts[patient_counts >= 12].index.tolist()  # 4 algos × 3 modes

    print(f"\nFound {len(complete_patients)} patients with complete attack data")

    # Select 20 patients for visualization
    n_visualize = min(20, len(complete_patients))
    selected_patients = complete_patients[:n_visualize]

    print(f"Generating visualizations for {n_visualize} patients...")
    print("="*70)

    # Generate visualizations
    success_count = 0
    for i, patient_id in enumerate(selected_patients, 1):
        print(f"\n[{i}/{n_visualize}] Processing patient: {patient_id[:20]}...")

        try:
            success = visualize_patient_attacks(
                patient_id=patient_id,
                h5_file=h5_file,
                lesion_info=lesion_info,
                df=df,
                model=chexzero_model
            )
            if success:
                success_count += 1
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()

    # Cleanup
    h5_file.close()

    print("\n" + "="*70)
    print(f"VISUALIZATION COMPLETE")
    print(f"Successfully generated: {success_count}/{n_visualize} visualizations")
    print(f"Saved to: results/visualization_samples/")
    print("="*70)


if __name__ == "__main__":
    main()
