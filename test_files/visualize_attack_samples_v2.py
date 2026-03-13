"""
Simplified Attack Visualization
=================================

Generate before/after comparison images for 20 patients
showing all attack algorithms and modes.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import h5py
import json

def denormalize_clip_image(img_tensor):
    """Denormalize CLIP-normalized image for visualization"""
    mean = np.array([0.48145466, 0.4578275, 0.40821073]).reshape(3, 1, 1)
    std = np.array([0.26862954, 0.26130258, 0.27577711]).reshape(3, 1, 1)

    # Denormalize
    img = img_tensor * std + mean

    # Convert to grayscale
    img_gray = img.mean(axis=0)

    # Clip to valid range
    img_gray = np.clip(img_gray, 0, 1)

    return img_gray


def create_lesion_mask(lesion_info_dict, image_size=(224, 224)):
    """Create binary lesion mask from lesion info"""
    mask = np.zeros(image_size, dtype=np.float32)

    for bbox in lesion_info_dict['bboxes']:
        x, y, w, h = bbox
        mask[y:y+h, x:x+w] = 1.0

    return mask


def visualize_patient(patient_id, patient_idx, clean_image, lesion_info, df, save_dir):
    """
    Create visualization for one patient

    Layout:
    Row 0: Original image with lesion overlay + statistics
    Row 1-4: FGSM, PGD, C&W, DeepFool (columns: lesion, random_patch, full)
    """
    os.makedirs(save_dir, exist_ok=True)

    # Get lesion mask
    lesion_data = lesion_info[patient_id]
    lesion_mask = create_lesion_mask(lesion_data)

    # Filter results for this patient
    patient_df = df[df['patient_id'] == patient_id]

    if len(patient_df) == 0:
        print(f"  No data for patient {patient_id}")
        return False

    # Denormalize image
    clean_img = denormalize_clip_image(clean_image)

    # Create figure
    fig = plt.figure(figsize=(18, 16))
    gs = GridSpec(5, 4, figure=fig, hspace=0.35, wspace=0.25,
                  left=0.05, right=0.95, top=0.94, bottom=0.03)

    # Title
    fig.suptitle(f'Patient: {patient_id[:16]}... (Index: {patient_idx})',
                 fontsize=14, fontweight='bold')

    # Row 0: Original image
    ax0 = fig.add_subplot(gs[0, :3])
    ax0.imshow(clean_img, cmap='gray')

    # Overlay lesion mask
    lesion_overlay = np.zeros((*lesion_mask.shape, 4))
    lesion_overlay[lesion_mask > 0] = [1, 0, 0, 0.4]  # Red with alpha
    ax0.imshow(lesion_overlay)

    # Get clean probability from first result
    clean_prob = patient_df.iloc[0]['clean_prob']

    ax0.set_title(f'Original Image | Clean Probability: {clean_prob:.4f} | Lesion Area: {int(lesion_mask.sum())} pixels',
                  fontsize=11, fontweight='bold')
    ax0.axis('off')

    # Statistics box
    ax_stats = fig.add_subplot(gs[0, 3])
    ax_stats.axis('off')

    # Calculate success rates
    modes = ['lesion', 'random_patch', 'full']
    mode_names = {'lesion': 'Lesion', 'random_patch': 'Random', 'full': 'Full'}

    stats_text = "SUCCESS SUMMARY\\n" + "=" * 25 + "\\n\\n"

    for mode in modes:
        mode_df = patient_df[patient_df['mode'] == mode]
        successes = mode_df['success'].sum()
        total = len(mode_df)
        stats_text += f"{mode_names[mode]:12s}: {int(successes)}/{total}\\n"

    stats_text += "\\n" + "ALGORITHMS:\\n"
    algorithms = ['fgsm', 'pgd', 'cw', 'deepfool']
    algo_names = {'fgsm': 'FGSM', 'pgd': 'PGD', 'cw': 'C&W', 'deepfool': 'DeepFool'}

    for algo in algorithms:
        algo_df = patient_df[patient_df['algorithm'] == algo]
        successes = algo_df['success'].sum()
        total = len(algo_df)
        stats_text += f"{algo_names[algo]:12s}: {int(successes)}/{total}\\n"

    ax_stats.text(0.1, 0.9, stats_text, transform=ax_stats.transAxes,
                  fontsize=9, verticalalignment='top', family='monospace',
                  bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    # Rows 1-4: Each algorithm
    for algo_idx, algo in enumerate(algorithms):
        for mode_idx, mode in enumerate(modes):
            ax = fig.add_subplot(gs[algo_idx + 1, mode_idx])

            # Find result
            result_row = patient_df[
                (patient_df['algorithm'] == algo) &
                (patient_df['mode'] == mode)
            ]

            if len(result_row) == 0:
                ax.text(0.5, 0.5, 'NO DATA', ha='center', va='center',
                       fontsize=12, color='red', fontweight='bold')
                ax.set_title(f'{algo_names[algo]} - {mode_names[mode]}', fontsize=9)
                ax.axis('off')
                continue

            result = result_row.iloc[0]

            # Show image
            ax.imshow(clean_img, cmap='gray')

            # Prepare text
            success = bool(result['success'])
            success_symbol = 'OK' if success else 'X'
            success_color = 'green' if success else 'red'

            info_text = (
                f"Clean: {result['clean_prob']:.3f}\\n"
                f"Adv:   {result['adv_prob']:.3f}\\n"
                f"Drop:  {result['confidence_drop']:.3f}\\n"
                f"L2:    {result['l2_norm']:.2f}\\n"
                f"Linf:  {result['linf_norm']:.4f}\\n"
                f"L0:    {int(result['l0_norm']):,}"
            )

            # Add info box
            props = dict(boxstyle='round', facecolor='white', alpha=0.85)
            ax.text(0.03, 0.97, info_text, transform=ax.transAxes, fontsize=7,
                   verticalalignment='top', bbox=props, family='monospace')

            # Add success badge
            badge_props = dict(boxstyle='round', facecolor=success_color, alpha=0.9)
            ax.text(0.97, 0.03, success_symbol, transform=ax.transAxes, fontsize=11,
                   verticalalignment='bottom', horizontalalignment='right',
                   bbox=badge_props, color='white', fontweight='bold')

            # Title
            title = f'{algo_names[algo]} - {mode_names[mode]}'
            ax.set_title(title, fontsize=9, fontweight='bold')
            ax.axis('off')

    # Legend in last column
    ax_legend = fig.add_subplot(gs[1:, 3])
    ax_legend.axis('off')

    legend_text = (
        "LEGEND\\n"
        "======\\n\\n"
        "SUCCESS: Model fooled\\n"
        "(Prob < 0.5)\\n\\n"
        "Clean: Original prob\\n"
        "Adv:   After attack\\n"
        "Drop:  Confidence drop\\n"
        "L2:    Perturbation L2\\n"
        "Linf:  Max pixel change\\n"
        "L0:    Pixels modified\\n\\n"
        "MODES:\\n"
        "------\\n"
        "Lesion: Attack only\\n"
        "        lesion region\\n\\n"
        "Random: Attack equal-\\n"
        "        area random\\n"
        "        patch\\n\\n"
        "Full:   Attack entire\\n"
        "        image"
    )

    ax_legend.text(0.05, 0.95, legend_text, transform=ax_legend.transAxes,
                  fontsize=7.5, verticalalignment='top', family='monospace',
                  bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

    # Save
    save_path = os.path.join(save_dir, f'patient_{patient_idx:03d}_{patient_id[:12]}.png')
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()

    return True


def main():
    """Generate visualizations for 20 patients"""
    print("=" * 70)
    print("ADVERSARIAL ATTACK VISUALIZATION - SIMPLIFIED")
    print("=" * 70)

    # Load data
    print("\\nLoading data...")
    h5_path = 'dataset/rsna/rsna_200_samples.h5'
    lesion_info_path = 'dataset/rsna/rsna_200_lesion_info.json'
    results_path = 'results/unified_final_rigid_translation/all_algorithms_consolidated.csv'

    h5_file = h5py.File(h5_path, 'r')
    images = h5_file['cxr'][:]  # (200, 3, 224, 224)

    with open(lesion_info_path, 'r') as f:
        lesion_info = json.load(f)

    df = pd.read_csv(results_path)

    print(f"Loaded {len(images)} images")
    print(f"Loaded {len(df)} attack results")

    # Get patient IDs with complete data (12 results each: 4 algos x 3 modes)
    patient_counts = df.groupby('patient_id').size()
    complete_patients = patient_counts[patient_counts >= 12].index.tolist()

    print(f"Found {len(complete_patients)} patients with complete data")

    # Select first 20
    n_visualize = min(20, len(complete_patients))
    selected_patients = complete_patients[:n_visualize]

    print(f"\\nGenerating visualizations for {n_visualize} patients...")
    print("=" * 70)

    save_dir = 'results/visualization_samples'
    os.makedirs(save_dir, exist_ok=True)

    # Map patient IDs to indices
    patient_id_list = lesion_info['patient_ids']
    lesion_data_dict = lesion_info['lesion_data']

    success_count = 0
    for i, patient_id in enumerate(selected_patients, 1):
        print(f"\\n[{i}/{n_visualize}] Patient: {patient_id[:24]}...")

        try:
            # Find patient index
            patient_idx = patient_id_list.index(patient_id)
            clean_image = images[patient_idx]

            success = visualize_patient(
                patient_id=patient_id,
                patient_idx=patient_idx,
                clean_image=clean_image,
                lesion_info=lesion_data_dict,
                df=df,
                save_dir=save_dir
            )

            if success:
                success_count += 1
                print(f"  [OK] Saved successfully")

        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()

    h5_file.close()

    print("\\n" + "=" * 70)
    print(f"COMPLETE: Generated {success_count}/{n_visualize} visualizations")
    print(f"Location: {save_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
