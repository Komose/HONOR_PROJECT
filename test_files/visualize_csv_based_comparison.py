"""
CSV-Based Attack Visualization
===============================
Generate visualizations showing real attack differences using CSV metrics
(No need to re-run attacks - uses existing experiment results)
"""

import os
import h5py
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches


def denormalize_clip_image(img_tensor):
    """Denormalize for visualization"""
    mean = np.array([0.48145466, 0.4578275, 0.40821073]).reshape(3, 1, 1)
    std = np.array([0.26862954, 0.26130258, 0.27577711]).reshape(3, 1, 1)
    img = img_tensor * std + mean
    img_gray = img.mean(axis=0)
    return np.clip(img_gray, 0, 1)


def create_lesion_mask(lesion_info_dict):
    """Create lesion mask"""
    mask = np.zeros((224, 224), dtype=np.float32)
    for bbox in lesion_info_dict['bboxes']:
        x, y, w, h = bbox
        mask[y:y+h, x:x+w] = 1.0
    return mask


def visualize_patient_metrics(
    patient_id,
    patient_idx,
    clean_image,
    lesion_info,
    df_patient,
    save_path
):
    """
    Visualize attack results using CSV metrics
    Shows clean image + attack success/metrics for each algorithm and mode
    """
    clean_img = denormalize_clip_image(clean_image)
    lesion_mask = create_lesion_mask(lesion_info)

    # Create figure
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(5, 4, figure=fig, hspace=0.5, wspace=0.35,
                  left=0.05, right=0.95, top=0.93, bottom=0.05)

    # Get clean probability
    clean_prob = df_patient.iloc[0]['clean_prob']

    # Title
    fig.suptitle(
        f'Patient {patient_idx}: {patient_id[:20]}...\\n'
        f'Clean Probability: {clean_prob:.4f} | Lesion Area: {int(lesion_mask.sum())} pixels',
        fontsize=15, fontweight='bold'
    )

    # Row 0: Clean image with lesion marked
    ax_clean = fig.add_subplot(gs[0, :2])
    ax_clean.imshow(clean_img, cmap='gray')

    # Overlay lesion
    lesion_overlay = np.zeros((*lesion_mask.shape, 4))
    lesion_overlay[lesion_mask > 0] = [1, 0, 0, 0.5]
    ax_clean.imshow(lesion_overlay)

    ax_clean.set_title('Clean Image with Lesion (Red)', fontsize=12, fontweight='bold')
    ax_clean.axis('off')

    # Summary statistics box
    ax_summary = fig.add_subplot(gs[0, 2:])
    ax_summary.axis('off')

    # Calculate success rates
    algorithms = ['fgsm', 'pgd', 'cw', 'deepfool']
    modes = ['lesion', 'random_patch', 'full']

    summary_text = "ATTACK SUCCESS SUMMARY\\n" + "=" * 50 + "\\n\\n"

    for mode in modes:
        mode_name = {'lesion': 'Lesion', 'random_patch': 'Random Patch', 'full': 'Full Image'}[mode]
        mode_df = df_patient[df_patient['mode'] == mode]
        success_count = int(mode_df['success'].sum())
        total = len(mode_df)
        summary_text += f"{mode_name:15s}: {success_count}/{total} algorithms succeeded\\n"

    summary_text += "\\n" + "=" * 50 + "\\n"
    summary_text += "BY ALGORITHM:\\n\\n"

    for algo in algorithms:
        algo_name = {'fgsm': 'FGSM', 'pgd': 'PGD', 'cw': 'C&W', 'deepfool': 'DeepFool'}[algo]
        algo_df = df_patient[df_patient['algorithm'] == algo]
        success_count = int(algo_df['success'].sum())
        total = len(algo_df)
        summary_text += f"{algo_name:10s}: {success_count}/{total} modes succeeded\\n"

    ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes,
                    fontsize=10, verticalalignment='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # Rows 1-4: Results for each algorithm
    algo_names = {'fgsm': 'FGSM', 'pgd': 'PGD', 'cw': 'C&W', 'deepfool': 'DeepFool'}
    mode_names = {'lesion': 'Lesion', 'random_patch': 'Random', 'full': 'Full'}

    for algo_idx, algo in enumerate(algorithms):
        for mode_idx, mode in enumerate(modes):
            ax = fig.add_subplot(gs[algo_idx + 1, mode_idx])

            # Get result
            result = df_patient[
                (df_patient['algorithm'] == algo) &
                (df_patient['mode'] == mode)
            ]

            if len(result) == 0:
                ax.text(0.5, 0.5, 'NO DATA', ha='center', va='center',
                       fontsize=14, color='red', fontweight='bold')
                ax.set_title(f'{algo_names[algo]} - {mode_names[mode]}', fontsize=10)
                ax.axis('off')
                continue

            result = result.iloc[0]

            # Show clean image as base
            ax.imshow(clean_img, cmap='gray', alpha=0.7)

            # Visual indicator of attack strength (L2 norm as heatmap overlay)
            if mode == 'lesion':
                attack_mask = lesion_mask
            elif mode == 'full':
                attack_mask = np.ones_like(lesion_mask)
            else:  # random_patch
                # Approximate: show uniform intensity across image
                attack_mask = np.ones_like(lesion_mask) * 0.3

            # Color-code by success
            success = bool(result['success'])
            color = [0, 1, 0, 0.4] if success else [1, 0, 0, 0.4]  # Green/Red

            attack_overlay = np.zeros((*attack_mask.shape, 4))
            attack_overlay[attack_mask > 0] = color
            ax.imshow(attack_overlay)

            # Add metrics text
            metrics_text = (
                f"Clean:  {result['clean_prob']:.4f}\\n"
                f"Adv:    {result['adv_prob']:.4f}\\n"
                f"Drop:   {result['confidence_drop']:.4f}\\n"
                f"\\n"
                f"L2:     {result['l2_norm']:.2f}\\n"
                f"L∞:     {result['linf_norm']:.4f}\\n"
                f"L0:     {int(result['l0_norm']):,}"
            )

            props = dict(boxstyle='round', facecolor='white', alpha=0.9)
            ax.text(0.03, 0.97, metrics_text, transform=ax.transAxes,
                   fontsize=8, verticalalignment='top', bbox=props,
                   family='monospace')

            # Success badge
            badge_text = '✓ SUCCESS' if success else '✗ FAILED'
            badge_color = 'green' if success else 'red'
            badge_props = dict(boxstyle='round', facecolor=badge_color, alpha=0.9)
            ax.text(0.97, 0.03, badge_text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='bottom', horizontalalignment='right',
                   bbox=badge_props, color='white', fontweight='bold')

            # Title
            title = f'{algo_names[algo]} - {mode_names[mode]}'
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.axis('off')

    # Legend in last column
    ax_legend = fig.add_subplot(gs[1:, 3])
    ax_legend.axis('off')

    legend_text = (
        "METRICS EXPLANATION\\n"
        "===================\\n\\n"
        "SUCCESS: Adv prob < 0.5\\n"
        "(Model fooled)\\n\\n"
        "Clean:  Original prob\\n"
        "Adv:    After attack\\n"
        "Drop:   Confidence drop\\n\\n"
        "L2:     Perturbation\\n"
        "        magnitude\\n"
        "L∞:     Max pixel\\n"
        "        change\\n"
        "L0:     # pixels\\n"
        "        modified\\n\\n"
        "COLOR OVERLAY:\\n"
        "Green = Attack succeeded\\n"
        "Red   = Attack failed\\n\\n"
        "BRIGHTNESS indicates\\n"
        "attack region"
    )

    ax_legend.text(0.05, 0.95, legend_text, transform=ax_legend.transAxes,
                  fontsize=9, verticalalignment='top', family='monospace',
                  bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.savefig(save_path, dpi=110, bbox_inches='tight')
    plt.close()

    return True


def main():
    """Generate visualizations from CSV data"""
    print("=" * 70)
    print("CSV-BASED ATTACK VISUALIZATION")
    print("(Using real experiment results - no need to re-run attacks)")
    print("=" * 70)

    # Load data
    print("\\nLoading data...")
    h5_file = h5py.File('dataset/rsna/rsna_200_samples.h5', 'r')
    images = h5_file['cxr'][:]

    with open('dataset/rsna/rsna_200_lesion_info.json', 'r') as f:
        lesion_info = json.load(f)

    df = pd.read_csv('results/unified_final_rigid_translation/all_algorithms_consolidated.csv')

    patient_ids = lesion_info['patient_ids']
    lesion_data = lesion_info['lesion_data']

    print(f"Loaded {len(images)} images")
    print(f"Loaded {len(df)} attack results")

    # Get patients with complete data
    patient_counts = df.groupby('patient_id').size()
    complete_patients = patient_counts[patient_counts >= 12].index.tolist()

    print(f"Found {len(complete_patients)} patients with complete data")

    # Select 20 diverse patients
    selected_indices = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45,
                       50, 60, 70, 80, 90, 100, 120, 140, 160, 180]

    save_dir = 'results/csv_based_visualization'
    os.makedirs(save_dir, exist_ok=True)

    print(f"\\nGenerating {len(selected_indices)} visualizations...")
    print("=" * 70)

    success_count = 0
    for i, idx in enumerate(selected_indices, 1):
        patient_id = patient_ids[idx]

        # Check if patient has complete data
        if patient_id not in complete_patients:
            print(f"\\n[{i}/{len(selected_indices)}] Patient {idx}: SKIPPED (incomplete data)")
            continue

        print(f"\\n[{i}/{len(selected_indices)}] Patient {idx}: {patient_id[:24]}...")

        try:
            # Get patient data
            df_patient = df[df['patient_id'] == patient_id]

            save_path = os.path.join(
                save_dir,
                f'patient_{idx:03d}_{patient_id[:12]}.png'
            )

            success = visualize_patient_metrics(
                patient_id=patient_id,
                patient_idx=idx,
                clean_image=images[idx],
                lesion_info=lesion_data[patient_id],
                df_patient=df_patient,
                save_path=save_path
            )

            if success:
                success_count += 1
                print(f"  [OK] Saved: {save_path}")

        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()

    h5_file.close()

    print("\\n" + "=" * 70)
    print(f"COMPLETE: Generated {success_count}/{len(selected_indices)} visualizations")
    print(f"Location: {save_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
