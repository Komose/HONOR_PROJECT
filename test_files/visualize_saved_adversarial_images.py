"""
Visualize Saved Adversarial Images
===================================
Load .npy files and create comparison visualizations
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import glob


def denormalize_clip_image(img_array):
    """Denormalize CLIP image"""
    mean = np.array([0.48145466, 0.4578275, 0.40821073]).reshape(3, 1, 1)
    std = np.array([0.26862954, 0.26130258, 0.27577711]).reshape(3, 1, 1)
    img = img_array * std + mean
    img_gray = img.mean(axis=0)
    return np.clip(img_gray, 0, 1)


def visualize_patient_attacks(patient_dir, save_path):
    """Visualize all attacks for one patient"""

    patient_id = os.path.basename(patient_dir)

    # Load clean image
    clean_path = os.path.join(patient_dir, 'clean.npy')
    if not os.path.exists(clean_path):
        print(f"  Clean image not found in {patient_dir}")
        return False

    clean_img = np.load(clean_path)
    clean_vis = denormalize_clip_image(clean_img)

    # Find available algorithms
    available_files = glob.glob(os.path.join(patient_dir, '*_lesion.npy'))
    algorithms = [os.path.basename(f).replace('_lesion.npy', '') for f in available_files
                  if 'mask' not in f and 'clean' not in f]

    if not algorithms:
        print(f"  No attack results found in {patient_dir}")
        return False

    print(f"  Found algorithms: {algorithms}")

    # Create figure
    n_algos = len(algorithms)
    fig = plt.figure(figsize=(20, 4 * n_algos + 3))
    gs = GridSpec(n_algos + 1, 7, figure=fig, hspace=0.4, wspace=0.3)

    fig.suptitle(f'Patient: {patient_id}\\nReal Adversarial Attack Results',
                 fontsize=15, fontweight='bold')

    # Row 0: Original image with masks
    ax0 = fig.add_subplot(gs[0, :2])
    ax0.imshow(clean_vis, cmap='gray')
    ax0.set_title('Clean Image', fontsize=12, fontweight='bold')
    ax0.axis('off')

    # Show lesion mask
    lesion_mask_path = os.path.join(patient_dir, 'mask_lesion.npy')
    if os.path.exists(lesion_mask_path):
        ax1 = fig.add_subplot(gs[0, 2])
        lesion_mask = np.load(lesion_mask_path)
        lesion_mask_vis = lesion_mask[0] if lesion_mask.ndim == 3 else lesion_mask

        ax1.imshow(clean_vis, cmap='gray')
        overlay = np.zeros((*lesion_mask_vis.shape, 4))
        overlay[lesion_mask_vis > 0] = [1, 0, 0, 0.5]
        ax1.imshow(overlay)
        ax1.set_title(f'Lesion Mask\\n{int(lesion_mask_vis.sum())} pixels', fontsize=10)
        ax1.axis('off')

    # Show random patch mask
    random_mask_path = os.path.join(patient_dir, 'mask_random_patch.npy')
    if os.path.exists(random_mask_path):
        ax2 = fig.add_subplot(gs[0, 3])
        random_mask = np.load(random_mask_path)
        random_mask_vis = random_mask[0] if random_mask.ndim == 3 else random_mask

        ax2.imshow(clean_vis, cmap='gray')
        overlay = np.zeros((*random_mask_vis.shape, 4))
        overlay[random_mask_vis > 0] = [0, 0, 1, 0.5]
        ax2.imshow(overlay)
        ax2.set_title(f'Random Patch\\n{int(random_mask_vis.sum())} pixels', fontsize=10)
        ax2.axis('off')

    # Legend
    ax_legend = fig.add_subplot(gs[0, 4:])
    ax_legend.axis('off')
    legend_text = (
        "VISUALIZATION GUIDE\\n"
        "==================\\n"
        "Each row shows one algorithm:\\n"
        "  - Clean / Adv / Difference\\n"
        "  - For 3 modes: Lesion, Random, Full\\n\\n"
        "Difference images:\\n"
        "  - Amplified 30× for visibility\\n"
        "  - Brighter = larger change"
    )
    ax_legend.text(0.05, 0.95, legend_text, transform=ax_legend.transAxes,
                  fontsize=9, verticalalignment='top', family='monospace',
                  bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Rows 1+: Each algorithm
    modes = ['lesion', 'random_patch', 'full']
    mode_names = {'lesion': 'Lesion', 'random_patch': 'Random', 'full': 'Full'}

    for algo_idx, algo in enumerate(algorithms):
        for mode_idx, mode in enumerate(modes):
            adv_path = os.path.join(patient_dir, f'{algo}_{mode}.npy')

            if not os.path.exists(adv_path):
                continue

            # Load adversarial image
            adv_img = np.load(adv_path)
            adv_vis = denormalize_clip_image(adv_img)

            # Compute difference
            diff = adv_vis - clean_vis
            diff_amplified = np.clip(diff * 30 + 0.5, 0, 1)

            col_base = mode_idx * 2

            # Clean
            ax1 = fig.add_subplot(gs[algo_idx + 1, col_base])
            ax1.imshow(clean_vis, cmap='gray')
            if mode_idx == 0:
                ax1.set_ylabel(algo.upper(), fontsize=11, fontweight='bold', rotation=0,
                             labelpad=40, va='center')
            ax1.set_title(f'{mode_names[mode]}\\nClean', fontsize=9)
            ax1.axis('off')

            # Adversarial
            ax2 = fig.add_subplot(gs[algo_idx + 1, col_base + 1])
            ax2.imshow(adv_vis, cmap='gray')

            # Compute L2 norm
            l2_norm = np.linalg.norm(diff)
            linf_norm = np.max(np.abs(diff))

            info_text = f"L2: {l2_norm:.2f}\\nL∞: {linf_norm:.4f}"
            props = dict(boxstyle='round', facecolor='white', alpha=0.85)
            ax2.text(0.03, 0.97, info_text, transform=ax2.transAxes,
                    fontsize=8, verticalalignment='top', bbox=props, family='monospace')

            ax2.set_title('Adversarial', fontsize=9)
            ax2.axis('off')

        # Perturbation visualization in last column
        lesion_adv_path = os.path.join(patient_dir, f'{algo}_lesion.npy')
        if os.path.exists(lesion_adv_path):
            ax_pert = fig.add_subplot(gs[algo_idx + 1, 6])
            lesion_adv = denormalize_clip_image(np.load(lesion_adv_path))
            pert = np.clip((lesion_adv - clean_vis) * 30 + 0.5, 0, 1)
            ax_pert.imshow(pert, cmap='hot')
            ax_pert.set_title('Perturbation\\n(Lesion, ×30)', fontsize=8)
            ax_pert.axis('off')

    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()

    return True


def main():
    """Visualize all saved adversarial images"""
    print("=" * 70)
    print("VISUALIZE SAVED ADVERSARIAL IMAGES")
    print("=" * 70)

    source_dir = 'results/adversarial_images_saved'
    output_dir = 'results/adversarial_visualizations'
    os.makedirs(output_dir, exist_ok=True)

    # Find all patient directories
    patient_dirs = glob.glob(os.path.join(source_dir, '*-*'))

    if not patient_dirs:
        print(f"\\nNo patient directories found in {source_dir}")
        return

    print(f"\\nFound {len(patient_dirs)} patient(s) with saved images")
    print("=" * 70)

    success_count = 0
    for i, patient_dir in enumerate(patient_dirs, 1):
        patient_id = os.path.basename(patient_dir)
        print(f"\\n[{i}/{len(patient_dirs)}] Patient: {patient_id}...")

        try:
            save_path = os.path.join(output_dir, f'{patient_id}.png')
            success = visualize_patient_attacks(patient_dir, save_path)

            if success:
                success_count += 1
                print(f"  [OK] Saved: {save_path}")

        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()

    print("\\n" + "=" * 70)
    print(f"COMPLETE: Generated {success_count}/{len(patient_dirs)} visualizations")
    print(f"Output: {output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
