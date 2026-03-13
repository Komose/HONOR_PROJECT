"""
Use Existing Framework to Generate Real Adversarial Visualizations
===================================================================
"""

import os
import sys
import h5py
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Import from existing framework
from rsna_attack_framework import CheXzeroWrapper

print("Loading dependencies...")


def denormalize_clip_image(img_tensor):
    """Denormalize CLIP image"""
    mean = np.array([0.48145466, 0.4578275, 0.40821073]).reshape(3, 1, 1)
    std = np.array([0.26862954, 0.26130258, 0.27577711]).reshape(3, 1, 1)
    img = img_tensor * std + mean
    img_gray = img.mean(axis=0)
    return np.clip(img_gray, 0, 1)


def create_lesion_mask(lesion_info_dict):
    """Create lesion mask"""
    mask = np.zeros((3, 224, 224), dtype=np.float32)
    for bbox in lesion_info_dict['bboxes']:
        x, y, w, h = bbox
        mask[:, y:y+h, x:x+w] = 1.0
    return torch.from_numpy(mask)


def simple_fgsm(model, image, mask, epsilon=8/255):
    """Simple FGSM using existing wrapper"""
    image_adv = image.clone().requires_grad_(True)

    # Forward
    prob = model(image_adv)

    # Backward (minimize probability)
    loss = prob.mean()
    loss.backward()

    # Generate perturbation
    grad_sign = image_adv.grad.sign()
    perturbation = epsilon * grad_sign * mask

    adv_image = image + perturbation

    # Get results
    with torch.no_grad():
        clean_prob = model(image).item()
        adv_prob = model(adv_image).item()

    diff = (adv_image - image).detach()

    return adv_image.detach(), {
        'clean_prob': clean_prob,
        'adv_prob': adv_prob,
        'success': adv_prob < 0.5,
        'l2_norm': torch.norm(diff).item(),
        'linf_norm': torch.max(torch.abs(diff)).item()
    }


def visualize_patient(
    patient_id, patient_idx, clean_image, lesion_mask,
    fgsm_lesion_result, fgsm_full_result, save_path
):
    """Visualize FGSM results"""
    clean_img = denormalize_clip_image(clean_image.squeeze(0).cpu().numpy())
    lesion_mask_vis = lesion_mask[0].cpu().numpy()

    fgsm_lesion_img = denormalize_clip_image(fgsm_lesion_result[0].squeeze(0).cpu().numpy())
    fgsm_full_img = denormalize_clip_image(fgsm_full_result[0].squeeze(0).cpu().numpy())

    # Perturbations
    pert_lesion = np.clip((fgsm_lesion_img - clean_img) * 30 + 0.5, 0, 1)
    pert_full = np.clip((fgsm_full_img - clean_img) * 30 + 0.5, 0, 1)

    # Create figure
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))

    metrics_l = fgsm_lesion_result[1]
    metrics_f = fgsm_full_result[1]

    fig.suptitle(
        f'Patient {patient_idx}: {patient_id[:20]}...\\n'
        f'Clean Prob: {metrics_l["clean_prob"]:.4f}',
        fontsize=14, fontweight='bold'
    )

    # Row 0: Original
    axes[0, 0].imshow(clean_img, cmap='gray')
    axes[0, 0].set_title('Original', fontsize=11, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(clean_img, cmap='gray')
    overlay = np.zeros((*lesion_mask_vis.shape, 4))
    overlay[lesion_mask_vis > 0] = [1, 0, 0, 0.5]
    axes[0, 1].imshow(overlay)
    axes[0, 1].set_title(f'Lesion Mask\\n{int(lesion_mask_vis.sum())} pixels', fontsize=11)
    axes[0, 1].axis('off')

    axes[0, 2].axis('off')
    axes[0, 3].axis('off')

    # Row 1: FGSM-Lesion
    axes[1, 0].imshow(clean_img, cmap='gray')
    axes[1, 0].set_title('FGSM-Lesion\\nClean', fontsize=10)
    axes[1, 0].axis('off')

    axes[1, 1].imshow(fgsm_lesion_img, cmap='gray')
    color = 'green' if metrics_l['success'] else 'red'
    text = 'SUCCESS' if metrics_l['success'] else 'FAILED'
    axes[1, 1].text(0.5, 0.05, text, transform=axes[1, 1].transAxes,
                   fontsize=10, ha='center', color='white', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor=color, alpha=0.9))
    axes[1, 1].set_title(f'Adversarial\\nProb: {metrics_l["adv_prob"]:.4f}', fontsize=10)
    axes[1, 1].axis('off')

    axes[1, 2].imshow(pert_lesion, cmap='hot')
    axes[1, 2].set_title(f'Perturbation (×30)\\nL2: {metrics_l["l2_norm"]:.2f}', fontsize=10)
    axes[1, 2].axis('off')

    diff_lesion = np.abs(fgsm_lesion_img - clean_img)
    axes[1, 3].imshow(diff_lesion, cmap='hot', vmin=0, vmax=0.1)
    axes[1, 3].set_title(f'Difference\\nL∞: {metrics_l["linf_norm"]:.4f}', fontsize=10)
    axes[1, 3].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()


def main():
    """Generate real adversarial samples using existing framework"""
    print("=" * 70)
    print("REAL ADVERSARIAL SAMPLES - USING EXISTING FRAMEWORK")
    print("=" * 70)

    print("\\nLoading data...")
    h5_file = h5py.File('dataset/rsna/rsna_200_samples.h5', 'r')
    images = h5_file['cxr'][:]

    with open('dataset/rsna/rsna_200_lesion_info.json', 'r') as f:
        lesion_info = json.load(f)

    patient_ids = lesion_info['patient_ids']
    lesion_data = lesion_info['lesion_data']

    print("Loading CheXzero model with existing wrapper...")
    device = torch.device('cpu')

    model = CheXzeroWrapper(
        model_path='CheXzero/checkpoints/chexzero_weights/best_64_0.0001_original_35000_0.864.pt',
        device=device
    )
    model.eval()

    print("Model loaded!\\n")

    # Select 10 patients
    selected_indices = [5, 20, 40, 60, 80, 100, 120, 140, 160, 180]

    save_dir = 'results/real_adversarial_final'
    os.makedirs(save_dir, exist_ok=True)

    print(f"Generating {len(selected_indices)} real adversarial samples...")
    print("=" * 70)

    for i, idx in enumerate(selected_indices, 1):
        patient_id = patient_ids[idx]
        print(f"\\n[{i}/{len(selected_indices)}] Patient {idx}: {patient_id[:24]}...")

        try:
            clean_image = torch.from_numpy(images[idx]).float().unsqueeze(0).to(device)
            lesion_mask = create_lesion_mask(lesion_data[patient_id]).unsqueeze(0).to(device)
            full_mask = torch.ones_like(clean_image).to(device)

            print("  Running FGSM-Lesion...", end=' ')
            fgsm_lesion = simple_fgsm(model, clean_image, lesion_mask)
            print(f"{'SUCCESS' if fgsm_lesion[1]['success'] else 'FAILED'}")

            print("  Running FGSM-Full...", end=' ')
            fgsm_full = simple_fgsm(model, clean_image, full_mask)
            print(f"{'SUCCESS' if fgsm_full[1]['success'] else 'FAILED'}")

            save_path = os.path.join(save_dir, f'patient_{idx:03d}_{patient_id[:12]}.png')
            visualize_patient(patient_id, idx, clean_image, lesion_mask,
                            fgsm_lesion, fgsm_full, save_path)

            print(f"  [OK] Saved: {save_path}")

        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()

    h5_file.close()

    print("\\n" + "=" * 70)
    print("COMPLETE!")
    print(f"Results: {save_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
