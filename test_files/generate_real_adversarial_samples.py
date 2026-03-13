"""
Generate REAL Adversarial Sample Images
========================================
On-the-fly generation of adversarial examples with visualization
"""

import os
import sys
import h5py
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Add CheXzero to path
sys.path.insert(0, 'CheXzero')

print("Loading dependencies...")

def denormalize_clip_image(img_tensor):
    """Denormalize CLIP image for visualization"""
    mean = np.array([0.48145466, 0.4578275, 0.40821073]).reshape(3, 1, 1)
    std = np.array([0.26862954, 0.26130258, 0.27577711]).reshape(3, 1, 1)
    img = img_tensor * std + mean
    img_gray = img.mean(axis=0)
    return np.clip(img_gray, 0, 1)


def create_lesion_mask(lesion_info_dict):
    """Create lesion mask from bbox info"""
    mask = np.zeros((3, 224, 224), dtype=np.float32)
    for bbox in lesion_info_dict['bboxes']:
        x, y, w, h = bbox
        mask[:, y:y+h, x:x+w] = 1.0
    return torch.from_numpy(mask)


class CheXzeroWrapper(nn.Module):
    """Wrapper for CheXzero model to output logits for attacks"""

    def __init__(self, chexzero_model):
        super().__init__()
        self.chexzero = chexzero_model

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) images
        Returns:
            logits: (B, 2) with [normal_logit, pneumonia_logit]
        """
        prob = self.chexzero(x)  # Get pneumonia probability

        # Convert to logits
        eps = 1e-7
        prob = torch.clamp(prob, eps, 1 - eps)
        pos_logit = torch.log(prob / (1 - prob))
        neg_logit = torch.zeros_like(pos_logit)

        logits = torch.stack([neg_logit, pos_logit], dim=1)
        return logits


def fgsm_attack(model, image, mask, epsilon=8/255):
    """
    Fast Gradient Sign Method (FGSM)

    Returns:
        adv_image: adversarial image
        metrics: dict with attack info
    """
    image_adv = image.clone().detach().requires_grad_(True)

    # Forward pass
    output = model(image_adv)

    # Get clean prediction
    with torch.no_grad():
        clean_output = model(image)
        clean_prob = torch.softmax(clean_output, dim=1)[0, 1].item()

    # Target: normal class (class 0)
    target = torch.zeros(output.size(0), dtype=torch.long, device=output.device)

    # Compute loss
    loss = F.cross_entropy(output, target)

    # Backward
    loss.backward()

    # Generate perturbation
    grad_sign = image_adv.grad.sign()
    perturbation = epsilon * grad_sign * mask

    # Create adversarial image
    adv_image = image + perturbation

    # Clamp to valid range (CLIP normalization allows negative values)
    # Just ensure reasonable bounds
    adv_image = torch.clamp(adv_image, image.min() - 0.5, image.max() + 0.5)

    # Get adversarial prediction
    with torch.no_grad():
        adv_output = model(adv_image)
        adv_prob = torch.softmax(adv_output, dim=1)[0, 1].item()

    # Compute metrics
    diff = (adv_image - image).detach()
    l2_norm = torch.norm(diff).item()
    linf_norm = torch.max(torch.abs(diff)).item()
    l0_norm = torch.sum(torch.abs(diff) > 1e-6).item()

    success = adv_prob < 0.5

    return adv_image.detach(), {
        'clean_prob': clean_prob,
        'adv_prob': adv_prob,
        'success': success,
        'l2_norm': l2_norm,
        'linf_norm': linf_norm,
        'l0_norm': l0_norm,
        'confidence_drop': clean_prob - adv_prob
    }


def pgd_attack(model, image, mask, epsilon=8/255, alpha=2/255, steps=40):
    """
    Projected Gradient Descent (PGD)

    Returns:
        adv_image: adversarial image
        metrics: dict with attack info
    """
    # Get clean prediction
    with torch.no_grad():
        clean_output = model(image)
        clean_prob = torch.softmax(clean_output, dim=1)[0, 1].item()

    # Initialize with small random perturbation
    adv_image = image.clone().detach()
    adv_image = adv_image + torch.zeros_like(adv_image).uniform_(-epsilon, epsilon) * mask

    for step in range(steps):
        adv_image.requires_grad = True

        # Forward
        output = model(adv_image)

        # Target: normal class
        target = torch.zeros(output.size(0), dtype=torch.long, device=output.device)

        # Loss
        loss = F.cross_entropy(output, target)

        # Backward
        loss.backward()

        # Update
        with torch.no_grad():
            grad_sign = adv_image.grad.sign()
            adv_image = adv_image + alpha * grad_sign * mask

            # Project back to epsilon ball
            perturbation = adv_image - image
            perturbation = torch.clamp(perturbation, -epsilon, epsilon) * mask
            adv_image = image + perturbation

            # Clamp to valid range
            adv_image = torch.clamp(adv_image, image.min() - 0.5, image.max() + 0.5)

        adv_image = adv_image.detach()

    # Get final prediction
    with torch.no_grad():
        adv_output = model(adv_image)
        adv_prob = torch.softmax(adv_output, dim=1)[0, 1].item()

    # Compute metrics
    diff = (adv_image - image).detach()
    l2_norm = torch.norm(diff).item()
    linf_norm = torch.max(torch.abs(diff)).item()
    l0_norm = torch.sum(torch.abs(diff) > 1e-6).item()

    success = adv_prob < 0.5

    return adv_image, {
        'clean_prob': clean_prob,
        'adv_prob': adv_prob,
        'success': success,
        'l2_norm': l2_norm,
        'linf_norm': linf_norm,
        'l0_norm': l0_norm,
        'confidence_drop': clean_prob - adv_prob
    }


def visualize_attack_result(
    patient_id,
    patient_idx,
    clean_image,
    lesion_mask,
    fgsm_lesion_result,
    fgsm_full_result,
    pgd_lesion_result,
    pgd_full_result,
    save_path
):
    """
    Create comprehensive visualization showing real adversarial examples
    """
    # Denormalize images
    clean_img = denormalize_clip_image(clean_image.squeeze(0).cpu().numpy())
    lesion_mask_vis = lesion_mask[0].cpu().numpy()

    # FGSM results
    fgsm_lesion_img = denormalize_clip_image(fgsm_lesion_result[0].squeeze(0).cpu().numpy())
    fgsm_full_img = denormalize_clip_image(fgsm_full_result[0].squeeze(0).cpu().numpy())

    # PGD results
    pgd_lesion_img = denormalize_clip_image(pgd_lesion_result[0].squeeze(0).cpu().numpy())
    pgd_full_img = denormalize_clip_image(pgd_full_result[0].squeeze(0).cpu().numpy())

    # Compute differences (amplified for visibility)
    fgsm_lesion_diff = np.abs(fgsm_lesion_img - clean_img)
    fgsm_full_diff = np.abs(fgsm_full_img - clean_img)
    pgd_lesion_diff = np.abs(pgd_lesion_img - clean_img)
    pgd_full_diff = np.abs(pgd_full_img - clean_img)

    # Amplified perturbations
    fgsm_lesion_pert = np.clip((fgsm_lesion_img - clean_img) * 20 + 0.5, 0, 1)
    fgsm_full_pert = np.clip((fgsm_full_img - clean_img) * 20 + 0.5, 0, 1)
    pgd_lesion_pert = np.clip((pgd_lesion_img - clean_img) * 20 + 0.5, 0, 1)
    pgd_full_pert = np.clip((pgd_full_img - clean_img) * 20 + 0.5, 0, 1)

    # Create figure
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 5, figure=fig, hspace=0.35, wspace=0.3)

    # Title
    fig.suptitle(
        f'Patient {patient_idx}: {patient_id[:20]}...\\n'
        f'REAL Adversarial Examples Generated On-The-Fly',
        fontsize=14, fontweight='bold'
    )

    # Row 0: Original image
    ax0 = fig.add_subplot(gs[0, :2])
    ax0.imshow(clean_img, cmap='gray')
    lesion_overlay = np.zeros((*lesion_mask_vis.shape, 4))
    lesion_overlay[lesion_mask_vis > 0] = [1, 0, 0, 0.5]
    ax0.imshow(lesion_overlay)
    ax0.set_title(f'Original Image\\nLesion Area: {int(lesion_mask_vis.sum())} pixels',
                  fontsize=11, fontweight='bold')
    ax0.axis('off')

    # Legend
    ax_legend = fig.add_subplot(gs[0, 2:])
    ax_legend.axis('off')
    fgsm_l_m = fgsm_lesion_result[1]
    fgsm_f_m = fgsm_full_result[1]
    pgd_l_m = pgd_lesion_result[1]
    pgd_f_m = pgd_full_result[1]

    legend_text = (
        f"FGSM-Lesion:  Clean={fgsm_l_m['clean_prob']:.3f} → Adv={fgsm_l_m['adv_prob']:.3f} "
        f"{'✓' if fgsm_l_m['success'] else '✗'}\\n"
        f"              L2={fgsm_l_m['l2_norm']:.2f}, L∞={fgsm_l_m['linf_norm']:.4f}\\n\\n"
        f"FGSM-Full:    Clean={fgsm_f_m['clean_prob']:.3f} → Adv={fgsm_f_m['adv_prob']:.3f} "
        f"{'✓' if fgsm_f_m['success'] else '✗'}\\n"
        f"              L2={fgsm_f_m['l2_norm']:.2f}, L∞={fgsm_f_m['linf_norm']:.4f}\\n\\n"
        f"PGD-Lesion:   Clean={pgd_l_m['clean_prob']:.3f} → Adv={pgd_l_m['adv_prob']:.3f} "
        f"{'✓' if pgd_l_m['success'] else '✗'}\\n"
        f"              L2={pgd_l_m['l2_norm']:.2f}, L∞={pgd_l_m['linf_norm']:.4f}\\n\\n"
        f"PGD-Full:     Clean={pgd_f_m['clean_prob']:.3f} → Adv={pgd_f_m['adv_prob']:.3f} "
        f"{'✓' if pgd_f_m['success'] else '✗'}\\n"
        f"              L2={pgd_f_m['l2_norm']:.2f}, L∞={pgd_f_m['linf_norm']:.4f}"
    )

    ax_legend.text(0.05, 0.95, legend_text, transform=ax_legend.transAxes,
                   fontsize=9, verticalalignment='top', family='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # Row 1: FGSM-Lesion
    attacks = [
        ('FGSM-Lesion', clean_img, fgsm_lesion_img, fgsm_lesion_diff, fgsm_lesion_pert, fgsm_l_m),
        ('FGSM-Full', clean_img, fgsm_full_img, fgsm_full_diff, fgsm_full_pert, fgsm_f_m),
        ('PGD-Lesion', clean_img, pgd_lesion_img, pgd_lesion_diff, pgd_lesion_pert, pgd_l_m),
        ('PGD-Full', clean_img, pgd_full_img, pgd_full_diff, pgd_full_pert, pgd_f_m),
    ]

    for row_idx, (name, clean, adv, diff, pert, metrics) in enumerate(attacks):
        if row_idx < 2:
            row = 1
            col_offset = row_idx * 2
        else:
            row = 2
            col_offset = (row_idx - 2) * 2

        # Column 1: Clean
        ax1 = fig.add_subplot(gs[row, col_offset])
        ax1.imshow(clean, cmap='gray')
        ax1.set_title(f'{name}\\nClean', fontsize=9)
        ax1.axis('off')

        # Column 2: Adversarial
        ax2 = fig.add_subplot(gs[row, col_offset + 1])
        ax2.imshow(adv, cmap='gray')

        success_color = 'green' if metrics['success'] else 'red'
        success_text = 'OK' if metrics['success'] else 'X'
        badge_props = dict(boxstyle='round', facecolor=success_color, alpha=0.9)
        ax2.text(0.95, 0.05, success_text, transform=ax2.transAxes,
                fontsize=10, color='white', fontweight='bold',
                ha='right', va='bottom', bbox=badge_props)

        ax2.set_title(f'Adversarial\\nProb: {metrics["adv_prob"]:.3f}', fontsize=9)
        ax2.axis('off')

    # Last column: Perturbation visualizations
    ax_p1 = fig.add_subplot(gs[1, 4])
    ax_p1.imshow(fgsm_lesion_pert, cmap='hot')
    ax_p1.set_title('FGSM-Lesion\\nPerturbation (×20)', fontsize=8)
    ax_p1.axis('off')

    ax_p2 = fig.add_subplot(gs[2, 4])
    ax_p2.imshow(pgd_lesion_pert, cmap='hot')
    ax_p2.set_title('PGD-Lesion\\nPerturbation (×20)', fontsize=8)
    ax_p2.axis('off')

    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()


def main():
    """Generate real adversarial samples"""
    print("=" * 70)
    print("REAL ADVERSARIAL SAMPLE GENERATION")
    print("=" * 70)

    # Load data
    print("\\nLoading data...")
    h5_file = h5py.File('dataset/rsna/rsna_200_samples.h5', 'r')
    images = h5_file['cxr'][:]

    with open('dataset/rsna/rsna_200_lesion_info.json', 'r') as f:
        lesion_info = json.load(f)

    patient_ids = lesion_info['patient_ids']
    lesion_data = lesion_info['lesion_data']

    # Load model
    print("Loading CheXzero model...")
    device = torch.device('cpu')  # Use CPU to avoid compatibility issues
    print(f"Device: {device}")

    try:
        # Import model builder
        from model import build_model
        import clip

        model_path = 'CheXzero/checkpoints/chexzero_weights/best_64_0.0001_original_35000_0.864.pt'

        # Load checkpoint (state_dict)
        checkpoint = torch.load(model_path, map_location=device)

        # Build CheXzero from checkpoint
        chexzero = build_model(checkpoint)
        chexzero = chexzero.to(device)
        chexzero.eval()

        # Wrap for attacks
        model = CheXzeroWrapper(chexzero).to(device)
        model.eval()

        print("Model loaded successfully!")

    except Exception as e:
        print(f"ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        return

    # Select 15 diverse patients
    selected_indices = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90,
                       100, 120, 140, 160, 180]

    save_dir = 'results/real_adversarial_samples'
    os.makedirs(save_dir, exist_ok=True)

    print(f"\\nGenerating adversarial samples for {len(selected_indices)} patients...")
    print("=" * 70)

    for i, idx in enumerate(selected_indices, 1):
        patient_id = patient_ids[idx]
        print(f"\\n[{i}/{len(selected_indices)}] Patient {idx}: {patient_id[:24]}...")

        try:
            # Prepare data
            clean_image = torch.from_numpy(images[idx]).float().unsqueeze(0).to(device)
            lesion_mask = create_lesion_mask(lesion_data[patient_id]).unsqueeze(0).to(device)
            full_mask = torch.ones_like(clean_image).to(device)

            # Run attacks
            print("  Running FGSM-Lesion...", end=' ')
            fgsm_lesion_adv, fgsm_lesion_metrics = fgsm_attack(
                model, clean_image, lesion_mask, epsilon=8/255
            )
            print(f"{'SUCCESS' if fgsm_lesion_metrics['success'] else 'FAILED'}")

            print("  Running FGSM-Full...", end=' ')
            fgsm_full_adv, fgsm_full_metrics = fgsm_attack(
                model, clean_image, full_mask, epsilon=8/255
            )
            print(f"{'SUCCESS' if fgsm_full_metrics['success'] else 'FAILED'}")

            print("  Running PGD-Lesion...", end=' ')
            pgd_lesion_adv, pgd_lesion_metrics = pgd_attack(
                model, clean_image, lesion_mask, epsilon=8/255, alpha=2/255, steps=40
            )
            print(f"{'SUCCESS' if pgd_lesion_metrics['success'] else 'FAILED'}")

            print("  Running PGD-Full...", end=' ')
            pgd_full_adv, pgd_full_metrics = pgd_attack(
                model, clean_image, full_mask, epsilon=8/255, alpha=2/255, steps=40
            )
            print(f"{'SUCCESS' if pgd_full_metrics['success'] else 'FAILED'}")

            # Visualize
            save_path = os.path.join(save_dir, f'patient_{idx:03d}_{patient_id[:12]}.png')
            visualize_attack_result(
                patient_id=patient_id,
                patient_idx=idx,
                clean_image=clean_image,
                lesion_mask=lesion_mask,
                fgsm_lesion_result=(fgsm_lesion_adv, fgsm_lesion_metrics),
                fgsm_full_result=(fgsm_full_adv, fgsm_full_metrics),
                pgd_lesion_result=(pgd_lesion_adv, pgd_lesion_metrics),
                pgd_full_result=(pgd_full_adv, pgd_full_metrics),
                save_path=save_path
            )

            print(f"  [OK] Saved: {save_path}")

        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()

    h5_file.close()

    print("\\n" + "=" * 70)
    print("COMPLETE!")
    print(f"Real adversarial samples saved to: {save_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
