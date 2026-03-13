"""
Generate REAL Adversarial Samples - ALL FOUR ALGORITHMS
=========================================================
FGSM, PGD, C&W, DeepFool - Complete visualization
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

sys.path.insert(0, 'CheXzero')

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


class CheXzeroWrapper(nn.Module):
    """Wrapper for attacks"""
    def __init__(self, chexzero_model):
        super().__init__()
        self.chexzero = chexzero_model

    def forward(self, x):
        prob = self.chexzero(x)
        eps = 1e-7
        prob = torch.clamp(prob, eps, 1 - eps)
        pos_logit = torch.log(prob / (1 - prob))
        neg_logit = torch.zeros_like(pos_logit)
        return torch.stack([neg_logit, pos_logit], dim=1)


def fgsm_attack(model, image, mask, epsilon=8/255):
    """FGSM attack"""
    image_adv = image.clone().detach().requires_grad_(True)
    output = model(image_adv)

    with torch.no_grad():
        clean_prob = torch.softmax(model(image), dim=1)[0, 1].item()

    target = torch.zeros(output.size(0), dtype=torch.long, device=output.device)
    loss = F.cross_entropy(output, target)
    loss.backward()

    perturbation = epsilon * image_adv.grad.sign() * mask
    adv_image = torch.clamp(image + perturbation, image.min() - 0.5, image.max() + 0.5)

    with torch.no_grad():
        adv_prob = torch.softmax(model(adv_image), dim=1)[0, 1].item()

    diff = (adv_image - image).detach()
    return adv_image.detach(), {
        'clean_prob': clean_prob,
        'adv_prob': adv_prob,
        'success': adv_prob < 0.5,
        'l2_norm': torch.norm(diff).item(),
        'linf_norm': torch.max(torch.abs(diff)).item(),
        'l0_norm': torch.sum(torch.abs(diff) > 1e-6).item()
    }


def pgd_attack(model, image, mask, epsilon=8/255, alpha=2/255, steps=40):
    """PGD attack"""
    with torch.no_grad():
        clean_prob = torch.softmax(model(image), dim=1)[0, 1].item()

    adv_image = image + torch.zeros_like(image).uniform_(-epsilon, epsilon) * mask

    for _ in range(steps):
        adv_image.requires_grad = True
        output = model(adv_image)
        target = torch.zeros(output.size(0), dtype=torch.long, device=output.device)
        loss = F.cross_entropy(output, target)
        loss.backward()

        with torch.no_grad():
            adv_image = adv_image + alpha * adv_image.grad.sign() * mask
            perturbation = torch.clamp(adv_image - image, -epsilon, epsilon) * mask
            adv_image = torch.clamp(image + perturbation, image.min() - 0.5, image.max() + 0.5)
        adv_image = adv_image.detach()

    with torch.no_grad():
        adv_prob = torch.softmax(model(adv_image), dim=1)[0, 1].item()

    diff = (adv_image - image).detach()
    return adv_image, {
        'clean_prob': clean_prob,
        'adv_prob': adv_prob,
        'success': adv_prob < 0.5,
        'l2_norm': torch.norm(diff).item(),
        'linf_norm': torch.max(torch.abs(diff)).item(),
        'l0_norm': torch.sum(torch.abs(diff) > 1e-6).item()
    }


def cw_attack(model, image, mask, c=50.0, kappa=0.01, steps=1000, lr=0.05):
    """C&W attack (L2-minimized)"""
    with torch.no_grad():
        clean_prob = torch.softmax(model(image), dim=1)[0, 1].item()

    # Optimize in tanh space
    w = torch.zeros_like(image, requires_grad=True)
    optimizer = torch.optim.Adam([w], lr=lr)

    best_adv = image.clone()
    best_l2 = float('inf')

    for step in range(steps):
        adv_image = (torch.tanh(w) + 1) / 2
        adv_image = adv_image * mask + image * (1 - mask)

        output = model(adv_image)
        logits = output[0]

        # Loss: maximize normal class, minimize pneumonia class
        loss_adv = torch.clamp(logits[1] - logits[0] + kappa, min=0)

        # L2 loss
        loss_l2 = torch.sum(((adv_image - image) * mask) ** 2)

        loss = loss_l2 + c * loss_adv

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track best
        if step % 100 == 0:
            with torch.no_grad():
                adv_prob = torch.softmax(output, dim=1)[0, 1].item()
                if adv_prob < 0.5:
                    l2 = torch.norm((adv_image - image) * mask).item()
                    if l2 < best_l2:
                        best_l2 = l2
                        best_adv = adv_image.clone()

    with torch.no_grad():
        adv_prob = torch.softmax(model(best_adv), dim=1)[0, 1].item()

    diff = (best_adv - image).detach()
    return best_adv.detach(), {
        'clean_prob': clean_prob,
        'adv_prob': adv_prob,
        'success': adv_prob < 0.5,
        'l2_norm': torch.norm(diff).item(),
        'linf_norm': torch.max(torch.abs(diff)).item(),
        'l0_norm': torch.sum(torch.abs(diff) > 1e-6).item()
    }


def deepfool_attack(model, image, mask, steps=50, overshoot=0.01):
    """DeepFool attack (L2-minimized)"""
    with torch.no_grad():
        clean_prob = torch.softmax(model(image), dim=1)[0, 1].item()

    adv_image = image.clone()
    total_pert = torch.zeros_like(image)

    for _ in range(steps):
        adv_image.requires_grad = True
        output = model(adv_image)
        logits = output[0]

        if logits[1] < logits[0]:  # Already flipped
            break

        # Compute gradient
        (logits[1] - logits[0]).backward()

        with torch.no_grad():
            grad = adv_image.grad
            grad_masked = grad * mask

            # Compute minimal perturbation
            w = logits[1] - logits[0]
            grad_norm = torch.norm(grad_masked)

            if grad_norm > 1e-10:
                pert = (1 + overshoot) * w / (grad_norm ** 2) * grad_masked
                total_pert = total_pert + pert
                adv_image = image + total_pert
                adv_image = torch.clamp(adv_image, image.min() - 0.5, image.max() + 0.5)

        adv_image = adv_image.detach()

    with torch.no_grad():
        adv_prob = torch.softmax(model(adv_image), dim=1)[0, 1].item()

    diff = (adv_image - image).detach()
    return adv_image, {
        'clean_prob': clean_prob,
        'adv_prob': adv_prob,
        'success': adv_prob < 0.5,
        'l2_norm': torch.norm(diff).item(),
        'linf_norm': torch.max(torch.abs(diff)).item(),
        'l0_norm': torch.sum(torch.abs(diff) > 1e-6).item()
    }


def visualize_all_algorithms(
    patient_id, patient_idx, clean_image, lesion_mask,
    results, save_path
):
    """Visualize all 4 algorithms × 2 modes"""
    clean_img = denormalize_clip_image(clean_image.squeeze(0).cpu().numpy())
    lesion_mask_vis = lesion_mask[0].cpu().numpy()

    fig = plt.figure(figsize=(22, 16))
    gs = GridSpec(5, 5, figure=fig, hspace=0.4, wspace=0.3)

    fig.suptitle(
        f'Patient {patient_idx}: {patient_id[:20]}...\\n'
        f'COMPLETE 4-Algorithm Comparison (FGSM, PGD, C&W, DeepFool)',
        fontsize=15, fontweight='bold'
    )

    # Row 0: Original
    ax0 = fig.add_subplot(gs[0, :2])
    ax0.imshow(clean_img, cmap='gray')
    overlay = np.zeros((*lesion_mask_vis.shape, 4))
    overlay[lesion_mask_vis > 0] = [1, 0, 0, 0.5]
    ax0.imshow(overlay)
    ax0.set_title(f'Original | Lesion: {int(lesion_mask_vis.sum())} pixels', fontsize=11, fontweight='bold')
    ax0.axis('off')

    # Summary
    ax_sum = fig.add_subplot(gs[0, 2:])
    ax_sum.axis('off')
    summary = "ATTACK SUCCESS SUMMARY\\n" + "="*40 + "\\n\\n"
    for algo in ['fgsm', 'pgd', 'cw', 'deepfool']:
        algo_name = {'fgsm':'FGSM', 'pgd':'PGD', 'cw':'C&W', 'deepfool':'DeepFool'}[algo]
        lesion_success = "OK" if results[f'{algo}_lesion'][1]['success'] else "X"
        full_success = "OK" if results[f'{algo}_full'][1]['success'] else "X"
        summary += f"{algo_name:10s}: Lesion={lesion_success}  Full={full_success}\\n"

    ax_sum.text(0.05, 0.95, summary, transform=ax_sum.transAxes,
               fontsize=10, family='monospace', va='top',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # Rows 1-4: Each algorithm
    algorithms = ['fgsm', 'pgd', 'cw', 'deepfool']
    algo_names = {'fgsm':'FGSM', 'pgd':'PGD', 'cw':'C&W', 'deepfool':'DeepFool'}

    for row_idx, algo in enumerate(algorithms):
        for mode_idx, mode in enumerate(['lesion', 'full']):
            key = f'{algo}_{mode}'
            adv_img_vis = denormalize_clip_image(results[key][0].squeeze(0).cpu().numpy())
            metrics = results[key][1]

            # Perturbation
            pert = np.clip((adv_img_vis - clean_img) * 20 + 0.5, 0, 1)

            col_base = mode_idx * 2

            # Clean
            ax1 = fig.add_subplot(gs[row_idx + 1, col_base])
            ax1.imshow(clean_img, cmap='gray')
            ax1.set_title(f'{algo_names[algo]}-{mode.title()}\\nClean', fontsize=9)
            ax1.axis('off')

            # Adversarial
            ax2 = fig.add_subplot(gs[row_idx + 1, col_base + 1])
            ax2.imshow(adv_img_vis, cmap='gray')

            color = 'green' if metrics['success'] else 'red'
            text = 'OK' if metrics['success'] else 'X'
            ax2.text(0.95, 0.05, text, transform=ax2.transAxes,
                    fontsize=10, color='white', fontweight='bold',
                    ha='right', va='bottom',
                    bbox=dict(boxstyle='round', facecolor=color, alpha=0.9))

            info = f"{metrics['adv_prob']:.3f}\\nL2:{metrics['l2_norm']:.1f}"
            ax2.text(0.05, 0.95, info, transform=ax2.transAxes,
                    fontsize=8, va='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

            ax2.set_title('Adversarial', fontsize=9)
            ax2.axis('off')

        # Perturbation in last column
        ax_p = fig.add_subplot(gs[row_idx + 1, 4])
        lesion_pert = denormalize_clip_image(results[f'{algo}_lesion'][0].squeeze(0).cpu().numpy())
        pert_vis = np.clip((lesion_pert - clean_img) * 20 + 0.5, 0, 1)
        ax_p.imshow(pert_vis, cmap='hot')
        ax_p.set_title(f'{algo_names[algo]}\\nPert. (×20)', fontsize=8)
        ax_p.axis('off')

    plt.savefig(save_path, dpi=110, bbox_inches='tight')
    plt.close()


def main():
    """Generate all 4 algorithms"""
    print("=" * 70)
    print("GENERATE ALL FOUR ALGORITHMS")
    print("=" * 70)

    print("\\nLoading data...")
    h5_file = h5py.File('dataset/rsna/rsna_200_samples.h5', 'r')
    images = h5_file['cxr'][:]

    with open('dataset/rsna/rsna_200_lesion_info.json', 'r') as f:
        lesion_info = json.load(f)

    patient_ids = lesion_info['patient_ids']
    lesion_data = lesion_info['lesion_data']

    print("Loading model...")
    device = torch.device('cpu')

    from model import build_model

    model_path = 'CheXzero/checkpoints/chexzero_weights/best_64_0.0001_original_35000_0.864.pt'
    checkpoint = torch.load(model_path, map_location=device)
    chexzero = build_model(checkpoint).to(device).eval()
    model = CheXzeroWrapper(chexzero).to(device).eval()

    print("Model loaded!\\n")

    # Select 10 patients
    selected_indices = [5, 20, 40, 60, 80, 100, 120, 140, 160, 180]

    save_dir = 'results/all_four_algorithms'
    os.makedirs(save_dir, exist_ok=True)

    print(f"Generating {len(selected_indices)} patients with ALL 4 algorithms...")
    print("=" * 70)

    for i, idx in enumerate(selected_indices, 1):
        patient_id = patient_ids[idx]
        print(f"\\n[{i}/{len(selected_indices)}] Patient {idx}: {patient_id[:24]}...")

        try:
            clean_image = torch.from_numpy(images[idx]).float().unsqueeze(0).to(device)
            lesion_mask = create_lesion_mask(lesion_data[patient_id]).unsqueeze(0).to(device)
            full_mask = torch.ones_like(clean_image).to(device)

            results = {}

            print("  FGSM...", end=' ')
            results['fgsm_lesion'] = fgsm_attack(model, clean_image, lesion_mask)
            results['fgsm_full'] = fgsm_attack(model, clean_image, full_mask)
            print("OK")

            print("  PGD...", end=' ')
            results['pgd_lesion'] = pgd_attack(model, clean_image, lesion_mask)
            results['pgd_full'] = pgd_attack(model, clean_image, full_mask)
            print("OK")

            print("  C&W...", end=' ')
            results['cw_lesion'] = cw_attack(model, clean_image, lesion_mask)
            results['cw_full'] = cw_attack(model, clean_image, full_mask)
            print("OK")

            print("  DeepFool...", end=' ')
            results['deepfool_lesion'] = deepfool_attack(model, clean_image, lesion_mask)
            results['deepfool_full'] = deepfool_attack(model, clean_image, full_mask)
            print("OK")

            save_path = os.path.join(save_dir, f'patient_{idx:03d}_{patient_id[:12]}.png')
            visualize_all_algorithms(patient_id, idx, clean_image, lesion_mask, results, save_path)

            print(f"  [DONE] Saved: {save_path}")

        except Exception as e:
            print(f"  [ERROR] {e}")

    h5_file.close()

    print("\\n" + "=" * 70)
    print("COMPLETE!")
    print(f"Results: {save_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
