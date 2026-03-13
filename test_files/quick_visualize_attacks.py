"""
Quick Visualization - Show Real Attack Differences
==================================================
Generate a few examples quickly to demonstrate real adversarial perturbations
"""

import os
import sys
import h5py
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Add CheXzero to path
sys.path.insert(0, 'CheXzero')
from model import build_model
import clip


def denormalize_clip_image(img_tensor):
    """Denormalize for visualization"""
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


def fgsm_attack(model, image, mask, epsilon=8/255, target_prob=0.5):
    """
    Simple FGSM attack implementation

    Args:
        model: CheXzero model wrapped for attack
        image: (1, 3, H, W) clean image
        mask: (1, 3, H, W) binary mask
        epsilon: perturbation magnitude

    Returns:
        adv_image: adversarial image
        success: bool
    """
    image = image.clone().detach().requires_grad_(True)

    # Get clean prediction
    output = model(image)
    clean_prob = output[0, 1].item()  # Pneumonia probability

    # Target: reduce pneumonia probability below 0.5
    target = torch.zeros_like(output)
    target[0, 0] = 1.0  # Target normal class

    # Compute loss
    loss = F.cross_entropy(output, target.argmax(dim=1))

    # Compute gradient
    loss.backward()

    # Generate perturbation
    grad_sign = image.grad.sign()

    # Apply mask
    perturbation = epsilon * grad_sign * mask

    # Create adversarial image
    adv_image = image + perturbation

    # Get adversarial prediction
    with torch.no_grad():
        adv_output = model(adv_image)
        adv_prob = adv_output[0, 1].item()

    success = adv_prob < 0.5

    return adv_image.detach(), success, clean_prob, adv_prob


def visualize_attack_comparison(
    patient_id,
    patient_idx,
    clean_image,
    lesion_info,
    model,
    device,
    save_path
):
    """
    Create a simple but clear visualization showing attack effects
    """
    # Prepare data
    lesion_mask = create_lesion_mask(lesion_info)
    clean_tensor = torch.from_numpy(clean_image).float().unsqueeze(0).to(device)
    lesion_mask_tensor = lesion_mask.unsqueeze(0).to(device)

    # Wrap model for attack
    from unified_attack_framework_fixed import CheXzeroForAttack
    attack_model = CheXzeroForAttack(model).to(device)
    attack_model.eval()

    # Run FGSM attack on lesion
    adv_lesion, success_lesion, clean_prob, adv_prob_lesion = fgsm_attack(
        attack_model, clean_tensor, lesion_mask_tensor, epsilon=8/255
    )

    # Run FGSM attack on full image
    full_mask = torch.ones_like(clean_tensor).to(device)
    adv_full, success_full, _, adv_prob_full = fgsm_attack(
        attack_model, clean_tensor, full_mask, epsilon=8/255
    )

    # Convert to numpy for visualization
    clean_np = denormalize_clip_image(clean_image)
    adv_lesion_np = denormalize_clip_image(adv_lesion.squeeze(0).cpu().numpy())
    adv_full_np = denormalize_clip_image(adv_full.squeeze(0).cpu().numpy())
    lesion_mask_np = lesion_mask[0].numpy()

    # Compute perturbations (amplified for visibility)
    pert_lesion = (adv_lesion_np - clean_np) * 20 + 0.5
    pert_full = (adv_full_np - clean_np) * 20 + 0.5
    pert_lesion = np.clip(pert_lesion, 0, 1)
    pert_full = np.clip(pert_full, 0, 1)

    # Create visualization
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    fig.suptitle(f'Patient {patient_idx}: {patient_id[:20]}...\\n'
                 f'Clean Probability: {clean_prob:.4f}',
                 fontsize=14, fontweight='bold')

    # Row 0: Original images
    axes[0, 0].imshow(clean_np, cmap='gray')
    axes[0, 0].set_title('Clean Image', fontsize=11, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(clean_np, cmap='gray')
    lesion_overlay = np.zeros((*lesion_mask_np.shape, 4))
    lesion_overlay[lesion_mask_np > 0] = [1, 0, 0, 0.5]
    axes[0, 1].imshow(lesion_overlay)
    axes[0, 1].set_title(f'Lesion Region\\n({int(lesion_mask_np.sum())} pixels)',
                        fontsize=11, fontweight='bold')
    axes[0, 1].axis('off')

    axes[0, 2].axis('off')
    axes[0, 3].axis('off')

    # Row 1: Lesion attack
    axes[1, 0].imshow(clean_np, cmap='gray')
    axes[1, 0].set_title('Clean', fontsize=10)
    axes[1, 0].axis('off')

    axes[1, 1].imshow(adv_lesion_np, cmap='gray')
    color = 'green' if success_lesion else 'red'
    symbol = '✓' if success_lesion else '✗'
    axes[1, 1].set_title(f'After Lesion Attack {symbol}\\nProb: {adv_prob_lesion:.4f}',
                        fontsize=10, color=color, fontweight='bold')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(pert_lesion, cmap='hot')
    axes[1, 2].set_title('Perturbation (×20)', fontsize=10)
    axes[1, 2].axis('off')

    diff_lesion = np.abs(adv_lesion_np - clean_np)
    axes[1, 3].imshow(diff_lesion, cmap='hot', vmin=0, vmax=0.1)
    axes[1, 3].set_title('Absolute Difference', fontsize=10)
    axes[1, 3].axis('off')

    # Row 2: Full image attack
    axes[2, 0].imshow(clean_np, cmap='gray')
    axes[2, 0].set_title('Clean', fontsize=10)
    axes[2, 0].axis('off')

    axes[2, 1].imshow(adv_full_np, cmap='gray')
    color = 'green' if success_full else 'red'
    symbol = '✓' if success_full else '✗'
    axes[2, 1].set_title(f'After Full Attack {symbol}\\nProb: {adv_prob_full:.4f}',
                        fontsize=10, color=color, fontweight='bold')
    axes[2, 1].axis('off')

    axes[2, 2].imshow(pert_full, cmap='hot')
    axes[2, 2].set_title('Perturbation (×20)', fontsize=10)
    axes[2, 2].axis('off')

    diff_full = np.abs(adv_full_np - clean_np)
    axes[2, 3].imshow(diff_full, cmap='hot', vmin=0, vmax=0.1)
    axes[2, 3].set_title('Absolute Difference', fontsize=10)
    axes[2, 3].axis('off')

    # Add row labels
    fig.text(0.02, 0.65, 'LESION\\nATTACK', fontsize=12, fontweight='bold',
             va='center', ha='center', rotation=90)
    fig.text(0.02, 0.35, 'FULL\\nATTACK', fontsize=12, fontweight='bold',
             va='center', ha='center', rotation=90)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()

    return success_lesion, success_full


def main():
    """Generate quick visualizations"""
    print("=" * 70)
    print("QUICK ATTACK VISUALIZATION - FGSM Examples")
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
    print("Loading model...")
    device = torch.device('cpu')  # Use CPU to avoid CLIP CUDA issues
    print(f"Device: {device}")

    model_path = 'CheXzero/checkpoints/best_64_5e-05_original_22000_0.862.pt'
    clip_model, _ = clip.load("ViT-B/32", device=device)
    chexzero = build_model(clip_model)
    chexzero.load_state_dict(torch.load(model_path, map_location=device))
    chexzero.eval()

    # Select 10 patients
    selected_indices = [0, 10, 20, 30, 40, 50, 60, 80, 100, 120]

    save_dir = 'results/quick_attack_viz'
    os.makedirs(save_dir, exist_ok=True)

    print(f"\\nGenerating {len(selected_indices)} visualizations...")
    print("=" * 70)

    for i, idx in enumerate(selected_indices, 1):
        patient_id = patient_ids[idx]
        print(f"\\n[{i}/{len(selected_indices)}] Patient {idx}: {patient_id[:24]}...")

        try:
            save_path = os.path.join(save_dir, f'attack_{idx:03d}_{patient_id[:12]}.png')

            success_lesion, success_full = visualize_attack_comparison(
                patient_id=patient_id,
                patient_idx=idx,
                clean_image=images[idx],
                lesion_info=lesion_data[patient_id],
                model=chexzero,
                device=device,
                save_path=save_path
            )

            print(f"  Lesion attack: {'SUCCESS' if success_lesion else 'FAILED'}")
            print(f"  Full attack:   {'SUCCESS' if success_full else 'FAILED'}")
            print(f"  Saved: {save_path}")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    h5_file.close()

    print("\\n" + "=" * 70)
    print(f"DONE! Check {save_dir}/ for results")
    print("=" * 70)


if __name__ == "__main__":
    main()
