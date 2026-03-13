"""
RSNA Adversarial Attack Framework
==================================

This module implements lesion-targeted and full-image adversarial attacks
for evaluating CheXzero's robustness on RSNA Pneumonia dataset.

Key Features:
- Lesion-targeted attacks (perturb only lesion regions)
- Full-image attacks (baseline comparison)
- Support for FGSM, PGD, C&W, DeepFool
- Integrated with CheXzero model
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import h5py
import json
from tqdm import tqdm
import pandas as pd

# Add CheXzero to path
sys.path.insert(0, 'CheXzero')
from model import build_model
import clip


class RSNADataset(data.Dataset):
    """Dataset for RSNA images with lesion masks."""

    def __init__(self, h5_path, lesion_info_path):
        super().__init__()
        self.h5_file = h5py.File(h5_path, 'r')
        self.images = self.h5_file['cxr']

        with open(lesion_info_path, 'r') as f:
            lesion_data = json.load(f)

        self.patient_ids = lesion_data['patient_ids']
        self.lesion_info = lesion_data['lesion_data']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.from_numpy(self.images[idx])  # (3, 224, 224)
        patient_id = self.patient_ids[idx]

        # Generate lesion mask
        mask = self._generate_mask(patient_id)

        return {
            'image': image,
            'mask': mask,
            'patient_id': patient_id,
            'num_lesions': self.lesion_info[patient_id]['num_lesions']
        }

    def _generate_mask(self, patient_id):
        """Generate binary mask from lesion bounding boxes."""
        mask = torch.zeros(224, 224, dtype=torch.float32)

        bboxes = self.lesion_info[patient_id]['bboxes']
        for bbox in bboxes:
            x, y, w, h = bbox
            mask[y:y+h, x:x+w] = 1.0

        # Expand to 3 channels: (3, 224, 224)
        mask = mask.unsqueeze(0).expand(3, -1, -1)

        return mask

    def close(self):
        self.h5_file.close()


class CheXzeroWrapper(nn.Module):
    """
    Wrapper around CheXzero model for adversarial attacks.

    Outputs: Pneumonia probability (softmax of positive vs negative templates)
    """

    def __init__(self, model_path, device='cuda'):
        super().__init__()
        self.device = device

        # Load CheXzero model
        checkpoint = torch.load(model_path, map_location=device)
        self.model = build_model(checkpoint)
        self.model = self.model.to(device)
        self.model.eval()

        # Precompute text embeddings for Pneumonia
        self._precompute_text_embeddings()

    def _precompute_text_embeddings(self):
        """Precompute text embeddings for positive and negative templates."""
        with torch.no_grad():
            # Positive: "Pneumonia"
            pos_text = clip.tokenize(["Pneumonia"], context_length=77).to(self.device)
            pos_features = self.model.encode_text(pos_text)
            self.pos_text_features = pos_features / pos_features.norm(dim=-1, keepdim=True)

            # Negative: "Normal" (IMPROVED template)
            neg_text = clip.tokenize(["Normal"], context_length=77).to(self.device)
            neg_features = self.model.encode_text(neg_text)
            self.neg_text_features = neg_features / neg_features.norm(dim=-1, keepdim=True)

    def forward(self, images):
        """
        Forward pass.

        Args:
            images: (B, 3, 224, 224) tensor

        Returns:
            probs: (B,) tensor of Pneumonia probabilities
        """
        # Encode images
        image_features = self.model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Compute logits for positive and negative templates
        pos_logits = (image_features @ self.pos_text_features.T).squeeze(-1)  # (B,)
        neg_logits = (image_features @ self.neg_text_features.T).squeeze(-1)  # (B,)

        # Softmax to get probabilities
        pos_exp = torch.exp(pos_logits)
        neg_exp = torch.exp(neg_logits)
        probs = pos_exp / (pos_exp + neg_exp)

        return probs

    def predict(self, images):
        """Prediction without gradients."""
        with torch.no_grad():
            return self.forward(images)


def fgsm_attack(model, images, masks, epsilon, targeted=False, attack_mode='lesion'):
    """
    Fast Gradient Sign Method (FGSM) attack.

    Args:
        model: CheXzeroWrapper instance
        images: (B, 3, H, W) tensor, clean images
        masks: (B, 3, H, W) tensor, lesion masks (1 for lesion, 0 for background)
        epsilon: perturbation budget (in normalized space, e.g., 8/255)
        targeted: if True, maximize probability; if False, minimize probability
        attack_mode: 'lesion' or 'full'

    Returns:
        adv_images: adversarial images
        perturbations: perturbations applied
    """
    images = images.clone().detach().requires_grad_(True)

    # Forward pass
    outputs = model(images)

    # Loss: maximize or minimize Pneumonia probability
    if targeted:
        loss = outputs.mean()  # Maximize
    else:
        loss = -outputs.mean()  # Minimize

    # Backward pass
    model.zero_grad()
    loss.backward()

    # Get gradient
    grad = images.grad.data

    # Apply mask if lesion attack
    if attack_mode == 'lesion':
        grad = grad * masks  # Zero out gradients outside lesion

    # FGSM step
    sign_grad = grad.sign()
    perturbations = epsilon * sign_grad

    # Apply perturbation
    adv_images = images + perturbations

    # Apply mask to perturbation (ensure only lesion is modified)
    if attack_mode == 'lesion':
        adv_images = images * (1 - masks) + adv_images * masks

    return adv_images.detach(), perturbations.detach()


def pgd_attack(model, images, masks, epsilon, alpha, num_steps, targeted=False, attack_mode='lesion'):
    """
    Projected Gradient Descent (PGD) attack.

    Args:
        model: CheXzeroWrapper instance
        images: (B, 3, H, W) tensor, clean images
        masks: (B, 3, H, W) tensor, lesion masks
        epsilon: perturbation budget (L_inf)
        alpha: step size
        num_steps: number of iterations
        targeted: if True, maximize probability; if False, minimize probability
        attack_mode: 'lesion' or 'full'

    Returns:
        adv_images: adversarial images
        perturbations: final perturbations
    """
    adv_images = images.clone().detach()

    for step in range(num_steps):
        adv_images.requires_grad = True

        # Forward pass
        outputs = model(adv_images)

        # Loss
        if targeted:
            loss = outputs.mean()
        else:
            loss = -outputs.mean()

        # Backward pass
        model.zero_grad()
        loss.backward()

        # Get gradient
        grad = adv_images.grad.data

        # Apply mask if lesion attack
        if attack_mode == 'lesion':
            grad = grad * masks

        # PGD step
        adv_images = adv_images + alpha * grad.sign()

        # Project back to epsilon ball
        perturbations = torch.clamp(adv_images - images, -epsilon, epsilon)

        # Apply mask to perturbation
        if attack_mode == 'lesion':
            perturbations = perturbations * masks

        adv_images = images + perturbations
        adv_images = adv_images.detach()

    return adv_images, perturbations


def evaluate_attack(model, clean_images, adv_images, masks, attack_mode):
    """
    Evaluate attack success and perturbation metrics.

    Args:
        model: CheXzeroWrapper instance
        clean_images: (B, 3, H, W) clean images
        adv_images: (B, 3, H, W) adversarial images
        masks: (B, 3, H, W) lesion masks
        attack_mode: 'lesion' or 'full'

    Returns:
        metrics: dict of evaluation metrics
    """
    with torch.no_grad():
        # Predictions
        clean_probs = model.predict(clean_images).cpu().numpy()
        adv_probs = model.predict(adv_images).cpu().numpy()

        # Perturbations
        perturbations = (adv_images - clean_images).cpu().numpy()

        # Success: flip prediction from positive (>=0.5) to negative (<0.5)
        clean_pred = (clean_probs >= 0.5).astype(int)
        adv_pred = (adv_probs >= 0.5).astype(int)
        success = (clean_pred != adv_pred).astype(int)

        # Compute metrics
        if attack_mode == 'lesion':
            # Only measure perturbation in lesion region
            masks_np = masks.cpu().numpy()
            masked_perturbations = perturbations * masks_np
            num_pixels = masks_np.sum(axis=(1, 2, 3))
        else:
            # Measure perturbation in full image
            masked_perturbations = perturbations
            num_pixels = np.prod(perturbations.shape[1:])

        # L0 norm: number of perturbed pixels
        l0_norms = (np.abs(masked_perturbations) > 1e-6).sum(axis=(1, 2, 3))

        # L2 norm
        l2_norms = np.sqrt((masked_perturbations ** 2).sum(axis=(1, 2, 3)))

        # L_inf norm
        linf_norms = np.abs(masked_perturbations).max(axis=(1, 2, 3))

        metrics = {
            'clean_probs': clean_probs,
            'adv_probs': adv_probs,
            'success': success,
            'prob_change': adv_probs - clean_probs,
            'l0_norms': l0_norms,
            'l2_norms': l2_norms,
            'linf_norms': linf_norms,
            'num_pixels': num_pixels if isinstance(num_pixels, int) else num_pixels
        }

        return metrics


def summarize_results(metrics_list, attack_name, attack_mode):
    """
    Summarize attack results.

    Args:
        metrics_list: list of metric dicts
        attack_name: name of attack (e.g., 'FGSM', 'PGD')
        attack_mode: 'lesion' or 'full'

    Returns:
        summary: dict of summary statistics
    """
    # Concatenate all metrics
    all_success = np.concatenate([m['success'] for m in metrics_list])
    all_prob_change = np.concatenate([m['prob_change'] for m in metrics_list])
    all_l0 = np.concatenate([m['l0_norms'] for m in metrics_list])
    all_l2 = np.concatenate([m['l2_norms'] for m in metrics_list])
    all_linf = np.concatenate([m['linf_norms'] for m in metrics_list])

    summary = {
        'attack': attack_name,
        'mode': attack_mode,
        'num_samples': len(all_success),
        'success_rate': all_success.mean(),
        'num_success': all_success.sum(),
        'prob_change_mean': all_prob_change.mean(),
        'prob_change_std': all_prob_change.std(),
        'l0_mean': all_l0.mean(),
        'l0_std': all_l0.std(),
        'l2_mean': all_l2.mean(),
        'l2_std': all_l2.std(),
        'linf_mean': all_linf.mean(),
        'linf_std': all_linf.std(),
    }

    return summary


if __name__ == '__main__':
    # Test the framework
    print("Testing RSNA Attack Framework...")

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = CheXzeroWrapper(
        model_path='CheXzero/checkpoints/chexzero_weights/best_64_0.0001_original_35000_0.864.pt',
        device=device
    )
    print("Model loaded!")

    # Load dataset
    dataset = RSNADataset(
        h5_path='dataset/rsna/rsna_200_samples.h5',
        lesion_info_path='dataset/rsna/rsna_200_lesion_info.json'
    )
    print(f"Dataset loaded: {len(dataset)} samples")

    # Test on one sample
    sample = dataset[0]
    image = sample['image'].unsqueeze(0).to(device)
    mask = sample['mask'].unsqueeze(0).to(device)

    print(f"\nTest sample: {sample['patient_id']}")
    print(f"  Image shape: {image.shape}")
    print(f"  Mask shape: {mask.shape}")
    print(f"  Mask coverage: {mask.sum().item() / mask.numel() * 100:.2f}%")

    # Clean prediction
    clean_prob = model.predict(image).item()
    print(f"  Clean prediction: {clean_prob:.4f}")

    # Test FGSM attack (lesion mode)
    print("\nTesting FGSM attack (lesion mode)...")
    adv_image, pert = fgsm_attack(model, image, mask, epsilon=8/255, attack_mode='lesion')
    adv_prob = model.predict(adv_image).item()
    print(f"  Adversarial prediction: {adv_prob:.4f}")
    print(f"  Prob change: {adv_prob - clean_prob:.4f}")
    print(f"  Attack success: {(clean_prob >= 0.5 and adv_prob < 0.5)}")

    # Test FGSM attack (full mode)
    print("\nTesting FGSM attack (full mode)...")
    adv_image_full, pert_full = fgsm_attack(model, image, mask, epsilon=8/255, attack_mode='full')
    adv_prob_full = model.predict(adv_image_full).item()
    print(f"  Adversarial prediction: {adv_prob_full:.4f}")
    print(f"  Prob change: {adv_prob_full - clean_prob:.4f}")
    print(f"  Attack success: {(clean_prob >= 0.5 and adv_prob_full < 0.5)}")

    dataset.close()
    print("\nFramework test complete!")
