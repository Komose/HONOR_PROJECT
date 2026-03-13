"""
Run Attacks and SAVE Adversarial Images
========================================
Modified version that saves adversarial sample images for visualization

Generates:
- CSV with metrics (like before)
- .npy files with actual adversarial images (NEW!)
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm

from rsna_attack_framework import RSNADataset, CheXzeroWrapper
from unified_attack_framework_fixed import (
    fgsm_attack_unified,
    pgd_attack_unified,
    cw_attack_unified,
    deepfool_attack_unified
)
from multi_metric_attack_framework import (
    extract_lung_region_mask,
    generate_equivalent_random_mask
)


def run_attack_with_image_save(
    algorithm_name,
    model,
    patient_id,
    clean_image,
    lesion_mask,
    attack_params,
    device,
    save_dir
):
    """
    Run attack and SAVE adversarial images

    Returns:
        results: dict with metrics
        saved_paths: dict with paths to saved images
    """
    # Prepare masks for all modes
    lung_mask = extract_lung_region_mask(clean_image)

    try:
        random_mask, random_info = generate_equivalent_random_mask(
            lesion_mask=lesion_mask,
            lung_mask=lung_mask,
            image=clean_image,
            max_attempts=1000,  # Increased attempts
            overlap_threshold=0.05  # Max 5% overlap with lesion (strict!)
        )
        print(f"    Random mask: overlap={random_info.get('overlap_ratio', 0):.1%}, method={random_info.get('method', 'unknown')}")
    except Exception as e:
        print(f"    Warning: Could not generate random mask for {patient_id}: {e}")
        # Skip this patient instead of using lesion mask!
        return None, None

    full_mask = torch.ones_like(clean_image)

    # Create output directory
    patient_dir = os.path.join(save_dir, patient_id[:12])
    os.makedirs(patient_dir, exist_ok=True)

    results = {}
    saved_paths = {}

    # Clean image
    with torch.no_grad():
        clean_prob = model(clean_image.unsqueeze(0).to(device)).item()

    # Save clean image
    clean_path = os.path.join(patient_dir, 'clean.npy')
    np.save(clean_path, clean_image.cpu().numpy())

    # Run attacks for all 3 modes
    for mode, mask in [('lesion', lesion_mask), ('random_patch', random_mask), ('full', full_mask)]:
        try:
            # Select attack function
            if algorithm_name == 'fgsm':
                adv_image, success = fgsm_attack_unified(
                    model, clean_image.unsqueeze(0), mask.unsqueeze(0),
                    **attack_params
                )
            elif algorithm_name == 'pgd':
                adv_image, success = pgd_attack_unified(
                    model, clean_image.unsqueeze(0), mask.unsqueeze(0),
                    **attack_params
                )
            elif algorithm_name == 'cw':
                adv_image, success = cw_attack_unified(
                    model, clean_image.unsqueeze(0), mask.unsqueeze(0),
                    **attack_params
                )
            elif algorithm_name == 'deepfool':
                adv_image, success = deepfool_attack_unified(
                    model, clean_image.unsqueeze(0), mask.unsqueeze(0),
                    **attack_params
                )

            # Get adversarial probability
            with torch.no_grad():
                adv_prob = model(adv_image.to(device)).item()

            # Compute metrics
            diff = (adv_image - clean_image.unsqueeze(0)).squeeze(0)
            l2_norm = torch.norm(diff).item()
            linf_norm = torch.max(torch.abs(diff)).item()
            l0_norm = torch.sum(torch.abs(diff) > 1e-6).item()

            # Save adversarial image
            adv_path = os.path.join(patient_dir, f'{algorithm_name}_{mode}.npy')
            np.save(adv_path, adv_image.squeeze(0).cpu().numpy())

            # Save mask
            mask_path = os.path.join(patient_dir, f'mask_{mode}.npy')
            np.save(mask_path, mask.cpu().numpy())

            # Store results
            results[mode] = {
                'clean_prob': clean_prob,
                'adv_prob': adv_prob,
                'success': int(success),
                'l2_norm': l2_norm,
                'linf_norm': linf_norm,
                'l0_norm': l0_norm
            }

            saved_paths[mode] = {
                'adv_image': adv_path,
                'mask': mask_path
            }

        except Exception as e:
            print(f"      Error in {mode}: {e}")
            results[mode] = None
            saved_paths[mode] = None

    return results, saved_paths


def main():
    """Generate adversarial samples and save images"""
    print("=" * 70)
    print("RUN ATTACKS AND SAVE ADVERSARIAL IMAGES")
    print("=" * 70)

    # Configuration
    device = 'cpu'
    n_samples = 10  # Generate 10 samples (user request)

    output_dir = 'results/adversarial_images_saved'
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    print("\\nLoading CheXzero model...")
    model = CheXzeroWrapper(
        model_path='CheXzero/checkpoints/chexzero_weights/best_64_0.0001_original_35000_0.864.pt',
        device=device
    )
    model.eval()
    print("Model loaded!")

    # Load dataset
    print("Loading dataset...")
    dataset = RSNADataset(
        h5_path='dataset/rsna/rsna_200_samples.h5',
        lesion_info_path='dataset/rsna/rsna_200_lesion_info.json'
    )

    # Select patients with medium-sized lesions (can generate random patches)
    # These indices have lesions between 2000-8000 pixels (tested)
    suitable_indices = [3, 5, 6, 8, 10, 16, 17, 19, 20, 22]
    selected_indices = suitable_indices[:min(n_samples, len(suitable_indices))]

    # Attack parameters
    attack_configs = {
        'fgsm': {'epsilon': 8/255, 'use_torchattacks': False},
        'pgd': {'epsilon': 8/255, 'alpha': 2/255, 'num_steps': 40},
        'cw': {'c': 50.0, 'kappa': 0.01, 'steps': 1000, 'lr': 0.05},
        'deepfool': {'steps': 50, 'overshoot': 0.01}
    }

    all_results = []

    print(f"\\nGenerating adversarial samples for {n_samples} patients...")
    print("=" * 70)

    for i, idx in enumerate(selected_indices, 1):
        sample = dataset[idx]
        patient_id = sample['patient_id']
        clean_image = sample['image']
        lesion_mask = sample['mask']

        print(f"\\n[{i}/{n_samples}] Patient: {patient_id[:24]}...")

        for algo_name, algo_params in attack_configs.items():
            print(f"  {algo_name.upper()}...", end=' ', flush=True)

            try:
                results, saved_paths = run_attack_with_image_save(
                    algorithm_name=algo_name,
                    model=model,
                    patient_id=patient_id,
                    clean_image=clean_image,
                    lesion_mask=lesion_mask,
                    attack_params=algo_params,
                    device=device,
                    save_dir=output_dir
                )

                # Add to results
                for mode, metrics in results.items():
                    if metrics:
                        all_results.append({
                            'patient_id': patient_id,
                            'algorithm': algo_name,
                            'mode': mode,
                            **metrics,
                            'image_path': saved_paths[mode]['adv_image'] if saved_paths[mode] else None
                        })

                success_count = sum(1 for m in results.values() if m and m['success'])
                print(f"OK ({success_count}/3 succeeded)")

            except Exception as e:
                print(f"ERROR: {e}")

    dataset.close()

    # Save CSV
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(output_dir, 'results_with_image_paths.csv')
    df.to_csv(csv_path, index=False)

    print("\\n" + "=" * 70)
    print("COMPLETE!")
    print(f"Results CSV: {csv_path}")
    print(f"Adversarial images saved in: {output_dir}/")
    print(f"Total images saved: {len(all_results)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
