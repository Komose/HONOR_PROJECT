"""
Run Adversarial Attacks on RSNA Dataset
========================================

This script runs lesion-targeted vs full-image adversarial attacks
and compares their effectiveness.

Attacks implemented:
- FGSM (Fast Gradient Sign Method)
- PGD (Projected Gradient Descent)
"""

import os
import sys
import argparse
import numpy as np
import torch
from torch.utils import data
import pandas as pd
import json
from tqdm import tqdm
import time

from rsna_attack_framework import (
    RSNADataset, CheXzeroWrapper,
    fgsm_attack, pgd_attack,
    evaluate_attack, summarize_results
)


def run_attack_experiment(
    model,
    dataloader,
    attack_fn,
    attack_params,
    attack_name,
    attack_mode,
    device,
    output_dir
):
    """
    Run attack on all samples and collect results.

    Args:
        model: CheXzeroWrapper instance
        dataloader: PyTorch DataLoader
        attack_fn: attack function (fgsm_attack or pgd_attack)
        attack_params: dict of attack parameters
        attack_name: name of attack
        attack_mode: 'lesion' or 'full'
        device: torch device
        output_dir: directory to save results

    Returns:
        results_df: DataFrame with per-sample results
        summary: dict of summary statistics
    """
    print(f"\n{'='*70}")
    print(f"Running {attack_name} Attack - {attack_mode.upper()} MODE")
    print(f"{'='*70}")
    print(f"Parameters: {attack_params}")
    print()

    all_results = []
    metrics_list = []

    model.eval()

    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"{attack_name}-{attack_mode}")):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        patient_ids = batch['patient_id']

        # Perform attack
        adv_images, perturbations = attack_fn(
            model=model,
            images=images,
            masks=masks,
            attack_mode=attack_mode,
            **attack_params
        )

        # Evaluate
        metrics = evaluate_attack(model, images, adv_images, masks, attack_mode)
        metrics_list.append(metrics)

        # Store per-sample results
        for i in range(len(patient_ids)):
            all_results.append({
                'patient_id': patient_ids[i],
                'attack': attack_name,
                'mode': attack_mode,
                'clean_prob': metrics['clean_probs'][i],
                'adv_prob': metrics['adv_probs'][i],
                'prob_change': metrics['prob_change'][i],
                'success': metrics['success'][i],
                'l0_norm': metrics['l0_norms'][i],
                'l2_norm': metrics['l2_norms'][i],
                'linf_norm': metrics['linf_norms'][i],
            })

    # Create DataFrame
    results_df = pd.DataFrame(all_results)

    # Compute summary
    summary = summarize_results(metrics_list, attack_name, attack_mode)

    # Print summary
    print(f"\n{'-'*70}")
    print(f"RESULTS SUMMARY: {attack_name} - {attack_mode.upper()}")
    print(f"{'-'*70}")
    print(f"  Samples: {summary['num_samples']}")
    print(f"  Success Rate: {summary['success_rate']*100:.2f}% ({summary['num_success']}/{summary['num_samples']})")
    print(f"  Prob Change: {summary['prob_change_mean']:.4f} ± {summary['prob_change_std']:.4f}")
    print(f"  L2 Norm: {summary['l2_mean']:.2f} ± {summary['l2_std']:.2f}")
    print(f"  L∞ Norm: {summary['linf_mean']:.4f} ± {summary['linf_std']:.4f}")
    print(f"{'-'*70}\n")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results_csv = os.path.join(output_dir, f'{attack_name.lower()}_{attack_mode}_results.csv')
    results_df.to_csv(results_csv, index=False)
    print(f"Results saved to: {results_csv}")

    return results_df, summary


def main():
    parser = argparse.ArgumentParser(description='Run RSNA adversarial attacks')
    parser.add_argument('--model_path', type=str,
                        default='CheXzero/checkpoints/chexzero_weights/best_64_0.0001_original_35000_0.864.pt',
                        help='Path to CheXzero model')
    parser.add_argument('--data_h5', type=str,
                        default='dataset/rsna/rsna_200_samples.h5',
                        help='Path to HDF5 file')
    parser.add_argument('--lesion_info', type=str,
                        default='dataset/rsna/rsna_200_lesion_info.json',
                        help='Path to lesion info JSON')
    parser.add_argument('--output_dir', type=str,
                        default='results/attacks',
                        help='Output directory')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to attack (default: all)')

    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    print("\nLoading CheXzero model...")
    model = CheXzeroWrapper(model_path=args.model_path, device=device)
    print("Model loaded!")

    # Load dataset
    print(f"\nLoading dataset...")
    dataset = RSNADataset(h5_path=args.data_h5, lesion_info_path=args.lesion_info)

    if args.num_samples is not None:
        dataset = torch.utils.data.Subset(dataset, range(args.num_samples))

    dataloader = data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    print(f"Dataset loaded: {len(dataset)} samples, batch size={args.batch_size}")

    # Define attacks
    attacks = []

    # FGSM attacks
    for eps in [4/255, 8/255, 16/255]:
        attacks.append({
            'name': f'FGSM_eps{int(eps*255)}',
            'function': fgsm_attack,
            'params': {'epsilon': eps, 'targeted': False}
        })

    # PGD attacks
    for steps in [10, 20, 40]:
        attacks.append({
            'name': f'PGD_steps{steps}',
            'function': pgd_attack,
            'params': {'epsilon': 8/255, 'alpha': 2/255, 'num_steps': steps, 'targeted': False}
        })

    # Run experiments
    all_results = []
    all_summaries = []

    start_time = time.time()

    for attack_config in attacks:
        for mode in ['lesion', 'full']:
            results_df, summary = run_attack_experiment(
                model=model,
                dataloader=dataloader,
                attack_fn=attack_config['function'],
                attack_params=attack_config['params'],
                attack_name=attack_config['name'],
                attack_mode=mode,
                device=device,
                output_dir=args.output_dir
            )

            all_results.append(results_df)
            all_summaries.append(summary)

    # Combine all results
    combined_results = pd.concat(all_results, ignore_index=True)
    combined_results_path = os.path.join(args.output_dir, 'all_results.csv')
    combined_results.to_csv(combined_results_path, index=False)

    # Save summaries
    summaries_df = pd.DataFrame(all_summaries)
    summaries_path = os.path.join(args.output_dir, 'attack_summaries.csv')
    summaries_df.to_csv(summaries_path, index=False)

    # Print comparison
    print(f"\n{'='*70}")
    print("ATTACK COMPARISON: LESION vs FULL")
    print(f"{'='*70}\n")

    print(summaries_df.to_string(index=False))

    elapsed_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Total time: {elapsed_time/60:.2f} minutes")
    print(f"Results saved to: {args.output_dir}/")
    print(f"{'='*70}")

    # Close dataset
    if hasattr(dataset, 'dataset'):  # Subset
        dataset.dataset.close()
    else:
        dataset.close()


if __name__ == '__main__':
    main()
