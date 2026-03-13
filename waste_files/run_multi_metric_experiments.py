"""
Run Multi-Metric Sensitivity Analysis Experiments
"""

import os, sys, argparse, warnings, torch
from torch.utils.data import DataLoader
import pandas as pd
import time
from datetime import datetime

warnings.filterwarnings('ignore')

from rsna_attack_framework import RSNADataset, CheXzeroWrapper, pgd_attack
from multi_metric_attack_framework import (
    run_experiment_a_fixed_linf,
    run_experiment_b_fixed_l0,
    run_experiment_c_fixed_l2
)

def main():
    parser = argparse.ArgumentParser(description='Run Multi-Metric Experiments')
    parser.add_argument('--experiments', type=str, default='ABC')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--output_dir', type=str, default='results/multi_metric')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    print("\n" + "="*80)
    print("MULTI-METRIC SENSITIVITY ANALYSIS")
    print("="*80)
    print(f"Experiments: {args.experiments}, Batch: {args.batch_size}, Device: {args.device}")
    print("="*80 + "\n")

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading dataset...")
    dataset = RSNADataset(
        h5_path='dataset/rsna/rsna_200_samples.h5',
        lesion_info_path='dataset/rsna/rsna_200_lesion_info.json'
    )
    print(f"Loaded {len(dataset)} samples\n")

    print("Loading CheXzero model...")
    model = CheXzeroWrapper(
        model_path='CheXzero/checkpoints/chexzero_weights/best_64_0.0001_original_35000_0.864.pt',
        device=args.device
    )
    print("Model loaded\n")

    all_results = {}
    experiment_times = {}

    # Experiment A
    if 'A' in args.experiments.upper():
        print("\n" + "#"*80)
        print("# EXPERIMENT A: Fixed Linf")
        print("#"*80 + "\n")
        start = time.time()
        
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        results_a = run_experiment_a_fixed_linf(
            model, dataloader, pgd_attack, [4/255, 8/255], args.device, args.output_dir, "PGD"
        )
        all_results['A'] = results_a
        experiment_times['A'] = time.time() - start
        print(f"\nExp A done in {experiment_times['A']:.1f}s\n")

    # Experiment B
    if 'B' in args.experiments.upper():
        print("\n" + "#"*80)
        print("# EXPERIMENT B: Fixed L0")
        print("#"*80 + "\n")
        start = time.time()
        
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        results_b = run_experiment_b_fixed_l0(
            model, dataloader, pgd_attack, args.device, args.output_dir, "PGD", 8/255
        )
        all_results['B'] = results_b
        experiment_times['B'] = time.time() - start
        print(f"\nExp B done in {experiment_times['B']:.1f}s\n")

    # Experiment C
    if 'C' in args.experiments.upper():
        print("\n" + "#"*80)
        print("# EXPERIMENT C: Fixed L2")
        print("#"*80 + "\n")
        start = time.time()
        
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        results_c = run_experiment_c_fixed_l2(
            model, dataloader, pgd_attack, args.device, args.output_dir, "PGD", 8/255
        )
        all_results['C'] = results_c
        experiment_times['C'] = time.time() - start
        print(f"\nExp C done in {experiment_times['C']:.1f}s\n")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    total = sum(experiment_times.values())
    print(f"\nTotal time: {total:.1f}s ({total/60:.1f} min)")
    for exp, t in experiment_times.items():
        print(f"  Exp {exp}: {t:.1f}s")

    if all_results:
        print("\nSaving results...")
        all_df = pd.concat([df.assign(experiment_name=e) for e, df in all_results.items()], ignore_index=True)
        all_df.to_csv(os.path.join(args.output_dir, 'all_experiments_consolidated.csv'), index=False)
        
        summary = all_df.groupby(['experiment', 'mode']).agg({
            'success': ['mean', 'std'],
            'confidence_drop': ['mean', 'std'],
            'efficiency': ['mean', 'std'],
            'l2_norm': ['mean', 'std']
        }).round(4)
        summary.to_csv(os.path.join(args.output_dir, 'summary_statistics.csv'))
        print(summary)

    dataset.close()
    print("\n" + "="*80)
    print("DONE! Results in: " + args.output_dir)
    print("="*80 + "\n")

if __name__ == "__main__":
    main()

# Add attack function wrapper for compatibility
def pgd_attack_wrapper(model, images, masks, attack_mode, epsilon, num_steps=40, alpha=None):
    """Wrapper to adapt pgd_attack to new framework interface"""
    if alpha is None:
        alpha = epsilon / 4
    return pgd_attack(model, images, masks, epsilon, alpha, num_steps, False, attack_mode)
