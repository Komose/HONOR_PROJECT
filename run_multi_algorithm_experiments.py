"""
Run Multi-Algorithm Multi-Metric Experiments
=============================================

Comprehensive evaluation across:
- Algorithms: FGSM, PGD, C&W, DeepFool
- Metrics: L∞, L0, L2
- Modes: Lesion, Random Patch, Full

Usage:
    python run_multi_algorithm_experiments.py --algorithms all --samples 200

Author: Multi-Algorithm Framework
Date: March 2026
"""

import os
import sys
import argparse
import warnings
import torch
from torch.utils.data import DataLoader
import pandas as pd
import time
from datetime import datetime
from tqdm import tqdm

warnings.filterwarnings('ignore')

from rsna_attack_framework import RSNADataset, CheXzeroWrapper
from unified_attack_framework_fixed import (
    fgsm_attack_unified,
    pgd_attack_unified,
    cw_attack_unified,
    deepfool_attack_unified,
    compute_metrics
)
from multi_metric_attack_framework import (
    extract_lung_region_mask,
    generate_equivalent_random_mask
)


def run_algorithm_experiment(
    algorithm_name: str,
    model,
    dataloader,
    attack_params: dict,
    device: str,
    output_dir: str,
    enable_random_patch: bool = False
):
    """
    Run experiment for a specific attack algorithm.

    Args:
        algorithm_name: 'fgsm', 'pgd', 'cw', or 'deepfool'
        model: CheXzeroWrapper instance
        dataloader: DataLoader
        attack_params: attack-specific parameters
        device: 'cuda' or 'cpu'
        output_dir: directory to save results
        enable_random_patch: whether to include random patch control

    Returns:
        results_df: DataFrame with results
    """
    print(f"\n{'='*80}")
    print(f"ALGORITHM: {algorithm_name.upper()}")
    print(f"{'='*80}")
    print(f"Parameters: {attack_params}")
    print()

    # Select attack function
    if algorithm_name == 'fgsm':
        attack_fn = fgsm_attack_unified
    elif algorithm_name == 'pgd':
        attack_fn = pgd_attack_unified
    elif algorithm_name == 'cw':
        attack_fn = cw_attack_unified
    elif algorithm_name == 'deepfool':
        attack_fn = deepfool_attack_unified
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")

    all_results = []
    random_mask_failures = 0

    # Determine modes to test
    if enable_random_patch and dataloader.batch_size == 1:
        modes = ['lesion', 'random_patch', 'full']
    else:
        modes = ['lesion', 'full']

    for mode in modes:
        print(f"\n--- Mode: {mode.upper()} ---")

        mode_results = []

        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"{algorithm_name}-{mode}")):
            images = batch['image'].to(device)
            lesion_masks = batch['mask'].to(device)
            patient_ids = batch['patient_id']

            # For random patch mode, generate equal-area random masks
            if mode == 'random_patch' and dataloader.batch_size == 1:
                try:
                    lung_mask = extract_lung_region_mask(images[0])
                    lung_mask = lung_mask.to(device)
                    random_mask, random_info = generate_equivalent_random_mask(
                        lesion_mask=lesion_masks[0],
                        lung_mask=lung_mask,
                        image=images[0],  # CRITICAL: Pass original image for tissue validation
                        max_attempts=500
                    )
                    masks = random_mask.unsqueeze(0).to(device)

                    # Log tissue validation results
                    if random_info['mean_intensity'] <= -1.0:
                        print(f"WARNING: Random patch has low intensity ({random_info['mean_intensity']:.3f}) for {patient_ids[0]}")
                except ValueError as e:
                    random_mask_failures += 1
                    print(f"Random mask generation failed for {patient_ids[0]}: {e}")
                    continue
            else:
                masks = lesion_masks

            # Run attack
            try:
                adv_images, perturbations = attack_fn(
                    model=model,
                    images=images,
                    masks=masks,
                    attack_mode=mode,
                    **attack_params
                )
            except Exception as e:
                print(f"WARNING: Attack failed for {patient_ids}: {e}")
                continue

            # Evaluate
            with torch.no_grad():
                clean_probs = model(images).cpu()
                adv_probs = model(adv_images).cpu()

            # Compute metrics
            metrics = compute_metrics(
                clean_images=images,
                adv_images=adv_images,
                clean_probs=clean_probs,
                adv_probs=adv_probs,
                perturbations=perturbations
            )

            # Store per-sample results
            for i in range(len(patient_ids)):
                result = {
                    'algorithm': algorithm_name,
                    'mode': mode,
                    'patient_id': patient_ids[i],
                    'clean_prob': metrics['clean_prob'][i],
                    'adv_prob': metrics['adv_prob'][i],
                    'success': metrics['success'][i],
                    'confidence_drop': metrics['confidence_drop'][i],
                    'efficiency': metrics['efficiency'][i],
                    'l2_norm': metrics['l2_norm'][i],
                    'linf_norm': metrics['linf_norm'][i],
                    'l0_norm': metrics['l0_norm'][i],
                }

                # Add attack-specific parameters
                for key, value in attack_params.items():
                    result[f'param_{key}'] = value

                mode_results.append(result)

        all_results.extend(mode_results)

        # Print mode summary
        if mode_results:
            df_mode = pd.DataFrame(mode_results)
            print(f"\n{mode.upper()} Summary:")
            print(f"  Success Rate: {df_mode['success'].mean():.1%}")
            print(f"  Avg L2: {df_mode['l2_norm'].mean():.3f}")
            print(f"  Avg Efficiency: {df_mode['efficiency'].mean():.5f}")

        # CRITICAL: Incremental save after each mode (防熔断机制)
        if mode_results:
            os.makedirs(output_dir, exist_ok=True)
            checkpoint_file = os.path.join(output_dir, f'{algorithm_name}_checkpoint.csv')

            # Append to checkpoint file
            df_checkpoint = pd.DataFrame(mode_results)
            if os.path.exists(checkpoint_file):
                df_checkpoint.to_csv(checkpoint_file, mode='a', header=False, index=False)
                print(f"[CHECKPOINT] Appended {len(mode_results)} results to {checkpoint_file}")
            else:
                df_checkpoint.to_csv(checkpoint_file, mode='w', header=True, index=False)
                print(f"[CHECKPOINT] Created {checkpoint_file} with {len(mode_results)} results")

    if random_mask_failures > 0:
        print(f"\nRandom mask generation failures: {random_mask_failures}")

    # Save final consolidated results
    results_df = pd.DataFrame(all_results)
    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(
        os.path.join(output_dir, f'{algorithm_name}_results.csv'),
        index=False
    )
    print(f"[FINAL] Saved complete results to {algorithm_name}_results.csv")

    return results_df


def main():
    parser = argparse.ArgumentParser(description='Run Multi-Algorithm Experiments')
    parser.add_argument('--algorithms', type=str, default='all',
                        help='Algorithms to run: fgsm, pgd, cw, deepfool, or "all"')
    parser.add_argument('--samples', type=int, default=200,
                        help='Number of samples to use (default: 200)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size (use 1 for random patch mode)')
    parser.add_argument('--output_dir', type=str, default='results/multi_algorithm',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--enable_random_patch', action='store_true',
                        help='Enable random patch control group (requires batch_size=1)')
    args = parser.parse_args()

    print("\n" + "="*80)
    print("MULTI-ALGORITHM ROBUSTNESS EVALUATION")
    print("="*80)
    print(f"Algorithms: {args.algorithms}")
    print(f"Samples: {args.samples}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {args.device}")
    print(f"Output: {args.output_dir}")
    print("="*80 + "\n")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    print("Loading dataset...")
    dataset = RSNADataset(
        h5_path='dataset/rsna/rsna_200_samples.h5',
        lesion_info_path='dataset/rsna/rsna_200_lesion_info.json'
    )
    print(f"Dataset: {len(dataset)} samples\n")

    # Load model
    print("Loading CheXzero model...")
    model = CheXzeroWrapper(
        model_path='CheXzero/checkpoints/chexzero_weights/best_64_0.0001_original_35000_0.864.pt',
        device=args.device
    )
    print("Model loaded\n")

    # Define algorithms and their parameters
    # FINAL OPTIMIZED PARAMETERS (2026-03-06)
    # C&W: Pilot study empirical value (c=50.0)
    # DeepFool: Hyperparameter search result (steps=50, overshoot=0.01)
    algorithm_configs = {
        'fgsm': {
            'params': {'epsilon': 8/255, 'use_torchattacks': False},
            'batch_size': args.batch_size
        },
        'pgd': {
            'params': {'epsilon': 8/255, 'alpha': 2/255, 'num_steps': 40, 'use_torchattacks': False},
            'batch_size': args.batch_size
        },
        'cw': {
            'params': {'c': 50.0, 'kappa': 0.01, 'steps': 1000, 'lr': 0.05},
            'batch_size': 1  # C&W is memory intensive
        },
        'deepfool': {
            'params': {'steps': 50, 'overshoot': 0.01},
            'batch_size': 1  # DeepFool processes one at a time
        }
    }

    # Select algorithms to run
    if args.algorithms.lower() == 'all':
        algorithms_to_run = list(algorithm_configs.keys())
    else:
        algorithms_to_run = [a.strip().lower() for a in args.algorithms.split(',')]

    # Run experiments
    all_results = {}
    experiment_times = {}

    for algo_name in algorithms_to_run:
        if algo_name not in algorithm_configs:
            print(f"WARNING: Unknown algorithm '{algo_name}', skipping...")
            continue

        config = algorithm_configs[algo_name]

        # Create dataloader with appropriate batch size
        dataloader = DataLoader(
            dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=0
        )

        print(f"\n{'#'*80}")
        print(f"# RUNNING: {algo_name.upper()}")
        print(f"{'#'*80}")

        start_time = time.time()

        try:
            results_df = run_algorithm_experiment(
                algorithm_name=algo_name,
                model=model,
                dataloader=dataloader,
                attack_params=config['params'],
                device=args.device,
                output_dir=args.output_dir,
                enable_random_patch=(args.enable_random_patch and config['batch_size'] == 1)
            )

            all_results[algo_name] = results_df
            experiment_times[algo_name] = time.time() - start_time

            print(f"\n[OK] {algo_name.upper()} completed in {experiment_times[algo_name]:.1f}s")

        except Exception as e:
            print(f"\n[FAILED] {algo_name.upper()} failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Generate summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)

    total_time = sum(experiment_times.values())
    print(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f} min)")

    for algo, t in experiment_times.items():
        print(f"  {algo.upper()}: {t:.1f}s ({t/60:.1f} min)")

    # Consolidated results
    if all_results:
        print("\nSaving consolidated results...")
        all_df = pd.concat(
            [df for df in all_results.values()],
            ignore_index=True
        )
        all_df.to_csv(
            os.path.join(args.output_dir, 'all_algorithms_consolidated.csv'),
            index=False
        )

        # Summary statistics
        summary = all_df.groupby(['algorithm', 'mode']).agg({
            'success': ['mean', 'std', 'count'],
            'confidence_drop': ['mean', 'std'],
            'efficiency': ['mean', 'std'],
            'l2_norm': ['mean', 'std']
        }).round(4)

        summary.to_csv(os.path.join(args.output_dir, 'algorithm_comparison.csv'))

        print("\n" + "="*80)
        print("ALGORITHM COMPARISON")
        print("="*80)
        print(summary)

    # Cleanup
    dataset.close()

    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED!")
    print("="*80)
    print(f"\nResults saved to: {args.output_dir}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
