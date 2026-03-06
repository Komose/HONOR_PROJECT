"""
Hyperparameter Search for C&W and DeepFool Attacks
====================================================

Find optimal parameters before running full experiments.

Author: Multi-Algorithm Framework
Date: March 2026
"""

import os
import torch
from torch.utils.data import DataLoader, Subset
import pandas as pd
import numpy as np
from tqdm import tqdm
import itertools

from rsna_attack_framework import RSNADataset, CheXzeroWrapper
from unified_attack_framework_fixed import (
    cw_attack_unified,
    deepfool_attack_unified,
    compute_metrics
)


def hyperparameter_search_cw(
    model,
    dataloader,
    device: str,
    param_grid: dict,
    output_dir: str = 'results/hyperparam_search'
):
    """
    Grid search for C&W hyperparameters.

    Args:
        model: CheXzeroWrapper
        dataloader: Small validation set (10-20 samples)
        device: 'cuda' or 'cpu'
        param_grid: Dictionary of parameter lists
        output_dir: Where to save results
    """
    print("=" * 80)
    print("C&W HYPERPARAMETER SEARCH")
    print("=" * 80)
    print(f"Parameter grid:")
    for key, values in param_grid.items():
        print(f"  {key}: {values}")
    print()

    # Generate all parameter combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))

    print(f"Total combinations: {len(combinations)}")
    print()

    results = []

    for combo in tqdm(combinations, desc="C&W Grid Search"):
        params = dict(zip(keys, combo))

        print(f"\nTesting: {params}")

        all_success = []
        all_l2 = []
        all_conf_drop = []

        try:
            for batch in dataloader:
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)

                # Run attack with current parameters
                adv_images, perturbations = cw_attack_unified(
                    model=model,
                    images=images,
                    masks=masks,
                    attack_mode='lesion',  # Test on lesion mode
                    **params
                )

                # Evaluate
                with torch.no_grad():
                    clean_probs = model(images).cpu()
                    adv_probs = model(adv_images).cpu()

                metrics = compute_metrics(
                    clean_images=images,
                    adv_images=adv_images,
                    clean_probs=clean_probs,
                    adv_probs=adv_probs,
                    perturbations=perturbations
                )

                all_success.extend(metrics['success'])
                all_l2.extend(metrics['l2_norm'])
                all_conf_drop.extend(metrics['confidence_drop'])

            # Aggregate results
            result = {
                **params,
                'asr': np.mean(all_success),
                'avg_l2': np.mean(all_l2),
                'avg_conf_drop': np.mean(all_conf_drop),
                'efficiency': np.mean(all_conf_drop) / (np.mean(all_l2) + 1e-8)
            }

            print(f"  ASR: {result['asr']:.1%}, L2: {result['avg_l2']:.2f}, Eff: {result['efficiency']:.5f}")
            results.append(result)

        except Exception as e:
            print(f"  FAILED: {e}")
            continue

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'cw_hyperparam_search.csv'), index=False)

    # Find best parameters
    print("\n" + "=" * 80)
    print("C&W HYPERPARAMETER SEARCH RESULTS")
    print("=" * 80)

    # Sort by ASR (primary) and efficiency (secondary)
    df_sorted = df.sort_values(by=['asr', 'efficiency'], ascending=[False, False])

    print("\nTop 5 configurations by ASR:")
    print(df_sorted.head(5).to_string())

    print("\n" + "=" * 80)
    print("RECOMMENDED PARAMETERS:")
    best = df_sorted.iloc[0]
    for key in keys:
        print(f"  {key}: {best[key]}")
    print(f"  Expected ASR: {best['asr']:.1%}")
    print(f"  Expected L2: {best['avg_l2']:.2f}")
    print("=" * 80)

    return best.to_dict()


def hyperparameter_search_deepfool(
    model,
    dataloader,
    device: str,
    param_grid: dict,
    output_dir: str = 'results/hyperparam_search'
):
    """
    Grid search for DeepFool hyperparameters.
    """
    print("\n" + "=" * 80)
    print("DEEPFOOL HYPERPARAMETER SEARCH")
    print("=" * 80)
    print(f"Parameter grid:")
    for key, values in param_grid.items():
        print(f"  {key}: {values}")
    print()

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))

    print(f"Total combinations: {len(combinations)}")
    print()

    results = []

    for combo in tqdm(combinations, desc="DeepFool Grid Search"):
        params = dict(zip(keys, combo))

        print(f"\nTesting: {params}")

        all_success = []
        all_l2 = []
        all_conf_drop = []

        try:
            for batch in dataloader:
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)

                adv_images, perturbations = deepfool_attack_unified(
                    model=model,
                    images=images,
                    masks=masks,
                    attack_mode='lesion',
                    **params
                )

                with torch.no_grad():
                    clean_probs = model(images).cpu()
                    adv_probs = model(adv_images).cpu()

                metrics = compute_metrics(
                    clean_images=images,
                    adv_images=adv_images,
                    clean_probs=clean_probs,
                    adv_probs=adv_probs,
                    perturbations=perturbations
                )

                all_success.extend(metrics['success'])
                all_l2.extend(metrics['l2_norm'])
                all_conf_drop.extend(metrics['confidence_drop'])

            result = {
                **params,
                'asr': np.mean(all_success),
                'avg_l2': np.mean(all_l2),
                'avg_conf_drop': np.mean(all_conf_drop),
                'efficiency': np.mean(all_conf_drop) / (np.mean(all_l2) + 1e-8)
            }

            print(f"  ASR: {result['asr']:.1%}, L2: {result['avg_l2']:.2f}, Eff: {result['efficiency']:.5f}")
            results.append(result)

        except Exception as e:
            print(f"  FAILED: {e}")
            continue

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'deepfool_hyperparam_search.csv'), index=False)

    # Find best parameters
    print("\n" + "=" * 80)
    print("DEEPFOOL HYPERPARAMETER SEARCH RESULTS")
    print("=" * 80)

    df_sorted = df.sort_values(by=['asr', 'efficiency'], ascending=[False, False])

    print("\nTop 5 configurations by ASR:")
    print(df_sorted.head(5).to_string())

    print("\n" + "=" * 80)
    print("RECOMMENDED PARAMETERS:")
    best = df_sorted.iloc[0]
    for key in keys:
        print(f"  {key}: {best[key]}")
    print(f"  Expected ASR: {best['asr']:.1%}")
    print(f"  Expected L2: {best['avg_l2']:.2f}")
    print("=" * 80)

    return best.to_dict()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Hyperparameter search for C&W and DeepFool')
    parser.add_argument('--algorithms', type=str, default='cw,deepfool',
                        help='Algorithms to tune: cw, deepfool, or both')
    parser.add_argument('--n_samples', type=int, default=20,
                        help='Number of samples for validation (default: 20)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_dir', type=str, default='results/hyperparam_search')
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("HYPERPARAMETER SEARCH FOR ADVERSARIAL ATTACKS")
    print("=" * 80)
    print(f"Algorithms: {args.algorithms}")
    print(f"Validation samples: {args.n_samples}")
    print(f"Device: {args.device}")
    print("=" * 80 + "\n")

    # Load dataset (subset)
    print("Loading dataset...")
    dataset = RSNADataset(
        h5_path='dataset/rsna/rsna_200_samples.h5',
        lesion_info_path='dataset/rsna/rsna_200_lesion_info.json'
    )

    # Random subset for validation
    indices = np.random.choice(len(dataset), args.n_samples, replace=False)
    subset = Subset(dataset, indices)
    dataloader = DataLoader(subset, batch_size=1, shuffle=False)

    print(f"Validation set: {len(subset)} samples\n")

    # Load model
    print("Loading CheXzero model...")
    model = CheXzeroWrapper(
        model_path='CheXzero/checkpoints/chexzero_weights/best_64_0.0001_original_35000_0.864.pt',
        device=args.device
    )
    print("Model loaded\n")

    algorithms = [a.strip().lower() for a in args.algorithms.split(',')]
    best_params = {}

    # C&W hyperparameter search
    if 'cw' in algorithms:
        cw_param_grid = {
            'c': [0.1, 1.0, 10.0],              # Cost parameter
            'kappa': [0, 0.01, 0.05, 0.1],      # FIXED: Confidence margin (适合CheXzero的logit尺度)
            'steps': [500, 1000],               # Optimization steps
            'lr': [0.001, 0.01]                 # Learning rate
        }

        best_params['cw'] = hyperparameter_search_cw(
            model=model,
            dataloader=dataloader,
            device=args.device,
            param_grid=cw_param_grid,
            output_dir=args.output_dir
        )

    # DeepFool hyperparameter search
    if 'deepfool' in algorithms:
        deepfool_param_grid = {
            'steps': [50, 100, 150, 200],       # Iterations
            'overshoot': [0.01, 0.02, 0.05]     # Overshoot parameter
        }

        best_params['deepfool'] = hyperparameter_search_deepfool(
            model=model,
            dataloader=dataloader,
            device=args.device,
            param_grid=deepfool_param_grid,
            output_dir=args.output_dir
        )

    # Save best parameters
    import json
    with open(os.path.join(args.output_dir, 'best_params.json'), 'w') as f:
        json.dump(best_params, f, indent=2)

    print("\n" + "=" * 80)
    print("HYPERPARAMETER SEARCH COMPLETED!")
    print("=" * 80)
    print(f"Results saved to: {args.output_dir}")
    print("Use the recommended parameters for full experiments.")
    print("=" * 80 + "\n")

    dataset.close()


if __name__ == "__main__":
    main()
