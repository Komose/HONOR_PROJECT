"""
L2 Robustness Evaluation for C&W and DeepFool Attacks
======================================================

Complete evaluation of L2-constrained adversarial attacks with:
- Task 0: C&W convergence pilot study (3 patients, 4 step values)
- Task 1: C&W parameter sweep (c × kappa grid)
- Task 2: DeepFool overshoot sweep
- Task 3: Strict control variables (L0 alignment, survivor bias elimination)
- Task 4: Real-time checkpoint with resume capability
- Task 5: Visualization of 10 representative cases

Author: HONER Project
Date: 2026-03-13
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image
import os
import sys
import random
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Force unbuffered output for real-time logging
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

# ============================================================================
# Import Attack Functions and Utilities
# ============================================================================

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unified_attack_framework_fixed import cw_attack_unified, deepfool_attack_unified
from multi_metric_attack_framework import generate_equivalent_random_mask

# ============================================================================
# Configuration
# ============================================================================

OUTPUT_CSV = 'results/l2_robustness_evaluation.csv'
PILOT_LOG = 'results/cw_convergence_pilot_log.txt'
VIS_DIR = 'results/l2_visualizations'

# Task 1: C&W parameters (FIXED: wide range from gentle to reference)
CW_C_VALUES = [0.01, 0.1, 1.0, 5.0, 50.0]
CW_KAPPA_VALUES = [0.0, 1.0, 10.0]
CW_LR = 0.05

# Task 2: DeepFool parameters
DEEPFOOL_OVERSHOOT_VALUES = [0.01, 0.02, 0.05, 0.1]
DEEPFOOL_STEPS = 150

# Dataset
NUM_PATIENTS = 200
BATCH_SIZE = 1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Modes: MUST include all 3
MODES = ['lesion', 'random_patch', 'full']

# Visualization
NUM_VIS_PATIENTS = 10
VIS_PARAMS = {'cw': {'c': 50.0, 'kappa': 0.0}, 'deepfool': {'overshoot': 0.05}}

# ============================================================================
# Load Model and Dataset (Reuse from previous experiments)
# ============================================================================

def load_chexzero_model(device):
    """Load CheXzero model (same as previous experiments - reuse rsna_attack_framework)."""
    print("Loading CheXzero model...")
    from rsna_attack_framework import CheXzeroWrapper

    model_path = "CheXzero/checkpoints/chexzero_weights/best_64_0.0001_original_35000_0.864.pt"
    model = CheXzeroWrapper(model_path, device=device)
    model.eval()

    return model


def load_rsna_dataset(num_samples=200):
    """Load RSNA Pneumonia dataset (same 200 patients as L-inf experiments - reuse rsna_attack_framework)."""
    print(f"Loading RSNA dataset ({num_samples} patients)...")
    from rsna_attack_framework import RSNADataset

    # Use the same HDF5 file as previous experiments
    h5_path = 'dataset/rsna/rsna_200_samples.h5'
    lesion_info_path = 'dataset/rsna/rsna_200_lesion_info.json'

    dataset_full = RSNADataset(h5_path, lesion_info_path)

    # Extract first num_samples patients
    dataset = []
    for idx in range(min(num_samples, len(dataset_full))):
        sample = dataset_full[idx]
        dataset.append({
            'patient_id': sample['patient_id'],
            'image': sample['image'],
            'label': torch.tensor([1], dtype=torch.long),  # Pneumonia (from dataset_full)
            'mask': sample['mask']  # Lesion mask from dataset
        })

    print(f"Loaded {len(dataset)} patients")
    return dataset


# ============================================================================
# Checkpoint and Resume Logic
# ============================================================================

def load_completed_patients(csv_path):
    """
    Load already completed patients from CSV.
    A patient is considered complete if ALL parameter combinations have been run.
    """
    if not os.path.exists(csv_path):
        return set()

    df = pd.read_csv(csv_path)

    # Calculate expected number of runs per patient
    # C&W: 4*3=12 combinations × 3 modes = 36 runs
    # DeepFool: 4 combinations × 3 modes = 12 runs
    # Total: 48 runs per patient
    expected_runs_per_patient = (len(CW_C_VALUES) * len(CW_KAPPA_VALUES) + len(DEEPFOOL_OVERSHOOT_VALUES)) * len(MODES)

    # Count runs per patient
    patient_counts = df.groupby('patient_id').size()

    # Only consider patients with complete runs
    completed = set(patient_counts[patient_counts >= expected_runs_per_patient].index)

    print(f"[RESUME] Found {len(completed)} completed patients in checkpoint")
    return completed


def save_result_to_csv(result_dict, csv_path):
    """Append single result to CSV with immediate flush."""
    df = pd.DataFrame([result_dict])

    # Append mode: create header if file doesn't exist
    if not os.path.exists(csv_path):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(csv_path, index=False, mode='w')
    else:
        df.to_csv(csv_path, index=False, mode='a', header=False)


# ============================================================================
# Mask Generation (L0 Alignment)
# ============================================================================

def generate_masks_for_patient(image, lesion_mask, mode, device):
    """
    Generate masks for different attack modes.

    Args:
        image: (C, H, W) tensor
        lesion_mask: (H, W) or (C, H, W) tensor with lesion mask
        mode: 'lesion', 'random_patch', or 'full'
        device: torch device

    Returns:
        mask: torch.Tensor (C, H, W)
        error: str or None (if random_patch generation fails)
    """
    C, H, W = image.shape

    if mode == 'full':
        return torch.ones_like(image), None

    elif mode == 'lesion':
        # Use provided lesion mask (expand to 3 channels if needed)
        if lesion_mask.dim() == 2:
            mask = lesion_mask.unsqueeze(0).repeat(C, 1, 1)
        else:
            mask = lesion_mask
        return mask.to(device), None

    elif mode == 'random_patch':
        # Generate equivalent random patch (strict L0 alignment)
        try:
            # Extract lung region mask (for constraint checking)
            from multi_metric_attack_framework import extract_lung_region_mask
            lung_mask = extract_lung_region_mask(image)

            # Generate random mask with L0 alignment
            # Note: generate_equivalent_random_mask already returns (3, H, W) tensor
            random_mask, random_info = generate_equivalent_random_mask(
                lesion_mask=lesion_mask.cpu(),
                lung_mask=lung_mask.cpu(),
                image=image.cpu(),
                max_attempts=500
            )

            return random_mask.to(device), None

        except Exception as e:
            # Random patch generation failed (large lesion, geometric lock)
            return None, str(e)

    else:
        raise ValueError(f"Unknown mode: {mode}")


# ============================================================================
# Task 0: C&W Convergence Pilot Study
# ============================================================================

def pilot_study_cw_convergence(model, dataset, device):
    """
    Test C&W convergence with different step counts.

    Returns:
        optimal_steps: int
    """
    print("\n" + "="*90)
    print("TASK 0: C&W CONVERGENCE PILOT STUDY")
    print("="*90)

    # Randomly select 3 patients that can generate random_patch
    random.seed(42)
    pilot_patients = []

    for sample in random.sample(dataset, min(20, len(dataset))):
        # Test if random_patch can be generated
        image = sample['image'].to(device)
        lesion_mask = sample['mask'].to(device)

        _, error = generate_masks_for_patient(image, lesion_mask, 'random_patch', device)

        if error is None:
            pilot_patients.append(sample)
            if len(pilot_patients) == 3:
                break

    if len(pilot_patients) < 3:
        print(f"[WARNING] Only found {len(pilot_patients)} valid patients, continuing...")

    print(f"\nTesting {len(pilot_patients)} patients with steps = [100, 250, 500, 1000]")
    print(f"Parameters: c=50.0, kappa=0.0, lr=0.05\n")

    steps_list = [100, 250, 500, 1000]
    results = {steps: [] for steps in steps_list}

    # Test each step count
    for steps in steps_list:
        print(f"Testing steps={steps}...")

        for sample in pilot_patients:
            image = sample['image'].unsqueeze(0).to(device)
            lesion_mask = sample['mask'].to(device)

            # Test on lesion mode
            mask, _ = generate_masks_for_patient(sample['image'].to(device), lesion_mask, 'lesion', device)
            mask = mask.unsqueeze(0)

            # Run C&W attack
            try:
                adv_image, perturbation = cw_attack_unified(
                    model=model,
                    images=image,
                    masks=mask,
                    c=50.0,
                    kappa=0.0,
                    steps=steps,
                    lr=0.05,
                    attack_mode='lesion'
                )

                # Calculate L2 norm
                l2_norm = torch.norm(perturbation.flatten()).item()

                results[steps].append(l2_norm)

            except Exception as e:
                print(f"  [ERROR] Patient {sample['patient_id']}: {e}")

    # Analyze convergence
    print("\n" + "-"*90)
    print("CONVERGENCE ANALYSIS")
    print("-"*90)

    log_lines = []
    log_lines.append("="*90)
    log_lines.append("C&W CONVERGENCE PILOT STUDY")
    log_lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_lines.append("="*90)
    log_lines.append(f"\nTested {len(pilot_patients)} patients")
    log_lines.append(f"Parameters: c=50.0, kappa=0.0, lr=0.05\n")

    avg_l2 = {}
    for steps in steps_list:
        if len(results[steps]) > 0:
            avg = np.mean(results[steps])
            std = np.std(results[steps])
            avg_l2[steps] = avg

            line = f"steps={steps:4d}: L2={avg:8.3f} ± {std:6.3f}"
            print(line)
            log_lines.append(line)

    # Decision logic: find smallest steps with <2% relative error vs 1000
    reference_l2 = avg_l2.get(1000, float('inf'))
    optimal_steps = 1000  # Default fallback

    print("\nConvergence Decision:")
    log_lines.append("\nConvergence Decision:")

    for steps in [100, 250, 500]:
        if steps in avg_l2:
            rel_error = abs(avg_l2[steps] - reference_l2) / reference_l2 * 100
            line = f"  steps={steps}: relative error = {rel_error:.2f}%"
            print(line)
            log_lines.append(line)

            if rel_error < 2.0:
                optimal_steps = steps
                break

    decision = f"\n[DECISION] Using steps={optimal_steps} for C&W main experiments"
    print(decision)
    log_lines.append(decision)

    if optimal_steps < 1000:
        savings = (1 - optimal_steps / 1000) * 100
        note = f"  → Estimated time savings: {savings:.0f}%"
        print(note)
        log_lines.append(note)

    # Save log
    os.makedirs(os.path.dirname(PILOT_LOG), exist_ok=True)
    with open(PILOT_LOG, 'w') as f:
        f.write('\n'.join(log_lines))

    print(f"\n[SAVED] Pilot study log: {PILOT_LOG}")

    return optimal_steps


# ============================================================================
# Attack Execution
# ============================================================================

def run_attack(model, image, mask, algorithm, params, attack_mode, device):
    """
    Run single attack and return metrics.

    Returns:
        dict with keys: success, clean_prob, adv_prob, confidence_drop, l2_norm, linf_norm, l0_norm
    """
    try:
        # Get clean prediction (CheXzeroWrapper returns single probability)
        with torch.no_grad():
            clean_prob = model(image)[0].item()

        # Run attack
        if algorithm == 'cw':
            adv_image, perturbation = cw_attack_unified(
                model=model,
                images=image,
                masks=mask,
                c=params['c'],
                kappa=params['kappa'],
                steps=params['steps'],
                lr=params['lr'],
                attack_mode=attack_mode
            )

        elif algorithm == 'deepfool':
            adv_image, perturbation = deepfool_attack_unified(
                model=model,
                images=image,
                masks=mask,
                steps=params['steps'],
                overshoot=params['overshoot'],
                attack_mode=attack_mode
            )

        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        # Get adversarial prediction
        with torch.no_grad():
            adv_prob = model(adv_image)[0].item()

        # Calculate metrics
        l2_norm = torch.norm(perturbation.flatten()).item()
        linf_norm = torch.max(torch.abs(perturbation)).item()
        l0_norm = torch.count_nonzero(perturbation).item()

        success = 1.0 if adv_prob < 0.5 else 0.0
        confidence_drop = clean_prob - adv_prob

        return {
            'success': success,
            'clean_prob': clean_prob,
            'adv_prob': adv_prob,
            'confidence_drop': confidence_drop,
            'l2_norm': l2_norm,
            'linf_norm': linf_norm,
            'l0_norm': l0_norm,
            'adv_image': adv_image  # For visualization
        }

    except Exception as e:
        print(f"    [ERROR] Attack failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# Visualization
# ============================================================================

def visualize_attack_comparison(clean_img, adv_img, patient_id, algorithm, params, mode, save_dir):
    """Generate before/after/perturbation comparison plot."""

    # Convert to numpy
    clean_np = clean_img.squeeze(0).detach().cpu().numpy()[0]  # First channel
    adv_np = adv_img.squeeze(0).detach().cpu().numpy()[0]

    # Perturbation (normalized for visibility)
    perturbation = adv_np - clean_np
    pert_vis = (perturbation - perturbation.min()) / (perturbation.max() - perturbation.min() + 1e-8)

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(clean_np, cmap='gray')
    axes[0].set_title('Clean Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(adv_np, cmap='gray')
    axes[1].set_title('Adversarial Image', fontsize=14, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(pert_vis, cmap='hot')
    axes[2].set_title('Perturbation (Normalized)', fontsize=14, fontweight='bold')
    axes[2].axis('off')

    # Title
    if algorithm == 'cw':
        param_str = f"c={params['c']}, κ={params['kappa']}"
    else:
        param_str = f"overshoot={params['overshoot']}"

    fig.suptitle(f'{algorithm.upper()} Attack: {mode.upper()} mode\n{param_str}',
                 fontsize=16, fontweight='bold')

    plt.tight_layout()

    # Save
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{patient_id}_{algorithm}_{mode}_{param_str.replace('=', '').replace(',', '_').replace(' ', '')}.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return save_path


# ============================================================================
# Main Experiment Loop
# ============================================================================

def main():
    print("="*90)
    print("L2 ROBUSTNESS EVALUATION - C&W AND DEEPFOOL")
    print("="*90)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {DEVICE}")
    print(f"Output CSV: {OUTPUT_CSV}")
    print("="*90)

    # ========================================================================
    # 1. Load Model and Dataset
    # ========================================================================

    model = load_chexzero_model(DEVICE)
    dataset = load_rsna_dataset(NUM_PATIENTS)

    print(f"\nTotal patients: {len(dataset)}")
    print(f"Modes: {MODES}")
    print(f"C&W combinations: {len(CW_C_VALUES)} × {len(CW_KAPPA_VALUES)} = {len(CW_C_VALUES) * len(CW_KAPPA_VALUES)}")
    print(f"DeepFool combinations: {len(DEEPFOOL_OVERSHOOT_VALUES)}")

    total_experiments = len(dataset) * ((len(CW_C_VALUES) * len(CW_KAPPA_VALUES)) + len(DEEPFOOL_OVERSHOOT_VALUES)) * len(MODES)
    print(f"\nTotal experiments: {total_experiments}")

    # ========================================================================
    # 2. Load Checkpoint (Resume Logic)
    # ========================================================================

    completed_patients = load_completed_patients(OUTPUT_CSV)
    remaining_patients = [s for s in dataset if s['patient_id'] not in completed_patients]

    print(f"\n[CHECKPOINT] Completed: {len(completed_patients)}, Remaining: {len(remaining_patients)}")

    # ========================================================================
    # 3. Select Visualization Patients (Random)
    # ========================================================================

    random.seed(42)
    vis_patient_ids = set(random.sample([s['patient_id'] for s in remaining_patients],
                                       min(NUM_VIS_PATIENTS, len(remaining_patients))))

    print(f"[VISUALIZATION] Selected {len(vis_patient_ids)} patients for visualization")

    # ========================================================================
    # 4. TASK 0: C&W Convergence Pilot Study
    # ========================================================================

    optimal_cw_steps = pilot_study_cw_convergence(model, dataset, DEVICE)

    # ========================================================================
    # 5. Main Experiment Loop
    # ========================================================================

    print("\n" + "="*90)
    print("MAIN EXPERIMENTS BEGIN")
    print("="*90)

    skipped_patients = []

    for sample_idx, sample in enumerate(remaining_patients):
        patient_id = sample['patient_id']
        image = sample['image'].unsqueeze(0).to(DEVICE)
        lesion_mask = sample['mask'].to(DEVICE)  # Use lesion mask from dataset

        print(f"\n[{sample_idx+1}/{len(remaining_patients)}] Patient: {patient_id}")

        # ====================================================================
        # Pre-generate masks and check random_patch feasibility
        # ====================================================================

        masks = {}
        random_patch_failed = False

        for mode in MODES:
            mask, error = generate_masks_for_patient(sample['image'].to(DEVICE), lesion_mask, mode, DEVICE)

            if error is not None:
                print(f"  [SKIP] Random patch generation failed: {error}")
                print(f"  → Skipping ALL experiments for this patient (survivor bias elimination)")
                skipped_patients.append(patient_id)
                random_patch_failed = True
                break

            masks[mode] = mask.unsqueeze(0)  # Add batch dimension

        if random_patch_failed:
            continue

        # Verify L0 alignment
        l0_lesion = torch.count_nonzero(masks['lesion']).item()
        l0_random = torch.count_nonzero(masks['random_patch']).item()
        l0_error = abs(l0_lesion - l0_random) / l0_lesion * 100

        print(f"  L0 alignment: Lesion={l0_lesion}, Random={l0_random}, Error={l0_error:.2f}%")

        if l0_error > 0.5:
            print(f"  [WARNING] L0 alignment error > 0.5%, skipping patient")
            skipped_patients.append(patient_id)
            continue

        # ====================================================================
        # TASK 1: C&W Parameter Sweep
        # ====================================================================

        print(f"\n  C&W Experiments:")

        for c in CW_C_VALUES:
            for kappa in CW_KAPPA_VALUES:
                for mode in MODES:
                    params = {
                        'c': c,
                        'kappa': kappa,
                        'steps': optimal_cw_steps,
                        'lr': CW_LR
                    }

                    print(f"    c={c:5.1f}, κ={kappa:4.1f}, mode={mode:12s}...", end=' ')

                    result = run_attack(
                        model=model,
                        image=image,
                        mask=masks[mode],
                        algorithm='cw',
                        params=params,
                        attack_mode=mode,
                        device=DEVICE
                    )

                    if result is None:
                        print("FAILED")
                        continue

                    print(f"L2={result['l2_norm']:7.2f}, ASR={result['success']:.0f}")

                    # Save to CSV
                    csv_row = {
                        'patient_id': patient_id,
                        'algorithm': 'cw',
                        'mode': mode,
                        'c': c,
                        'kappa': kappa,
                        'overshoot': np.nan,
                        'steps': optimal_cw_steps,
                        'lr': CW_LR,
                        'success': result['success'],
                        'clean_prob': result['clean_prob'],
                        'adv_prob': result['adv_prob'],
                        'confidence_drop': result['confidence_drop'],
                        'l2_norm': result['l2_norm'],
                        'linf_norm': result['linf_norm'],
                        'l0_norm': result['l0_norm']
                    }

                    save_result_to_csv(csv_row, OUTPUT_CSV)

                    # Visualization (only for representative params)
                    if (patient_id in vis_patient_ids and
                        c == VIS_PARAMS['cw']['c'] and
                        kappa == VIS_PARAMS['cw']['kappa']):

                        vis_path = visualize_attack_comparison(
                            image, result['adv_image'],
                            patient_id, 'cw', params, mode, VIS_DIR
                        )
                        print(f"      [VIS] Saved: {vis_path}")

        # ====================================================================
        # TASK 2: DeepFool Overshoot Sweep
        # ====================================================================

        print(f"\n  DeepFool Experiments:")

        for overshoot in DEEPFOOL_OVERSHOOT_VALUES:
            for mode in MODES:
                params = {
                    'steps': DEEPFOOL_STEPS,
                    'overshoot': overshoot
                }

                print(f"    overshoot={overshoot:5.2f}, mode={mode:12s}...", end=' ')

                result = run_attack(
                    model=model,
                    image=image,
                    mask=masks[mode],
                    algorithm='deepfool',
                    params=params,
                    attack_mode=mode,
                    device=DEVICE
                )

                if result is None:
                    print("FAILED")
                    continue

                print(f"L2={result['l2_norm']:7.2f}, ASR={result['success']:.0f}")

                # Save to CSV
                csv_row = {
                    'patient_id': patient_id,
                    'algorithm': 'deepfool',
                    'mode': mode,
                    'c': np.nan,
                    'kappa': np.nan,
                    'overshoot': overshoot,
                    'steps': DEEPFOOL_STEPS,
                    'lr': np.nan,
                    'success': result['success'],
                    'clean_prob': result['clean_prob'],
                    'adv_prob': result['adv_prob'],
                    'confidence_drop': result['confidence_drop'],
                    'l2_norm': result['l2_norm'],
                    'linf_norm': result['linf_norm'],
                    'l0_norm': result['l0_norm']
                }

                save_result_to_csv(csv_row, OUTPUT_CSV)

                # Visualization
                if (patient_id in vis_patient_ids and
                    overshoot == VIS_PARAMS['deepfool']['overshoot']):

                    vis_path = visualize_attack_comparison(
                        image, result['adv_image'],
                        patient_id, 'deepfool', params, mode, VIS_DIR
                    )
                    print(f"      [VIS] Saved: {vis_path}")

    # ========================================================================
    # Final Summary
    # ========================================================================

    print("\n" + "="*90)
    print("EXPERIMENTS COMPLETE!")
    print("="*90)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nResults saved to: {OUTPUT_CSV}")
    print(f"Pilot log saved to: {PILOT_LOG}")
    print(f"Visualizations saved to: {VIS_DIR}/")

    if len(skipped_patients) > 0:
        print(f"\n[SURVIVOR BIAS] Skipped {len(skipped_patients)} patients due to random_patch failure:")
        for pid in skipped_patients[:10]:
            print(f"  - {pid}")
        if len(skipped_patients) > 10:
            print(f"  ... and {len(skipped_patients)-10} more")

    print("\n[NEXT STEP] Run post-hoc analysis and generate publication figures")
    print("="*90)


if __name__ == "__main__":
    main()
