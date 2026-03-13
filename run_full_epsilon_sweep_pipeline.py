"""
Full Epsilon Sweep Pipeline (Post Bug-Fix)
===========================================

Complete pipeline:
1. Run full epsilon sweep (FGSM + PGD)
2. Clean survivor bias (paired samples only)
3. Generate analysis report and plots

Author: HONER Project
Date: 2026-03-13 (Post mask-fix)
"""
import os
import sys
import subprocess
from datetime import datetime
import pandas as pd

print("="*90)
print("FULL EPSILON SWEEP PIPELINE (POST BUG-FIX)")
print("="*90)
print(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\nPipeline stages:")
print("  1. Run FGSM + PGD epsilon sweep (ε = 2,4,8,16,32/255)")
print("  2. Clean survivor bias (keep paired samples only)")
print("  3. Generate ASR comparison plots")
print("="*90)

# Configuration
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_dir = f'results/epsilon_sweep_fixed_{timestamp}'
batch_size = 1  # Required for random patch

print(f"\nOutput directory: {output_dir}")
print(f"Expected runtime: 6-8 hours")
print(f"Checkpoint: Enabled (saves after each mode)\n")

# ============================================================================
# STAGE 1: Run Epsilon Sweep
# ============================================================================
print("\n" + "="*90)
print("STAGE 1: Running Epsilon Sweep Experiment")
print("="*90)

cmd_experiment = [
    sys.executable,
    'run_multi_algorithm_experiments.py',
    '--algorithms', 'fgsm,pgd',
    '--samples', '200',
    '--batch_size', str(batch_size),
    '--enable_random_patch',
    '--output_dir', output_dir,
    '--device', 'cuda'
]

print(f"\nCommand: {' '.join(cmd_experiment)}\n")

try:
    result = subprocess.run(cmd_experiment, check=True, capture_output=False)
    print("\n[STAGE 1 COMPLETE] Epsilon sweep finished successfully")
except subprocess.CalledProcessError as e:
    print(f"\n[STAGE 1 FAILED] Experiment failed with exit code {e.returncode}")
    print("Pipeline aborted. Check experiment logs.")
    sys.exit(1)
except KeyboardInterrupt:
    print("\n[INTERRUPTED] Experiment interrupted by user")
    sys.exit(1)

# ============================================================================
# STAGE 2: Clean Survivor Bias
# ============================================================================
print("\n" + "="*90)
print("STAGE 2: Cleaning Survivor Bias")
print("="*90)
print("\nFiltering to paired samples (Lesion + Random + Full all successful)...")

consolidated_file = os.path.join(output_dir, 'all_algorithms_consolidated.csv')

if not os.path.exists(consolidated_file):
    print(f"[ERROR] Consolidated file not found: {consolidated_file}")
    print("Skipping survivor bias cleaning.")
else:
    try:
        # Read data
        df = pd.read_csv(consolidated_file)

        print(f"\nRaw data: {len(df)} records")
        print(f"Algorithms: {df['algorithm'].unique()}")
        print(f"Modes: {df['mode'].unique()}")

        # Get patient IDs that have all three modes
        patient_mode_counts = df.groupby(['algorithm', 'patient_id'])['mode'].nunique()

        # Filter to patients with all 3 modes (lesion, random_patch, full)
        paired_patients = patient_mode_counts[patient_mode_counts == 3].reset_index()

        # Get intersection of patient IDs across all algorithms
        patient_sets = [set(paired_patients[paired_patients['algorithm']==algo]['patient_id'])
                       for algo in df['algorithm'].unique()]
        common_patients = set.intersection(*patient_sets) if patient_sets else set()

        print(f"\nPatients with all 3 modes per algorithm:")
        for algo in df['algorithm'].unique():
            algo_paired = set(paired_patients[paired_patients['algorithm']==algo]['patient_id'])
            print(f"  {algo.upper()}: {len(algo_paired)}")

        print(f"\nPatients in ALL algorithms with all modes: {len(common_patients)}")

        # Filter dataframe
        df_cleaned = df[df['patient_id'].isin(common_patients)].copy()

        # Calculate dropout statistics
        original_patients = df['patient_id'].nunique()
        retained_patients = len(common_patients)
        dropout_rate = (original_patients - retained_patients) / original_patients * 100

        print(f"\n[SURVIVOR BIAS STATISTICS]")
        print(f"  Original patients:  {original_patients}")
        print(f"  Retained patients:  {retained_patients}")
        print(f"  Dropout:            {original_patients - retained_patients} ({dropout_rate:.1f}%)")
        print(f"  Retention rate:     {100-dropout_rate:.1f}%")

        # Save cleaned data
        cleaned_file = os.path.join(output_dir, 'CLEANED_PAIRED_RESULTS.csv')
        df_cleaned.to_csv(cleaned_file, index=False)

        print(f"\n[STAGE 2 COMPLETE] Cleaned data saved to:")
        print(f"  {cleaned_file}")
        print(f"  Final records: {len(df_cleaned)}")

    except Exception as e:
        print(f"[STAGE 2 ERROR] Survivor bias cleaning failed: {e}")
        import traceback
        traceback.print_exc()
        print("\nContinuing to next stage with original data...")
        cleaned_file = consolidated_file

# ============================================================================
# STAGE 3: Generate Plots
# ============================================================================
print("\n" + "="*90)
print("STAGE 3: Generating ASR Comparison Plots")
print("="*90)

plot_script = """
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Read cleaned data
df = pd.read_csv('{input_file}')

# Create plots directory
plot_dir = '{output_dir}/plots'
os.makedirs(plot_dir, exist_ok=True)

# Get epsilon values from param_epsilon column
epsilon_col = 'param_epsilon'
if epsilon_col not in df.columns:
    print("ERROR: param_epsilon column not found")
    import sys
    sys.exit(1)

# Map epsilon decimals to labels
epsilon_map = {{
    0.00784313725490196: 2,
    0.01568627450980392: 4,
    0.03137254901960784: 8,
    0.06274509803921569: 16,
    0.12549019607843137: 32
}}

df['epsilon_label'] = df[epsilon_col].apply(
    lambda x: min(epsilon_map.keys(), key=lambda k: abs(k-x))
).map(epsilon_map)

# Generate plots for each algorithm
for algo in ['fgsm', 'pgd']:
    df_algo = df[df['algorithm'] == algo]

    if len(df_algo) == 0:
        print(f"No data for {{algo.upper()}}, skipping...")
        continue

    # Calculate ASR for each epsilon and mode
    summary = df_algo.groupby(['epsilon_label', 'mode']).agg({{
        'success': ['mean', 'std', 'count']
    }}).reset_index()

    summary.columns = ['epsilon', 'mode', 'asr_mean', 'asr_std', 'count']
    summary['asr_mean'] *= 100  # Convert to percentage
    summary['asr_std'] *= 100

    # Pivot for plotting
    pivot = summary.pivot(index='epsilon', columns='mode', values='asr_mean')
    pivot_std = summary.pivot(index='epsilon', columns='mode', values='asr_std')

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    x = pivot.index
    width = 0.25
    x_pos = np.arange(len(x))

    colors = {{'lesion': '#e74c3c', 'random_patch': '#3498db', 'full': '#95a5a6'}}
    labels = {{'lesion': 'Lesion', 'random_patch': 'Random Patch', 'full': 'Full Image'}}

    for i, mode in enumerate(['lesion', 'random_patch', 'full']):
        if mode in pivot.columns:
            ax.bar(x_pos + i*width, pivot[mode], width,
                   yerr=pivot_std[mode] if mode in pivot_std.columns else None,
                   label=labels[mode], color=colors[mode], alpha=0.8,
                   capsize=5)

    ax.set_xlabel('Epsilon (ε/255)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Attack Success Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'{{algo.upper()}} Attack Success Rate vs Epsilon\\n(Post Bug-Fix, N={{len(df_algo["patient_id"].unique())}} paired patients)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels([f'{{int(e)}}' for e in x])
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 100)

    # Add annotations for key findings
    if 'lesion' in pivot.columns and 'random_patch' in pivot.columns:
        for idx, eps in enumerate(x):
            lesion_asr = pivot.loc[eps, 'lesion']
            random_asr = pivot.loc[eps, 'random_patch']
            if lesion_asr > random_asr:
                ratio = lesion_asr / random_asr if random_asr > 0 else float('inf')
                if ratio > 1.5:
                    ax.text(x_pos[idx], max(lesion_asr, random_asr) + 5,
                           f'{{ratio:.1f}}x', ha='center', fontsize=9,
                           color='red', fontweight='bold')

    plt.tight_layout()
    save_path = f'{{plot_dir}}/{{algo}}_asr_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[PLOT] Saved {{algo.upper()}} ASR plot: {{save_path}}")

print("\\n[STAGE 3 COMPLETE] All plots generated")
""".format(input_file=cleaned_file if 'cleaned_file' in locals() else consolidated_file,
           output_dir=output_dir)

# Write and execute plot script
plot_script_file = os.path.join(output_dir, 'generate_plots.py')
with open(plot_script_file, 'w') as f:
    f.write(plot_script)

try:
    subprocess.run([sys.executable, plot_script_file], check=True)
except subprocess.CalledProcessError as e:
    print(f"[STAGE 3 ERROR] Plot generation failed: {e}")
except Exception as e:
    print(f"[STAGE 3 ERROR] Unexpected error: {e}")

# ============================================================================
# PIPELINE COMPLETE
# ============================================================================
print("\n" + "="*90)
print("PIPELINE COMPLETE!")
print("="*90)
print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\nAll results saved to: {output_dir}/")
print("\nKey files:")
print(f"  - Raw data:     all_algorithms_consolidated.csv")
if 'cleaned_file' in locals():
    print(f"  - Cleaned data: CLEANED_PAIRED_RESULTS.csv")
print(f"  - Plots:        plots/fgsm_asr_comparison.png")
print(f"                  plots/pgd_asr_comparison.png")
print("="*90)
