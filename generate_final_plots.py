"""
Generate ASR Comparison Plots (Post Bug-Fix)
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

print("Generating ASR comparison plots...")

# Read cleaned data
input_file = 'results/epsilon_sweep_fixed_20260313_170727/CLEANED_PAIRED_RESULTS.csv'
df = pd.read_csv(input_file)

print(f"Loaded {len(df)} records")
print(f"Algorithms: {df['algorithm'].unique()}")
print(f"Patients: {df['patient_id'].nunique()}")

# Create plots directory
plot_dir = 'results/epsilon_sweep_fixed_20260313_170727/plots'
os.makedirs(plot_dir, exist_ok=True)

# Map epsilon decimals to labels
epsilon_map = {
    0.00784313725490196: 2,
    0.01568627450980392: 4,
    0.03137254901960784: 8,
    0.06274509803921569: 16,
    0.12549019607843137: 32
}

df['epsilon_label'] = df['param_epsilon'].apply(
    lambda x: min(epsilon_map.keys(), key=lambda k: abs(k-x))
).map(epsilon_map)

# Generate plots for each algorithm
for algo in ['fgsm', 'pgd']:
    df_algo = df[df['algorithm'] == algo]

    if len(df_algo) == 0:
        print(f"No data for {algo.upper()}, skipping...")
        continue

    print(f"\nProcessing {algo.upper()}...")

    # Calculate ASR for each epsilon and mode
    summary = df_algo.groupby(['epsilon_label', 'mode']).agg({
        'success': ['mean', 'std', 'count']
    }).reset_index()

    summary.columns = ['epsilon', 'mode', 'asr_mean', 'asr_std', 'count']
    summary['asr_mean'] *= 100
    summary['asr_std'] *= 100

    print(summary)

    # Pivot for plotting
    pivot = summary.pivot(index='epsilon', columns='mode', values='asr_mean')
    pivot_std = summary.pivot(index='epsilon', columns='mode', values='asr_std')

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 7))

    x = pivot.index
    width = 0.25
    x_pos = np.arange(len(x))

    colors = {'lesion': '#e74c3c', 'random_patch': '#3498db', 'full': '#95a5a6'}
    labels = {'lesion': 'Lesion (Semantic Target)',
              'random_patch': 'Random Patch (Non-Semantic)',
              'full': 'Full Image'}

    bars = {}
    for i, mode in enumerate(['lesion', 'random_patch', 'full']):
        if mode in pivot.columns:
            bars[mode] = ax.bar(
                x_pos + i*width,
                pivot[mode],
                width,
                yerr=pivot_std[mode] if mode in pivot_std.columns else None,
                label=labels[mode],
                color=colors[mode],
                alpha=0.85,
                capsize=5,
                edgecolor='black',
                linewidth=0.5
            )

    ax.set_xlabel('Epsilon (per-pixel L-inf constraint)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Attack Success Rate (%)', fontsize=13, fontweight='bold')

    title = f'{algo.upper()} Attack Success Rate vs Epsilon\n'
    title += f'Post Bug-Fix: Equal-Area Comparison (N={df_algo["patient_id"].nunique()} paired patients)'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    ax.set_xticks(x_pos + width)
    ax.set_xticklabels([f'{int(e)}/255' for e in x], fontsize=11)
    ax.legend(fontsize=11, loc='upper left', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 105)

    # Add value labels on bars
    for mode in ['lesion', 'random_patch', 'full']:
        if mode in bars:
            for bar in bars[mode]:
                height = bar.get_height()
                if height > 2:
                    ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                           f'{height:.1f}%',
                           ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Add significance annotations
    if 'lesion' in pivot.columns and 'random_patch' in pivot.columns:
        for idx, eps in enumerate(x):
            lesion_asr = pivot.loc[eps, 'lesion']
            random_asr = pivot.loc[eps, 'random_patch']
            if lesion_asr > random_asr + 5:
                ratio = lesion_asr / random_asr if random_asr > 0 else float('inf')
                y_pos = max(lesion_asr, random_asr) + 8
                ax.text(x_pos[idx], y_pos,
                       f'{ratio:.1f}x',
                       ha='center', fontsize=10,
                       color='darkred', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    plt.tight_layout()
    save_path = f'{plot_dir}/{algo}_asr_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[SAVED] {save_path}")

print("\nAll plots generated successfully!")
print(f"Location: {plot_dir}/")
