
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Read cleaned data
df = pd.read_csv('results/epsilon_sweep_fixed_20260313_170716\CLEANED_PAIRED_RESULTS.csv')

# Create plots directory
plot_dir = 'results/epsilon_sweep_fixed_20260313_170716/plots'
os.makedirs(plot_dir, exist_ok=True)

# Get epsilon values from param_epsilon column
epsilon_col = 'param_epsilon'
if epsilon_col not in df.columns:
    print("ERROR: param_epsilon column not found")
    import sys
    sys.exit(1)

# Map epsilon decimals to labels
epsilon_map = {
    0.00784313725490196: 2,
    0.01568627450980392: 4,
    0.03137254901960784: 8,
    0.06274509803921569: 16,
    0.12549019607843137: 32
}

df['epsilon_label'] = df[epsilon_col].apply(
    lambda x: min(epsilon_map.keys(), key=lambda k: abs(k-x))
).map(epsilon_map)

# Generate plots for each algorithm
for algo in ['fgsm', 'pgd']:
    df_algo = df[df['algorithm'] == algo]

    if len(df_algo) == 0:
        print(f"No data for {algo.upper()}, skipping...")
        continue

    # Calculate ASR for each epsilon and mode
    summary = df_algo.groupby(['epsilon_label', 'mode']).agg({
        'success': ['mean', 'std', 'count']
    }).reset_index()

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

    colors = {'lesion': '#e74c3c', 'random_patch': '#3498db', 'full': '#95a5a6'}
    labels = {'lesion': 'Lesion', 'random_patch': 'Random Patch', 'full': 'Full Image'}

    for i, mode in enumerate(['lesion', 'random_patch', 'full']):
        if mode in pivot.columns:
            ax.bar(x_pos + i*width, pivot[mode], width,
                   yerr=pivot_std[mode] if mode in pivot_std.columns else None,
                   label=labels[mode], color=colors[mode], alpha=0.8,
                   capsize=5)

    ax.set_xlabel('Epsilon (¦Å/255)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Attack Success Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'{algo.upper()} Attack Success Rate vs Epsilon\n(Post Bug-Fix, N={len(df_algo["patient_id"].unique())} paired patients)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels([f'{int(e)}' for e in x])
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
                           f'{ratio:.1f}x', ha='center', fontsize=9,
                           color='red', fontweight='bold')

    plt.tight_layout()
    save_path = f'{plot_dir}/{algo}_asr_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[PLOT] Saved {algo.upper()} ASR plot: {save_path}")

print("\n[STAGE 3 COMPLETE] All plots generated")
