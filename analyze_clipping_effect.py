"""
Analyze Pixel Clipping Effect on L2 Norm Differences
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("="*90)
print("PIXEL CLIPPING EFFECT ANALYSIS")
print("="*90)

# Load cleaned paired data
df = pd.read_csv('results/epsilon_sweep_fixed_20260313_170727/CLEANED_PAIRED_RESULTS.csv')

print(f"\nLoaded {len(df)} records from {df['patient_id'].nunique()} patients")

# Calculate normalized L2 per pixel: L2 / sqrt(L0)
df['l2_per_pixel'] = df['l2_norm'] / np.sqrt(df['l0_norm'])

# Analyze by algorithm and mode
print("\n" + "="*90)
print("L2 NORM ANALYSIS")
print("="*90)

for algo in ['fgsm', 'pgd']:
    print(f"\n{algo.upper()}:")
    print("-"*90)

    df_algo = df[df['algorithm'] == algo]

    summary = df_algo.groupby('mode').agg({
        'l0_norm': ['mean', 'std'],
        'l2_norm': ['mean', 'std'],
        'linf_norm': ['mean', 'std'],
        'l2_per_pixel': ['mean', 'std']
    }).round(4)

    print(summary)

# Detailed epsilon-wise analysis
print("\n" + "="*90)
print("EPSILON-WISE ANALYSIS: Does clipping explain L2 differences?")
print("="*90)

# Map epsilon
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

for algo in ['fgsm', 'pgd']:
    print(f"\n{algo.upper()} - Per Epsilon:")
    print("-"*90)

    df_algo = df[df['algorithm'] == algo]

    for eps in [2, 4, 8, 16, 32]:
        df_eps = df_algo[df_algo['epsilon_label'] == eps]

        if len(df_eps) == 0:
            continue

        print(f"\nε={eps}/255:")

        for mode in ['lesion', 'random_patch']:
            df_mode = df_eps[df_eps['mode'] == mode]

            if len(df_mode) == 0:
                continue

            l0_mean = df_mode['l0_norm'].mean()
            l2_mean = df_mode['l2_norm'].mean()
            linf_mean = df_mode['linf_norm'].mean()
            l2_per_pixel = df_mode['l2_per_pixel'].mean()

            # Theoretical L2 if no clipping: L2_theory = epsilon * sqrt(L0)
            l2_theory = linf_mean * np.sqrt(l0_mean)
            clipping_loss = (l2_theory - l2_mean) / l2_theory * 100 if l2_theory > 0 else 0

            print(f"  {mode:15s}: L0={l0_mean:8.0f}, L2={l2_mean:6.3f}, "
                  f"L∞={linf_mean:.6f}, L2/√L0={l2_per_pixel:.6f}")
            print(f"                   L2_theory={l2_theory:6.3f}, "
                  f"Clipping loss={(l2_theory-l2_mean):6.3f} ({clipping_loss:5.1f}%)")

# Check if L2 difference persists after normalizing by sqrt(L0)
print("\n" + "="*90)
print("KEY QUESTION: Does Lesion have higher L2/√L0 than Random?")
print("="*90)

for algo in ['fgsm', 'pgd']:
    df_algo = df[df['algorithm'] == algo]

    lesion_l2pp = df_algo[df_algo['mode']=='lesion']['l2_per_pixel'].mean()
    random_l2pp = df_algo[df_algo['mode']=='random_patch']['l2_per_pixel'].mean()

    print(f"\n{algo.upper()}:")
    print(f"  Lesion L2/√L0:  {lesion_l2pp:.6f}")
    print(f"  Random L2/√L0:  {random_l2pp:.6f}")
    print(f"  Ratio:          {lesion_l2pp/random_l2pp:.3f}")

    if abs(lesion_l2pp - random_l2pp) < 0.001:
        print("  → L2 difference is purely due to L0 difference (clipping uniform)")
    else:
        print("  → L2 difference persists even after L0 normalization!")
        print("  → Clipping effect is NOT the main cause")

# Plot L2 vs L0 to visualize relationship
print("\n" + "="*90)
print("Generating scatter plots...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for idx, algo in enumerate(['fgsm', 'pgd']):
    ax = axes[idx]
    df_algo = df[df['algorithm'] == algo]

    for mode, color, label in [('lesion', 'red', 'Lesion'),
                                ('random_patch', 'blue', 'Random Patch')]:
        df_mode = df_algo[df_algo['mode'] == mode]
        ax.scatter(df_mode['l0_norm'], df_mode['l2_norm'],
                  alpha=0.3, s=20, c=color, label=label)

    # Plot theoretical line: L2 = epsilon * sqrt(L0) for each epsilon
    for eps_val in df_algo['param_epsilon'].unique():
        l0_range = np.linspace(0, 50000, 100)
        l2_theory = eps_val * np.sqrt(l0_range)
        ax.plot(l0_range, l2_theory, '--', linewidth=1, alpha=0.5,
               label=f'Theory ε={eps_val*255:.0f}/255')

    ax.set_xlabel('L0 Norm (pixels modified)', fontsize=12, fontweight='bold')
    ax.set_ylabel('L2 Norm (perturbation energy)', fontsize=12, fontweight='bold')
    ax.set_title(f'{algo.upper()}: L2 vs L0 (Clipping Effect Analysis)',
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

plt.tight_layout()
save_path = 'results/epsilon_sweep_fixed_20260313_170727/plots/clipping_analysis.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Saved: {save_path}")

print("\n" + "="*90)
print("CONCLUSION")
print("="*90)
print("""
If pixel clipping is the main cause of L2 differences:
  → Lesion (high-intensity regions) would have LOWER L2 (more clipping)
  → Random (mixed regions) would have HIGHER L2 (less clipping)

If the data shows Lesion L2 > Random L2:
  → Clipping is NOT the main cause
  → Other factors dominate (gradient strength, mask topology, etc.)
""")
