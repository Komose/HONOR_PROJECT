"""
Extract epsilon comparison data to CSV format for visualization
"""
import pandas as pd

results = {}

# Load epsilon sweep data
df_sweep = pd.read_csv('results/epsilon_sweep_new/all_algorithms_consolidated.csv')

epsilon_map = {
    0.00392156862745098: 1,
    0.00784313725490196: 2,
    0.01568627450980392: 4,
    0.06274509803921569: 16
}

for eps_dec, eps_int in epsilon_map.items():
    df_eps = df_sweep[abs(df_sweep['param_epsilon'] - eps_dec) < 1e-6]

    for algo in ['fgsm', 'pgd']:
        df_algo = df_eps[df_eps['algorithm'] == algo]
        if len(df_algo) == 0:
            continue

        for mode in ['lesion', 'random_patch', 'full']:
            df_mode = df_algo[df_algo['mode'] == mode]
            if len(df_mode) == 0:
                continue

            key = (eps_int, algo, mode)
            results[key] = {
                'epsilon': eps_int,
                'algorithm': algo,
                'mode': mode,
                'n': len(df_mode),
                'asr': df_mode['success'].mean() * 100,
                'l2_mean': df_mode['l2_norm'].mean(),
                'l2_std': df_mode['l2_norm'].std(),
                'linf_mean': df_mode['linf_norm'].mean(),
                'efficiency_mean': df_mode['efficiency'].mean()
            }

# Load epsilon=32 data
df_32_fgsm = pd.read_csv('results/epsilon32_extreme_20260313_113624/fgsm_results.csv')
for mode in ['lesion', 'random_patch', 'full']:
    df_mode = df_32_fgsm[df_32_fgsm['mode'] == mode]
    if len(df_mode) > 0:
        key = (32, 'fgsm', mode)
        results[key] = {
            'epsilon': 32,
            'algorithm': 'fgsm',
            'mode': mode,
            'n': len(df_mode),
            'asr': df_mode['success'].mean() * 100,
            'l2_mean': df_mode['l2_norm'].mean(),
            'l2_std': df_mode['l2_norm'].std(),
            'linf_mean': df_mode['linf_norm'].mean(),
            'efficiency_mean': df_mode['efficiency'].mean()
        }

df_32_pgd = pd.read_csv('results/epsilon32_extreme_20260313_113624/pgd_checkpoint.csv')
df_mode = df_32_pgd[df_32_pgd['mode'] == 'lesion']
if len(df_mode) > 0:
    results[(32, 'pgd', 'lesion')] = {
        'epsilon': 32,
        'algorithm': 'pgd',
        'mode': 'lesion',
        'n': len(df_mode),
        'asr': df_mode['success'].mean() * 100,
        'l2_mean': df_mode['l2_norm'].mean(),
        'l2_std': df_mode['l2_norm'].std(),
        'linf_mean': df_mode['linf_norm'].mean(),
        'efficiency_mean': df_mode['efficiency'].mean()
    }

# Convert to DataFrame
df_out = pd.DataFrame([v for v in results.values()])
df_out = df_out.sort_values(['algorithm', 'epsilon', 'mode'])

# Save to CSV
output_path = 'results/epsilon_sweep_data_for_visualization.csv'
df_out.to_csv(output_path, index=False)

print(f"Saved epsilon comparison data to: {output_path}")
print(f"\nShape: {df_out.shape}")
print(f"\nPreview:")
print(df_out.head(15))
