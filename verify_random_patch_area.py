"""
CRITICAL VERIFICATION: Is Random Patch truly equal-area?
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("="*90)
print("验证 Random Patch 是否真的等面积")
print("="*90)

# Read epsilon sweep data
df = pd.read_csv('results/epsilon_sweep_new/all_algorithms_consolidated.csv')

# Filter FGSM epsilon=4/255 (clearest data)
df_eps4 = df[abs(df['param_epsilon'] - 0.01568627450980392) < 1e-6]
df_fgsm = df_eps4[df_eps4['algorithm'] == 'fgsm']

print("\n【FGSM epsilon=4/255】L0面积分布:")
print("-"*90)

for mode in ['lesion', 'random_patch', 'full']:
    df_mode = df_fgsm[df_fgsm['mode'] == mode]
    if len(df_mode) == 0:
        continue

    l0_values = df_mode['l0_norm'].values

    print(f"\n{mode.upper()} (N={len(df_mode)}):")
    print(f"  Mean L0:    {l0_values.mean():.0f}")
    print(f"  Std L0:     {l0_values.std():.0f}")
    print(f"  Min L0:     {l0_values.min():.0f}")
    print(f"  Max L0:     {l0_values.max():.0f}")
    print(f"  Median L0:  {np.median(l0_values):.0f}")

# Check epsilon=32/255
print("\n" + "="*90)
print("【FGSM epsilon=32/255】L0面积分布:")
print("-"*90)

df_32 = pd.read_csv('results/epsilon32_extreme_20260313_113624/fgsm_results.csv')

for mode in ['lesion', 'random_patch', 'full']:
    df_mode = df_32[df_32['mode'] == mode]
    if len(df_mode) == 0:
        continue

    l0_values = df_mode['l0_norm'].values

    print(f"\n{mode.upper()} (N={len(df_mode)}):")
    print(f"  Mean L0:    {l0_values.mean():.0f}")
    print(f"  Std L0:     {l0_values.std():.0f}")
    print(f"  Min L0:     {l0_values.min():.0f}")
    print(f"  Max L0:     {l0_values.max():.0f}")

# Pair-wise comparison for same patient
print("\n" + "="*90)
print("配对样本检查（同一患者的Lesion vs Random L0）:")
print("-"*90)

df_lesion = df_32[df_32['mode'] == 'lesion'][['patient_id', 'l0_norm']].rename(columns={'l0_norm': 'lesion_l0'})
df_random = df_32[df_32['mode'] == 'random_patch'][['patient_id', 'l0_norm']].rename(columns={'l0_norm': 'random_l0'})

df_paired = pd.merge(df_lesion, df_random, on='patient_id', how='inner')

if len(df_paired) > 0:
    print(f"\n配对样本数: {len(df_paired)}")
    print(f"\n前10个样本:")
    print(df_paired.head(10).to_string(index=False))

    # Calculate difference
    df_paired['diff'] = df_paired['random_l0'] - df_paired['lesion_l0']
    df_paired['diff_pct'] = (df_paired['diff'] / df_paired['lesion_l0']) * 100

    print(f"\n面积差异统计:")
    print(f"  Mean diff:     {df_paired['diff'].mean():.0f} pixels")
    print(f"  Std diff:      {df_paired['diff'].std():.0f} pixels")
    print(f"  Mean diff %:   {df_paired['diff_pct'].mean():.1f}%")

    if abs(df_paired['diff'].mean()) < 100:
        print("\n✓ PASS: Random Patch 是等面积的！ (差异<100像素)")
    else:
        print(f"\n✗ FAIL: Random Patch 不是等面积的！ (差异={df_paired['diff'].mean():.0f}像素)")

    # Check for outliers
    large_diff = df_paired[abs(df_paired['diff']) > 1000]
    if len(large_diff) > 0:
        print(f"\n警告: {len(large_diff)} 个样本的面积差异>1000像素")
else:
    print("\n无配对样本（患者ID不匹配）")

# Final verdict
print("\n" + "="*90)
print("最终结论:")
print("="*90)

lesion_mean = df_32[df_32['mode']=='lesion']['l0_norm'].mean()
random_mean = df_32[df_32['mode']=='random_patch']['l0_norm'].mean()
full_mean = df_32[df_32['mode']=='full']['l0_norm'].mean()

print(f"\nLesion平均L0:  {lesion_mean:.0f}")
print(f"Random平均L0:  {random_mean:.0f}")
print(f"Full平均L0:    {full_mean:.0f}")

if abs(random_mean - lesion_mean) / lesion_mean < 0.05:
    print("\n✓ Random Patch 是真正的等面积对照组")
    print("  → L∞悖论有效：即使等面积，Random ASR > Lesion ASR")
    print("  → 说明语义定位确实劣于非语义（在L∞约束下）")
elif abs(random_mean - full_mean) / full_mean < 0.05:
    print("\n✗ Random Patch 实际上是全图攻击！")
    print("  → 当前实验设计有严重错误")
    print("  → Random vs Lesion的对比混淆了面积和语义两个变量")
    print("  → 需要重新运行真正的等面积实验")
else:
    print("\n? Random Patch面积介于Lesion和Full之间")
    print(f"  → 面积比: Random/Lesion = {random_mean/lesion_mean:.2f}x")
    print("  → 需要进一步调查mask生成逻辑")
