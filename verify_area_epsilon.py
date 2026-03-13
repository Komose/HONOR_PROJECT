import pandas as pd
import numpy as np

print("="*90)
print("验证L∞攻击的面积与ε使用情况")
print("="*90)

# 读取ε=32/255的数据
df = pd.read_csv('results/epsilon32_extreme_20260313_113624/fgsm_results.csv')

print("\n【FGSM ε=32/255】关键指标对比:")
print("-"*90)

for mode in ['lesion', 'random_patch', 'full']:
    df_mode = df[df['mode'] == mode]
    if len(df_mode) == 0:
        continue

    print(f"\n{mode.upper()} 模式 (N={len(df_mode)}):")
    print(f"  ε (L∞):        {df_mode['linf_norm'].mean():.6f} (32/255 = {32/255:.6f})")
    print(f"  L0 (像素数):    {df_mode['l0_norm'].mean():.0f} ± {df_mode['l0_norm'].std():.0f}")
    print(f"  L2 (总能量):    {df_mode['l2_norm'].mean():.3f} ± {df_mode['l2_norm'].std():.3f}")
    print(f"  ASR:           {df_mode['success'].mean()*100:.1f}%")
    print(f"  效率 (ΔProb/L2): {df_mode['efficiency'].mean():.6f}")

# 计算比值
lesion_l0 = df[df['mode']=='lesion']['l0_norm'].mean()
random_l0 = df[df['mode']=='random_patch']['l0_norm'].mean()
full_l0 = df[df['mode']=='full']['l0_norm'].mean()

print("\n" + "="*90)
print("面积与能量比值:")
print("-"*90)
print(f"Random L0 / Lesion L0: {random_l0/lesion_l0:.2f}x (面积)")
print(f"Full L0 / Lesion L0:   {full_l0/lesion_l0:.2f}x (面积)")

lesion_l2 = df[df['mode']=='lesion']['l2_norm'].mean()
random_l2 = df[df['mode']=='random_patch']['l2_norm'].mean()
full_l2 = df[df['mode']=='full']['l2_norm'].mean()

print(f"\nRandom L2 / Lesion L2: {random_l2/lesion_l2:.2f}x (能量)")
print(f"Full L2 / Lesion L2:   {full_l2/lesion_l2:.2f}x (能量)")

# 理论计算
area_ratio = random_l0 / lesion_l0
theoretical_l2_ratio = np.sqrt(area_ratio)
actual_l2_ratio = random_l2 / lesion_l2

print("\n" + "="*90)
print("理论验证:")
print("-"*90)
print(f"面积比:            {area_ratio:.2f}x")
print(f"理论L2比 (√面积比): {theoretical_l2_ratio:.2f}x")
print(f"实际L2比:          {actual_l2_ratio:.2f}x")
print(f"符合度:            {(actual_l2_ratio/theoretical_l2_ratio)*100:.1f}%")

print("\n说明: L∞约束下，每个像素最大扰动ε固定，")
print("      因此 L2 ∝ √(L0·ε²) = ε·√L0")
print("      面积越大 → L2越大")
