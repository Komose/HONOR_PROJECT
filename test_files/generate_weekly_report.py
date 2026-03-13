"""
Weekly Report Generator: Visualization & Key Findings
======================================================

Generates publication-ready figures and statistical analysis
for advisor presentations.

Author: HONER Project
Date: 2026-03-12
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'

# Output directories
INPUT_DIR = Path('results/unified_final_rigid_translation')
OUTPUT_DIR = Path('results/weekly_report')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def load_data():
    """Load consolidated results."""
    print("=" * 80)
    print("LOADING DATA (CLEANED PAIRED DATASET)")
    print("=" * 80)

    data_path = INPUT_DIR / 'CLEANED_PAIRED_RESULTS.csv'
    df = pd.read_csv(data_path)

    print(f"Total records: {len(df)}")
    print(f"Algorithms: {df['algorithm'].unique()}")
    print(f"Modes: {df['mode'].unique()}")
    print(f"Patients: {df['patient_id'].nunique()}")
    print(f"Data: Survivor bias eliminated, paired samples only")

    return df


def compute_summary_statistics(df):
    """Compute key statistics for each algorithm-mode combination."""
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    summary = df.groupby(['algorithm', 'mode']).agg({
        'success': ['mean', 'std', 'count'],
        'l2_norm': 'mean',
        'efficiency': 'mean',
        'confidence_drop': 'mean'
    }).round(4)

    summary.columns = ['ASR', 'ASR_std', 'N', 'Avg_L2', 'Avg_Efficiency', 'Avg_ConfDrop']

    print("\n", summary)

    # Save to CSV
    summary.to_csv(OUTPUT_DIR / 'summary_statistics.csv')
    print(f"\n[OK] Saved: {OUTPUT_DIR / 'summary_statistics.csv'}")

    return summary


def plot_asr_comparison(df):
    """
    Figure 1: Attack Success Rate Comparison (Core Figure)
    """
    print("\n" + "=" * 80)
    print("FIGURE 1: ASR Comparison (Lesion vs Random Patch)")
    print("=" * 80)

    # Filter to lesion and random_patch only (exclude full for clarity)
    df_filtered = df[df['mode'].isin(['lesion', 'random_patch'])].copy()

    # Compute mean ASR
    asr_data = df_filtered.groupby(['algorithm', 'mode'])['success'].mean().reset_index()
    asr_data.columns = ['Algorithm', 'Mode', 'ASR']

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Pivot for grouped bar chart
    asr_pivot = asr_data.pivot(index='Algorithm', columns='Mode', values='ASR')

    # Plot
    asr_pivot.plot(kind='bar', ax=ax, width=0.7, edgecolor='black', linewidth=1.2)

    ax.set_title('Attack Success Rate: Lesion vs Random Patch\n(Rigid Translation, Paired Samples)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Attack Algorithm', fontsize=12)
    ax.set_ylabel('Attack Success Rate (ASR)', fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.legend(title='Attack Mode', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3)

    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig1_asr_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[OK] Saved: {OUTPUT_DIR / 'fig1_asr_comparison.png'}")

    # Print key findings
    print("\nKey Findings:")
    for algo in asr_pivot.index:
        lesion_asr = asr_pivot.loc[algo, 'lesion']
        random_asr = asr_pivot.loc[algo, 'random_patch']
        ratio = lesion_asr / random_asr if random_asr > 0 else float('inf')
        print(f"  {algo.upper()}: Lesion={lesion_asr:.3f}, Random={random_asr:.3f}, Ratio={ratio:.2f}x")


def plot_efficiency_comparison(df):
    """
    Figure 2: Attack Efficiency Comparison
    """
    print("\n" + "=" * 80)
    print("FIGURE 2: Attack Efficiency")
    print("=" * 80)

    df_filtered = df[df['mode'].isin(['lesion', 'random_patch'])].copy()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Box plot for efficiency
    sns.boxplot(data=df_filtered, x='algorithm', y='efficiency', hue='mode', ax=ax)

    ax.set_title('Attack Efficiency: Lesion vs Random Patch', fontsize=14, fontweight='bold')
    ax.set_xlabel('Attack Algorithm', fontsize=12)
    ax.set_ylabel('Efficiency (Confidence Drop / L2 Norm)', fontsize=12)
    ax.legend(title='Attack Mode', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig2_efficiency_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[OK] Saved: {OUTPUT_DIR / 'fig2_efficiency_comparison.png'}")


def plot_l2_norm_distribution(df):
    """
    Figure 3: L2 Norm Distribution
    """
    print("\n" + "=" * 80)
    print("FIGURE 3: L2 Norm Distribution")
    print("=" * 80)

    df_filtered = df[df['mode'].isin(['lesion', 'random_patch'])].copy()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    algorithms = df_filtered['algorithm'].unique()

    for idx, algo in enumerate(algorithms):
        ax = axes[idx]
        data = df_filtered[df_filtered['algorithm'] == algo]

        sns.violinplot(data=data, x='mode', y='l2_norm', ax=ax, palette='Set2')

        ax.set_title(f'{algo.upper()}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Attack Mode', fontsize=10)
        ax.set_ylabel('L2 Norm', fontsize=10)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('L2 Perturbation Magnitude Distribution', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig3_l2_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[OK] Saved: {OUTPUT_DIR / 'fig3_l2_distribution.png'}")


def statistical_significance_test(df):
    """
    Perform t-tests for lesion vs random_patch comparison
    """
    print("\n" + "=" * 80)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("=" * 80)

    results = []

    for algo in df['algorithm'].unique():
        lesion_data = df[(df['algorithm'] == algo) & (df['mode'] == 'lesion')]['success']
        random_data = df[(df['algorithm'] == algo) & (df['mode'] == 'random_patch')]['success']

        # Paired t-test (if same patients, otherwise independent)
        # Use independent t-test for safety
        t_stat, p_value = stats.ttest_ind(lesion_data, random_data)

        lesion_mean = lesion_data.mean()
        random_mean = random_data.mean()

        significant = "YES" if p_value < 0.05 else "NO"

        results.append({
            'Algorithm': algo.upper(),
            'Lesion_ASR': f"{lesion_mean:.3f}",
            'Random_ASR': f"{random_mean:.3f}",
            't_statistic': f"{t_stat:.4f}",
            'p_value': f"{p_value:.4f}",
            'Significant': significant
        })

        print(f"\n{algo.upper()}:")
        print(f"  Lesion ASR: {lesion_mean:.3f}")
        print(f"  Random ASR: {random_mean:.3f}")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_value:.4f}")
        print(f"  Significant (α=0.05): {significant}")

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / 'statistical_tests.csv', index=False)
    print(f"\n[OK] Saved: {OUTPUT_DIR / 'statistical_tests.csv'}")

    return results_df


def generate_key_findings(df, summary, stat_tests):
    """
    Generate a text report with key findings for weekly report
    """
    print("\n" + "=" * 80)
    print("GENERATING KEY FINDINGS REPORT")
    print("=" * 80)

    report_lines = []

    report_lines.append("=" * 80)
    report_lines.append("WEEKLY REPORT: KEY FINDINGS")
    report_lines.append("=" * 80)
    report_lines.append("")

    # Section 1: Overview
    report_lines.append("1. EXPERIMENT OVERVIEW")
    report_lines.append("-" * 80)
    report_lines.append(f"   Total Samples: {df['patient_id'].nunique()}")
    report_lines.append(f"   Algorithms Tested: {', '.join(df['algorithm'].unique())}")
    report_lines.append(f"   Attack Modes: {', '.join(df['mode'].unique())}")
    report_lines.append(f"   Total Attacks: {len(df)}")
    report_lines.append("")

    # Section 2: Core Research Question
    report_lines.append("2. CORE RESEARCH QUESTION")
    report_lines.append("-" * 80)
    report_lines.append("   Are lesion regions MORE vulnerable to adversarial attacks")
    report_lines.append("   than random non-lesion regions?")
    report_lines.append("")

    # Section 3: Key Results
    report_lines.append("3. KEY RESULTS (Lesion vs Random Patch)")
    report_lines.append("-" * 80)

    for algo in df['algorithm'].unique():
        lesion_asr = summary.loc[(algo, 'lesion'), 'ASR']
        random_asr = summary.loc[(algo, 'random_patch'), 'ASR']

        # Find corresponding statistical test
        stat_row = stat_tests[stat_tests['Algorithm'] == algo.upper()].iloc[0]
        p_val = float(stat_row['p_value'])

        if lesion_asr > random_asr:
            conclusion = f"Lesion is {lesion_asr/random_asr:.2f}x MORE vulnerable"
        else:
            conclusion = f"Random patch is {random_asr/lesion_asr:.2f}x MORE vulnerable"

        report_lines.append(f"\n   {algo.upper()}:")
        report_lines.append(f"      Lesion ASR:        {lesion_asr:.3f}")
        report_lines.append(f"      Random Patch ASR:  {random_asr:.3f}")
        report_lines.append(f"      Difference:        {abs(lesion_asr - random_asr):.3f}")
        report_lines.append(f"      p-value:           {p_val:.4f}")
        report_lines.append(f"      Conclusion:        {conclusion}")
        report_lines.append(f"      Significant:       {'YES (p<0.05)' if p_val < 0.05 else 'NO (p>=0.05)'}")

    report_lines.append("")

    # Section 4: Overall Conclusion
    report_lines.append("4. OVERALL CONCLUSION")
    report_lines.append("-" * 80)

    # Count how many algorithms show lesion > random
    lesion_higher = 0
    for algo in df['algorithm'].unique():
        lesion_asr = summary.loc[(algo, 'lesion'), 'ASR']
        random_asr = summary.loc[(algo, 'random_patch'), 'ASR']
        if lesion_asr > random_asr:
            lesion_higher += 1

    if lesion_higher >= 2:
        report_lines.append("   The majority of algorithms show HIGHER attack success")
        report_lines.append("   on lesion regions, suggesting that:")
        report_lines.append("")
        report_lines.append("   [OK] The model DOES rely on lesion features for diagnosis")
        report_lines.append("   [OK] Lesion regions are the model's decision-critical areas")
        report_lines.append("   [OK] Defense strategies should prioritize lesion protection")
    else:
        report_lines.append("   Interestingly, random non-lesion regions show HIGHER")
        report_lines.append("   vulnerability in most algorithms, suggesting:")
        report_lines.append("")
        report_lines.append("   ! The model may rely on background/peripheral features")
        report_lines.append("   ! Potential 'shortcut learning' behavior detected")
        report_lines.append("   ! Need to investigate model attention mechanisms")

    report_lines.append("")

    # Section 5: Implications
    report_lines.append("5. IMPLICATIONS FOR DEFENSE")
    report_lines.append("-" * 80)
    report_lines.append("   Based on these findings:")
    if lesion_higher >= 2:
        report_lines.append("   - Focus adversarial training on lesion regions")
        report_lines.append("   - Add robust lesion detection modules")
        report_lines.append("   - Validate model interpretability on lesion features")
    else:
        report_lines.append("   - Review model architecture for feature shortcuts")
        report_lines.append("   - Add attention mechanisms to force lesion focus")
        report_lines.append("   - Consider data augmentation to reduce background bias")

    report_lines.append("")
    report_lines.append("=" * 80)

    # Write to file
    report_text = "\n".join(report_lines)

    with open(OUTPUT_DIR / 'KEY_FINDINGS.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)

    # Print with encoding handling for Windows console
    try:
        print(report_text)
    except UnicodeEncodeError:
        # Fallback: print ASCII-safe version
        print(report_text.encode('ascii', errors='replace').decode('ascii'))
    print(f"\n[OK] Saved: {OUTPUT_DIR / 'KEY_FINDINGS.txt'}")

    return report_text


def plot_algorithm_ranking(df):
    """
    Figure 4: Algorithm Ranking by ASR
    """
    print("\n" + "=" * 80)
    print("FIGURE 4: Algorithm Effectiveness Ranking")
    print("=" * 80)

    # Compute overall ASR (average across all modes)
    overall_asr = df.groupby('algorithm')['success'].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.barh(range(len(overall_asr)), overall_asr.values, color='skyblue', edgecolor='black')
    ax.set_yticks(range(len(overall_asr)))
    ax.set_yticklabels([a.upper() for a in overall_asr.index])
    ax.set_xlabel('Overall Attack Success Rate', fontsize=12)
    ax.set_title('Algorithm Effectiveness Ranking', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, overall_asr.values)):
        ax.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig4_algorithm_ranking.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[OK] Saved: {OUTPUT_DIR / 'fig4_algorithm_ranking.png'}")


def main():
    """Main execution"""
    print("\n" + "=" * 80)
    print("WEEKLY REPORT GENERATOR")
    print("=" * 80)
    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 80)

    # Load data
    df = load_data()

    # Generate summary statistics
    summary = compute_summary_statistics(df)

    # Generate figures
    plot_asr_comparison(df)
    plot_efficiency_comparison(df)
    plot_l2_norm_distribution(df)
    plot_algorithm_ranking(df)

    # Statistical tests
    stat_tests = statistical_significance_test(df)

    # Generate key findings report
    generate_key_findings(df, summary, stat_tests)

    print("\n" + "=" * 80)
    print("[OK] WEEKLY REPORT GENERATION COMPLETE!")
    print("=" * 80)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print("\nGenerated Files:")
    print("  1. summary_statistics.csv      - Quantitative summary")
    print("  2. statistical_tests.csv       - Significance tests")
    print("  3. KEY_FINDINGS.txt            - Text report for presentation")
    print("  4. fig1_asr_comparison.png     - Core comparison figure")
    print("  5. fig2_efficiency_comparison.png")
    print("  6. fig3_l2_distribution.png")
    print("  7. fig4_algorithm_ranking.png")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
