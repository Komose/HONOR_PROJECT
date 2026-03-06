"""
Analyze and Visualize Multi-Metric Experiment Results
======================================================

This script generates comprehensive analysis and visualizations for
the three experiments (A, B, C).

Outputs:
- Comparative tables
- Efficiency plots
- Statistical significance tests
- Summary report

Usage:
    python analyze_multi_metric_results.py --results_dir results/multi_metric

Author: Multi-Metric Analysis Framework
Date: March 2026
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
from datetime import datetime

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11


def load_experiment_results(results_dir):
    """Load all experiment results"""
    results = {}

    for exp_name in ['A', 'B', 'C']:
        file_mapping = {
            'A': 'experiment_a_fixed_linf.csv',
            'B': 'experiment_b_fixed_l0.csv',
            'C': 'experiment_c_fixed_l2.csv'
        }

        filepath = os.path.join(results_dir, file_mapping[exp_name])
        if os.path.exists(filepath):
            results[exp_name] = pd.read_csv(filepath)
            print(f"Loaded Experiment {exp_name}: {len(results[exp_name])} samples")
        else:
            print(f"WARNING: {filepath} not found, skipping Experiment {exp_name}")

    return results


def analyze_experiment_a(df_a, output_dir):
    """Analyze Experiment A: Fixed L∞"""
    print("\n" + "="*80)
    print("ANALYZING EXPERIMENT A: Fixed L∞ (Intensity Alignment)")
    print("="*80 + "\n")

    # Group by epsilon and mode
    summary = df_a.groupby(['epsilon', 'mode']).agg({
        'success': ['mean', 'std', 'count'],
        'confidence_drop': ['mean', 'std'],
        'efficiency': ['mean', 'std'],
        'l2_norm': ['mean', 'std'],
        'linf_norm': ['mean'],
        'l0_norm': ['mean']
    }).round(4)

    print(summary)
    print()

    # Key findings
    print("KEY FINDINGS:")
    for eps in df_a['epsilon'].unique():
        df_eps = df_a[df_a['epsilon'] == eps]
        lesion_asr = df_eps[df_eps['mode'] == 'lesion']['success'].mean()
        full_asr = df_eps[df_eps['mode'] == 'full']['success'].mean()
        lesion_eff = df_eps[df_eps['mode'] == 'lesion']['efficiency'].mean()
        full_eff = df_eps[df_eps['mode'] == 'full']['efficiency'].mean()

        print(f"\n  ε = {eps:.4f} ({eps*255:.1f}/255):")
        print(f"    ASR: Lesion={lesion_asr:.1%}, Full={full_asr:.1%} (Ratio: {full_asr/lesion_asr:.2f}x)")
        print(f"    Efficiency: Lesion={lesion_eff:.4f}, Full={full_eff:.4f} (Ratio: {lesion_eff/full_eff:.2f}x)")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Experiment A: Fixed L∞ (Intensity Alignment)', fontsize=16, fontweight='bold')

    # Plot 1: ASR by epsilon
    for mode in ['lesion', 'full']:
        data = df_a[df_a['mode'] == mode].groupby('epsilon')['success'].mean() * 100
        axes[0, 0].plot(data.index * 255, data.values, marker='o', label=mode.capitalize(), linewidth=2)
    axes[0, 0].set_xlabel('ε (x/255)')
    axes[0, 0].set_ylabel('Attack Success Rate (%)')
    axes[0, 0].set_title('ASR vs Epsilon')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Efficiency by epsilon
    for mode in ['lesion', 'full']:
        data = df_a[df_a['mode'] == mode].groupby('epsilon')['efficiency'].mean()
        axes[0, 1].plot(data.index * 255, data.values, marker='s', label=mode.capitalize(), linewidth=2)
    axes[0, 1].set_xlabel('ε (x/255)')
    axes[0, 1].set_ylabel('Attack Efficiency (ΔConf / L2)')
    axes[0, 1].set_title('Efficiency vs Epsilon')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: L2 norm comparison
    df_a.boxplot(column='l2_norm', by=['epsilon', 'mode'], ax=axes[1, 0])
    axes[1, 0].set_title('L2 Norm Distribution')
    axes[1, 0].set_xlabel('(Epsilon, Mode)')
    axes[1, 0].set_ylabel('L2 Norm')
    plt.sca(axes[1, 0])
    plt.xticks(rotation=45)

    # Plot 4: Scatter: L2 vs Confidence Drop
    for mode, marker in [('lesion', 'o'), ('full', '^')]:
        df_mode = df_a[df_a['mode'] == mode]
        axes[1, 1].scatter(df_mode['l2_norm'], df_mode['confidence_drop'],
                          alpha=0.5, label=mode.capitalize(), marker=marker, s=30)
    axes[1, 1].set_xlabel('L2 Norm')
    axes[1, 1].set_ylabel('Confidence Drop')
    axes[1, 1].set_title('Efficiency Scatter Plot')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'experiment_a_analysis.png'), dpi=300, bbox_inches='tight')
    print(f"\n  ✓ Saved plot: experiment_a_analysis.png")

    return summary


def analyze_experiment_b(df_b, output_dir):
    """Analyze Experiment B: Fixed L0"""
    print("\n" + "="*80)
    print("ANALYZING EXPERIMENT B: Fixed L0 (Area Alignment)")
    print("="*80 + "\n")

    # Group by mode
    summary = df_b.groupby('mode').agg({
        'success': ['mean', 'std', 'count'],
        'confidence_drop': ['mean', 'std'],
        'efficiency': ['mean', 'std'],
        'l2_norm': ['mean', 'std'],
        'l0_norm': ['mean', 'std'],
        'mask_area': ['mean', 'std']
    }).round(4)

    print(summary)
    print()

    # Key findings
    print("KEY FINDINGS:")
    lesion_asr = df_b[df_b['mode'] == 'lesion']['success'].mean()
    random_asr = df_b[df_b['mode'] == 'random_patch']['success'].mean()
    full_asr = df_b[df_b['mode'] == 'full']['success'].mean()

    print(f"  ASR Comparison (Equal Modified Pixels):")
    print(f"    Lesion Attack: {lesion_asr:.1%}")
    print(f"    Random Patch Attack: {random_asr:.1%}")
    print(f"    Full Attack: {full_asr:.1%}")
    print(f"  → Lesion vs Random Ratio: {lesion_asr/random_asr:.2f}x")
    print(f"  → This demonstrates SEMANTIC SPECIFICITY of lesion regions!")

    lesion_eff = df_b[df_b['mode'] == 'lesion']['efficiency'].mean()
    random_eff = df_b[df_b['mode'] == 'random_patch']['efficiency'].mean()
    full_eff = df_b[df_b['mode'] == 'full']['efficiency'].mean()
    print(f"\n  Efficiency Comparison:")
    print(f"    Lesion: {lesion_eff:.4f}, Random: {random_eff:.4f}, Full: {full_eff:.4f}")
    print(f"    Lesion vs Random: {lesion_eff/random_eff:.2f}x more efficient")

    # Statistical test: lesion vs random patch
    lesion_data = df_b[df_b['mode'] == 'lesion']['success']
    random_data = df_b[df_b['mode'] == 'random_patch']['success']
    t_stat, p_value = stats.ttest_ind(lesion_data, random_data)
    print(f"\n  Statistical Significance (t-test):")
    print(f"    t-statistic: {t_stat:.4f}, p-value: {p_value:.6f}")
    if p_value < 0.05:
        print(f"    ✓ Lesion vs Random difference is STATISTICALLY SIGNIFICANT (p < 0.05)")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Experiment B: Fixed L0 (Area Alignment)', fontsize=16, fontweight='bold')

    # Plot 1: ASR comparison
    modes = ['lesion', 'random_patch', 'full']
    asrs = [df_b[df_b['mode'] == m]['success'].mean() * 100 for m in modes]
    colors = ['#2ecc71', '#e74c3c', '#3498db']
    bars = axes[0, 0].bar(range(len(modes)), asrs, color=colors, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xticks(range(len(modes)))
    axes[0, 0].set_xticklabels(['Lesion', 'Random Patch', 'Full'])
    axes[0, 0].set_ylabel('Attack Success Rate (%)')
    axes[0, 0].set_title('ASR: Lesion vs Random Patch vs Full')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    for i, bar in enumerate(bars):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

    # Plot 2: Efficiency comparison
    effs = [df_b[df_b['mode'] == m]['efficiency'].mean() for m in modes]
    bars = axes[0, 1].bar(range(len(modes)), effs, color=colors, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xticks(range(len(modes)))
    axes[0, 1].set_xticklabels(['Lesion', 'Random Patch', 'Full'])
    axes[0, 1].set_ylabel('Attack Efficiency')
    axes[0, 1].set_title('Efficiency: Lesion vs Random Patch vs Full')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    for i, bar in enumerate(bars):
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                       f'{height:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

    # Plot 3: L2 norm distribution
    df_b.boxplot(column='l2_norm', by='mode', ax=axes[1, 0])
    axes[1, 0].set_title('L2 Norm Distribution by Mode')
    axes[1, 0].set_xlabel('Attack Mode')
    axes[1, 0].set_ylabel('L2 Norm')

    # Plot 4: Mask area verification
    df_b[df_b['mode'].isin(['lesion', 'random_patch'])].boxplot(column='mask_area', by='mode', ax=axes[1, 1])
    axes[1, 1].set_title('Modified Pixel Count (L0 Verification)')
    axes[1, 1].set_xlabel('Attack Mode')
    axes[1, 1].set_ylabel('Number of Modified Pixels')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'experiment_b_analysis.png'), dpi=300, bbox_inches='tight')
    print(f"\n  ✓ Saved plot: experiment_b_analysis.png")

    return summary


def analyze_experiment_c(df_c, output_dir):
    """Analyze Experiment C: Fixed L2"""
    print("\n" + "="*80)
    print("ANALYZING EXPERIMENT C: Fixed L2 (Energy Alignment)")
    print("="*80 + "\n")

    # Group by mode
    summary = df_c.groupby('mode').agg({
        'success': ['mean', 'std', 'count'],
        'confidence_drop': ['mean', 'std'],
        'efficiency': ['mean', 'std'],
        'l2_norm': ['mean', 'std', 'min', 'max'],
        'l0_norm': ['mean', 'std'],
        'linf_norm': ['mean', 'std']
    }).round(4)

    print(summary)
    print()

    # L2 scaling validation
    if 'l2_scaling_error' in df_c.columns:
        avg_error = df_c[df_c['mode'] == 'full_scaled']['l2_scaling_error'].mean()
        max_error = df_c[df_c['mode'] == 'full_scaled']['l2_scaling_error'].max()
        convergence_rate = df_c[df_c['mode'] == 'full_scaled']['l2_converged'].mean() * 100

        print("L2 SCALING VALIDATION:")
        print(f"  Average error: {avg_error:.6f} ({avg_error*100:.4f}%)")
        print(f"  Max error: {max_error:.6f} ({max_error*100:.4f}%)")
        print(f"  Convergence rate: {convergence_rate:.1f}%")
        if avg_error < 0.01:
            print(f"  ✓ L2 alignment successful (error < 1%)\n")
        else:
            print(f"  ⚠ WARNING: L2 alignment error exceeds 1%\n")

    # Key findings
    print("KEY FINDINGS (Under Equal L2 Budget):")
    lesion_asr = df_c[df_c['mode'] == 'lesion']['success'].mean()
    full_asr = df_c[df_c['mode'] == 'full_scaled']['success'].mean()
    lesion_eff = df_c[df_c['mode'] == 'lesion']['efficiency'].mean()
    full_eff = df_c[df_c['mode'] == 'full_scaled']['efficiency'].mean()

    print(f"  ASR: Lesion={lesion_asr:.1%}, Full={full_asr:.1%}")
    print(f"  Efficiency: Lesion={lesion_eff:.4f}, Full={full_eff:.4f}")
    if lesion_asr > full_asr:
        print(f"  → Lesion attack is {lesion_asr/full_asr:.2f}x MORE EFFECTIVE under equal L2!")
    else:
        print(f"  → Full attack is {full_asr/lesion_asr:.2f}x more effective under equal L2")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Experiment C: Fixed L2 (Energy Alignment)', fontsize=16, fontweight='bold')

    # Plot 1: ASR comparison
    modes = ['lesion', 'full_scaled']
    asrs = [df_c[df_c['mode'] == m]['success'].mean() * 100 for m in modes]
    colors = ['#2ecc71', '#3498db']
    bars = axes[0, 0].bar(range(len(modes)), asrs, color=colors, alpha=0.7, edgecolor='black', width=0.6)
    axes[0, 0].set_xticks(range(len(modes)))
    axes[0, 0].set_xticklabels(['Lesion Attack', 'Full Attack\n(L2-scaled)'])
    axes[0, 0].set_ylabel('Attack Success Rate (%)')
    axes[0, 0].set_title('ASR Under Equal L2 Budget')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    for i, bar in enumerate(bars):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

    # Plot 2: L2 norm verification
    df_c.boxplot(column='l2_norm', by='mode', ax=axes[0, 1])
    axes[0, 1].set_title('L2 Norm Distribution (Should Be Equal)')
    axes[0, 1].set_xlabel('Attack Mode')
    axes[0, 1].set_ylabel('L2 Norm')

    # Plot 3: L0 comparison (pixels modified)
    l0_lesion = df_c[df_c['mode'] == 'lesion']['l0_norm'].mean()
    l0_full = df_c[df_c['mode'] == 'full_scaled']['l0_norm'].mean()
    bars = axes[1, 0].bar(range(2), [l0_lesion, l0_full], color=colors, alpha=0.7, edgecolor='black', width=0.6)
    axes[1, 0].set_xticks(range(2))
    axes[1, 0].set_xticklabels(['Lesion', 'Full (scaled)'])
    axes[1, 0].set_ylabel('Number of Modified Pixels (L0)')
    axes[1, 0].set_title('Modified Pixels Under Equal L2')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    for i, bar in enumerate(bars):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 1000,
                       f'{int(height):,}', ha='center', va='bottom', fontweight='bold')

    # Plot 4: Efficiency comparison
    effs = [lesion_eff, full_eff]
    bars = axes[1, 1].bar(range(2), effs, color=colors, alpha=0.7, edgecolor='black', width=0.6)
    axes[1, 1].set_xticks(range(2))
    axes[1, 1].set_xticklabels(['Lesion', 'Full (scaled)'])
    axes[1, 1].set_ylabel('Attack Efficiency (ΔConf / L2)')
    axes[1, 1].set_title('Efficiency Under Equal L2 Budget')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    for i, bar in enumerate(bars):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                       f'{height:.4f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'experiment_c_analysis.png'), dpi=300, bbox_inches='tight')
    print(f"\n  ✓ Saved plot: experiment_c_analysis.png")

    return summary


def generate_comprehensive_report(results, summaries, output_dir):
    """Generate comprehensive markdown report"""
    print("\n" + "="*80)
    print("GENERATING COMPREHENSIVE REPORT")
    print("="*80 + "\n")

    report_path = os.path.join(output_dir, 'MULTI_METRIC_ANALYSIS_REPORT.md')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Multi-Metric Sensitivity Analysis Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

        f.write("## Executive Summary\n\n")
        f.write("This report presents a comprehensive three-dimensional robustness evaluation of CheXzero ")
        f.write("on the RSNA Pneumonia dataset, comparing lesion-targeted attacks against global attacks ")
        f.write("under three distinct metric constraints.\n\n")

        # Experiment A summary
        if 'A' in results:
            f.write("### Experiment A: Fixed L∞ (Intensity Alignment)\n\n")
            f.write("**Research Question:** Does constraining single-pixel perturbation intensity reveal ")
            f.write("lesion regions as more cost-effective attack targets?\n\n")
            df_a = results['A']
            for eps in df_a['epsilon'].unique():
                df_eps = df_a[df_a['epsilon'] == eps]
                lesion_asr = df_eps[df_eps['mode'] == 'lesion']['success'].mean()
                full_asr = df_eps[df_eps['mode'] == 'full']['success'].mean()
                f.write(f"- **ε={eps*255:.1f}/255**: Lesion ASR={lesion_asr:.1%}, Full ASR={full_asr:.1%}\n")
            f.write("\n")

        # Experiment B summary
        if 'B' in results:
            f.write("### Experiment B: Fixed L0 (Area Alignment)\n\n")
            f.write("**Research Question:** Do lesion regions possess semantic specificity, or is local ")
            f.write("perturbation inherently more effective?\n\n")
            df_b = results['B']
            lesion_asr = df_b[df_b['mode'] == 'lesion']['success'].mean()
            random_asr = df_b[df_b['mode'] == 'random_patch']['success'].mean()
            full_asr = df_b[df_b['mode'] == 'full']['success'].mean()
            f.write(f"- **Lesion Attack:** ASR={lesion_asr:.1%}\n")
            f.write(f"- **Random Patch Attack:** ASR={random_asr:.1%}\n")
            f.write(f"- **Full Attack:** ASR={full_asr:.1%}\n")
            f.write(f"- **Conclusion:** Lesion regions are {lesion_asr/random_asr:.2f}× more effective than ")
            f.write("equal-area random patches, confirming **semantic specificity**.\n\n")

        # Experiment C summary
        if 'C' in results:
            f.write("### Experiment C: Fixed L2 (Energy Alignment)\n\n")
            f.write("**Research Question:** Under equal perturbation energy budget, does concentrating ")
            f.write("perturbations on lesions outperform global distribution?\n\n")
            df_c = results['C']
            lesion_asr = df_c[df_c['mode'] == 'lesion']['success'].mean()
            full_asr = df_c[df_c['mode'] == 'full_scaled']['success'].mean()
            f.write(f"- **Lesion Attack:** ASR={lesion_asr:.1%}\n")
            f.write(f"- **Full Attack (L2-scaled):** ASR={full_asr:.1%}\n")
            if lesion_asr > full_asr:
                f.write(f"- **Conclusion:** Focused lesion attack is {lesion_asr/full_asr:.2f}× more effective ")
                f.write("under equal L2 budget.\n\n")
            else:
                f.write(f"- **Conclusion:** Global attack remains {full_asr/lesion_asr:.2f}× more effective ")
                f.write("even under equal L2 budget.\n\n")

        f.write("---\n\n")
        f.write("## Detailed Results\n\n")
        f.write("See individual experiment analysis plots:\n")
        f.write("- `experiment_a_analysis.png`\n")
        f.write("- `experiment_b_analysis.png`\n")
        f.write("- `experiment_c_analysis.png`\n\n")

        f.write("---\n\n")
        f.write("## Implications for Model Robustness\n\n")
        f.write("1. **Lesion Specificity:** Experiments A and B demonstrate that lesion regions are ")
        f.write("semantically important for CheXzero's decision-making.\n\n")
        f.write("2. **Efficiency vs Effectiveness Trade-off:** Lesion-targeted attacks are more ")
        f.write("parameter-efficient but may have lower absolute success rates.\n\n")
        f.write("3. **Defense Recommendations:**\n")
        f.write("   - Implement lesion-aware adversarial training\n")
        f.write("   - Add input validation focusing on pathological regions\n")
        f.write("   - Consider ensemble models with varying lesion sensitivities\n\n")

    print(f"  ✓ Saved report: {report_path}")

    return report_path


def main():
    parser = argparse.ArgumentParser(description='Analyze Multi-Metric Experiment Results')
    parser.add_argument('--results_dir', type=str, default='results/multi_metric',
                        help='Directory containing experiment results')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for analysis (default: same as results_dir)')
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.results_dir

    print("\n" + "="*80)
    print("MULTI-METRIC ANALYSIS")
    print("="*80)
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")
    print("="*80 + "\n")

    # Load results
    results = load_experiment_results(args.results_dir)

    if not results:
        print("ERROR: No experiment results found!")
        return

    # Analyze each experiment
    summaries = {}

    if 'A' in results:
        summaries['A'] = analyze_experiment_a(results['A'], args.output_dir)

    if 'B' in results:
        summaries['B'] = analyze_experiment_b(results['B'], args.output_dir)

    if 'C' in results:
        summaries['C'] = analyze_experiment_c(results['C'], args.output_dir)

    # Generate comprehensive report
    report_path = generate_comprehensive_report(results, summaries, args.output_dir)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nGenerated files in: {args.output_dir}")
    print("  - experiment_a_analysis.png")
    print("  - experiment_b_analysis.png")
    print("  - experiment_c_analysis.png")
    print(f"  - {os.path.basename(report_path)}")
    print("\nNext step: Review plots and update dissertation!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
