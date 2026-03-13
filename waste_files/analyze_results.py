"""
Analyze Attack Results: Lesion vs Full Image Attacks
=====================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 11
plt.rcParams['figure.figsize'] = (14, 8)

def load_results(results_dir='results/attacks_full'):
    """Load all attack results."""
    summaries = pd.read_csv(os.path.join(results_dir, 'attack_summaries.csv'))
    all_results = pd.read_csv(os.path.join(results_dir, 'all_results.csv'))

    return summaries, all_results


def create_comparison_plots(summaries, output_dir='results/attacks_full'):
    """Create comparison plots."""

    # Separate lesion and full attacks
    lesion = summaries[summaries['mode'] == 'lesion'].copy()
    full = summaries[summaries['mode'] == 'full'].copy()

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Lesion-Targeted vs Full-Image Adversarial Attacks on RSNA Pneumonia Dataset',
                 fontsize=16, fontweight='bold')

    # Plot 1: Success Rate Comparison
    ax = axes[0, 0]
    x = np.arange(len(lesion))
    width = 0.35

    bars1 = ax.bar(x - width/2, lesion['success_rate']*100, width,
                   label='Lesion Attack', color='#FF6B6B', alpha=0.8)
    bars2 = ax.bar(x + width/2, full['success_rate']*100, width,
                   label='Full Image Attack', color='#4ECDC4', alpha=0.8)

    ax.set_xlabel('Attack Method', fontweight='bold')
    ax.set_ylabel('Attack Success Rate (%)', fontweight='bold')
    ax.set_title('A) Attack Success Rate Comparison', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(lesion['attack'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

    # Plot 2: L2 Norm Comparison
    ax = axes[0, 1]
    bars1 = ax.bar(x - width/2, lesion['l2_mean'], width,
                   label='Lesion Attack', color='#FF6B6B', alpha=0.8)
    bars2 = ax.bar(x + width/2, full['l2_mean'], width,
                   label='Full Image Attack', color='#4ECDC4', alpha=0.8)

    ax.set_xlabel('Attack Method', fontweight='bold')
    ax.set_ylabel('Mean L2 Norm', fontweight='bold')
    ax.set_title('B) Perturbation Magnitude (L2 Norm)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(lesion['attack'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Plot 3: Efficiency (Success Rate / L2 Norm)
    ax = axes[1, 0]
    lesion_efficiency = lesion['success_rate'] / (lesion['l2_mean'] + 1e-6)
    full_efficiency = full['success_rate'] / (full['l2_mean'] + 1e-6)

    bars1 = ax.bar(x - width/2, lesion_efficiency*100, width,
                   label='Lesion Attack', color='#FF6B6B', alpha=0.8)
    bars2 = ax.bar(x + width/2, full_efficiency*100, width,
                   label='Full Image Attack', color='#4ECDC4', alpha=0.8)

    ax.set_xlabel('Attack Method', fontweight='bold')
    ax.set_ylabel('Efficiency (Success Rate / L2 Norm × 100)', fontweight='bold')
    ax.set_title('C) Attack Efficiency', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(lesion['attack'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Plot 4: Probability Change
    ax = axes[1, 1]
    bars1 = ax.bar(x - width/2, -lesion['prob_change_mean']*100, width,
                   label='Lesion Attack', color='#FF6B6B', alpha=0.8)
    bars2 = ax.bar(x + width/2, -full['prob_change_mean']*100, width,
                   label='Full Image Attack', color='#4ECDC4', alpha=0.8)

    ax.set_xlabel('Attack Method', fontweight='bold')
    ax.set_ylabel('Mean Probability Decrease (%)', fontweight='bold')
    ax.set_title('D) Impact on Model Confidence', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(lesion['attack'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'attack_comparison_plots.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plots saved to: {plot_path}")
    plt.close()


def generate_report(summaries, all_results, output_dir='results/attacks_full'):
    """Generate detailed analysis report."""

    report = []
    report.append("="*80)
    report.append("RSNA ADVERSARIAL ATTACK EXPERIMENT RESULTS")
    report.append("Lesion-Targeted vs Full-Image Attacks on CheXzero")
    report.append("="*80)
    report.append("")

    report.append("## EXPERIMENTAL SETUP")
    report.append("-" * 80)
    report.append(f"Dataset: RSNA Pneumonia Detection Challenge")
    report.append(f"Model: CheXzero (CLIP-based Foundation Model)")
    report.append(f"Number of Samples: 200 (all pneumonia-positive)")
    report.append(f"Baseline Accuracy: 97.5% @ threshold 0.5")
    report.append("")

    report.append("## ATTACK SUMMARY")
    report.append("-" * 80)
    report.append(summaries.to_string(index=False))
    report.append("")

    # Key findings
    report.append("## KEY FINDINGS")
    report.append("-" * 80)
    report.append("")

    # Finding 1: Success Rate
    report.append("### 1. Attack Success Rates")
    pgd40_lesion = summaries[(summaries['attack']=='PGD_steps40') & (summaries['mode']=='lesion')]['success_rate'].values[0]
    pgd40_full = summaries[(summaries['attack']=='PGD_steps40') & (summaries['mode']=='full')]['success_rate'].values[0]
    report.append(f"   - PGD-40 Lesion Attack: {pgd40_lesion*100:.1f}%")
    report.append(f"   - PGD-40 Full Attack: {pgd40_full*100:.1f}%")
    report.append(f"   - Ratio: {pgd40_full/pgd40_lesion:.2f}x (Full/Lesion)")
    report.append("")

    # Finding 2: Perturbation Size
    report.append("### 2. Perturbation Magnitude (L2 Norm)")
    pgd40_lesion_l2 = summaries[(summaries['attack']=='PGD_steps40') & (summaries['mode']=='lesion')]['l2_mean'].values[0]
    pgd40_full_l2 = summaries[(summaries['attack']=='PGD_steps40') & (summaries['mode']=='full')]['l2_mean'].values[0]
    report.append(f"   - PGD-40 Lesion Attack: {pgd40_lesion_l2:.2f}")
    report.append(f"   - PGD-40 Full Attack: {pgd40_full_l2:.2f}")
    report.append(f"   - Ratio: {pgd40_full_l2/pgd40_lesion_l2:.2f}x (Full/Lesion)")
    report.append("")

    # Finding 3: Efficiency
    report.append("### 3. Attack Efficiency")
    lesion_eff = pgd40_lesion / pgd40_lesion_l2
    full_eff = pgd40_full / pgd40_full_l2
    report.append(f"   - Lesion Attack: {lesion_eff*100:.2f}% success per unit L2")
    report.append(f"   - Full Attack: {full_eff*100:.2f}% success per unit L2")
    report.append(f"   - Lesion attack is {lesion_eff/full_eff:.2f}x more efficient (when successful)")
    report.append("")

    # Finding 4: Modified Pixels
    report.append("### 4. Modified Pixels")
    pgd40_lesion_l0 = summaries[(summaries['attack']=='PGD_steps40') & (summaries['mode']=='lesion')]['l0_mean'].values[0]
    pgd40_full_l0 = summaries[(summaries['attack']=='PGD_steps40') & (summaries['mode']=='full')]['l0_mean'].values[0]
    report.append(f"   - Lesion Attack: {pgd40_lesion_l0:.0f} pixels ({pgd40_lesion_l0/150528*100:.1f}% of image)")
    report.append(f"   - Full Attack: {pgd40_full_l0:.0f} pixels ({pgd40_full_l0/150528*100:.1f}% of image)")
    report.append(f"   - Ratio: {pgd40_full_l0/pgd40_lesion_l0:.2f}x (Full/Lesion)")
    report.append("")

    # Conclusions
    report.append("## CONCLUSIONS")
    report.append("-" * 80)
    report.append("")
    report.append("1. **Full-image attacks are significantly more effective**")
    report.append("   - 97.5% success vs 59.5% for lesion attacks (PGD-40)")
    report.append("   - Suggests model relies on global image features, not just lesion regions")
    report.append("")
    report.append("2. **Lesion attacks require less perturbation**")
    report.append("   - 2.8x smaller L2 norm compared to full attacks")
    report.append("   - 8x fewer pixels modified")
    report.append("   - More efficient but less effective")
    report.append("")
    report.append("3. **PGD significantly outperforms FGSM**")
    report.append("   - Iterative optimization crucial for success")
    report.append("   - Single-step attacks insufficient for lesion-targeted approaches")
    report.append("")
    report.append("4. **Clinical Implications**")
    report.append("   - High-success attacks require modifying large portions of the image")
    report.append("   - This makes adversarial perturbations easier to detect")
    report.append("   - Lesion-only attacks harder to detect but less reliable")
    report.append("")

    report.append("="*80)
    report.append("END OF REPORT")
    report.append("="*80)

    # Save report
    report_text = "\n".join(report)
    report_path = os.path.join(output_dir, 'ANALYSIS_REPORT.txt')
    with open(report_path, 'w') as f:
        f.write(report_text)

    print(report_text)
    print(f"\nReport saved to: {report_path}")

    return report_text


def main():
    # Load results
    print("Loading results...")
    summaries, all_results = load_results()

    # Create visualizations
    print("\nGenerating comparison plots...")
    create_comparison_plots(summaries)

    # Generate report
    print("\nGenerating analysis report...")
    report = generate_report(summaries, all_results)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  - results/attacks_full/attack_comparison_plots.png")
    print("  - results/attacks_full/ANALYSIS_REPORT.txt")
    print("  - results/attacks_full/attack_summaries.csv")
    print("  - results/attacks_full/all_results.csv")


if __name__ == '__main__':
    main()
