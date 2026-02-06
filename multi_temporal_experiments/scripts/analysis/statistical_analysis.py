#!/usr/bin/env python3
"""
Statistical analysis for temporal sampling ablation experiments.

Computes paired t-tests and effect sizes (Cohen's dz) for comparing
temporal sampling conditions using 5-fold cross-validation results.

Formulas used:
--------------
1. Paired t-test:
   - H0: mean difference = 0
   - t = mean(differences) / (std(differences) / sqrt(n))
   - df = n - 1 = 4 (for 5 folds)
   - Two-sided p-value

2. Cohen's dz (effect size for paired designs):
   - dz = mean(differences) / std(differences)
   - Note: This is different from Cohen's d for independent samples
   - Interpretation (Cohen, 1988):
     * < 0.2: negligible
     * 0.2 - 0.5: small
     * 0.5 - 0.8: medium
     * 0.8 - 1.2: large
     * > 1.2: very large

Usage:
    python statistical_analysis.py
    python statistical_analysis.py --results-dir /path/to/experiments
"""

import argparse
import json
import numpy as np
from scipy import stats
from pathlib import Path


def load_results_from_history(exp_dir: Path) -> dict:
    """
    Load best IoU from each fold's history.json.

    Args:
        exp_dir: Base experiment directory

    Returns:
        dict mapping experiment name to list of per-fold IoUs
    """
    results = {}

    experiments = [
        ("exp001_v2", "annual"),
        ("exp002_v2", "biseasonal"),
        ("exp003_v2", "bitemporal"),
    ]

    for exp_name, condition in experiments:
        ious = []
        for fold in range(5):
            hist_path = exp_dir / f"{exp_name}_fold{fold}" / "history.json"
            if hist_path.exists():
                with open(hist_path) as f:
                    h = json.load(f)
                # val is a list of dicts, one per epoch
                val_ious = [epoch['iou'] for epoch in h['val']]
                best_iou = max(val_ious) * 100  # Convert to percentage
                ious.append(best_iou)
            else:
                print(f"WARNING: {hist_path} not found")

        if len(ious) == 5:
            results[condition] = np.array(ious)

    return results


def interpret_effect_size(dz: float) -> str:
    """
    Interpret Cohen's dz effect size.

    Thresholds based on Cohen (1988) conventions,
    adapted for paired designs.
    """
    d = abs(dz)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    elif d < 1.2:
        return "large"
    else:
        return "very large"


def paired_ttest_with_effect_size(x: np.ndarray, y: np.ndarray,
                                   name_x: str, name_y: str) -> dict:
    """
    Perform paired t-test and compute Cohen's dz.

    Args:
        x, y: Arrays of paired observations (e.g., per-fold IoUs)
        name_x, name_y: Names of conditions

    Returns:
        dict with statistical results
    """
    # Compute differences
    differences = x - y
    n = len(differences)

    # Descriptive statistics of differences
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)  # Sample std (ddof=1)
    se_diff = std_diff / np.sqrt(n)

    # Paired t-test (two-sided)
    # Formula: t = mean(diff) / SE(diff)
    t_statistic = mean_diff / se_diff
    df = n - 1
    p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), df))

    # Verify with scipy
    t_scipy, p_scipy = stats.ttest_rel(x, y)

    # Cohen's dz for paired designs
    # Formula: dz = mean(diff) / std(diff)
    cohens_dz = mean_diff / std_diff

    return {
        'comparison': f"{name_x} vs {name_y}",
        'n_pairs': n,
        'differences': differences,
        'mean_diff': mean_diff,
        'std_diff': std_diff,
        'se_diff': se_diff,
        't_statistic': t_statistic,
        't_scipy': t_scipy,  # For verification
        'df': df,
        'p_value': p_value,
        'p_scipy': p_scipy,  # For verification
        'cohens_dz': cohens_dz,
        'effect_interpretation': interpret_effect_size(cohens_dz),
    }


def print_results(results: dict):
    """Pretty print statistical results."""
    print(f"\n{'='*70}")
    print(f"Comparison: {results['comparison']}")
    print('='*70)

    print(f"\n1. Per-fold differences ({results['comparison'].split(' vs ')[0]} - {results['comparison'].split(' vs ')[1]}):")
    for i, d in enumerate(results['differences']):
        print(f"   Fold {i}: {d:+.2f} pp")

    print(f"\n2. Descriptive statistics of differences:")
    print(f"   Mean difference: {results['mean_diff']:.2f} pp")
    print(f"   Std of differences: {results['std_diff']:.2f} pp")
    print(f"   SE of mean difference: {results['se_diff']:.2f} pp")

    print(f"\n3. Paired t-test (two-sided, df={results['df']}):")
    print(f"   t-statistic: {results['t_statistic']:.2f}")
    print(f"   p-value: {results['p_value']:.4f}")
    print(f"   (scipy verification: t={results['t_scipy']:.2f}, p={results['p_scipy']:.4f})")

    sig = "YES" if results['p_value'] < 0.05 else "NO"
    print(f"   Significant at α=0.05? {sig}")

    print(f"\n4. Effect size (Cohen's dz for paired designs):")
    print(f"   dz = mean(diff) / std(diff) = {results['mean_diff']:.2f} / {results['std_diff']:.2f} = {results['cohens_dz']:.2f}")
    print(f"   Interpretation: {results['effect_interpretation']}")


def generate_latex_table(all_results: list) -> str:
    """Generate LaTeX table for statistical results."""
    latex = r"""
\begin{table}[h]
\centering
\caption{Statistical comparison of temporal sampling conditions (paired t-test, two-sided, $df=4$).}
\label{tab:stats}
\begin{tabular}{lcccc}
\toprule
\textbf{Comparison} & \textbf{Mean $\Delta$} & \textbf{$t$-statistic} & \textbf{$p$-value} & \textbf{Cohen's $d_z$} \\
\midrule
"""
    for r in all_results:
        latex += f"Annual vs {r['comparison'].split(' vs ')[1].capitalize()} & "
        latex += f"${r['mean_diff']:+.2f}$ pp & "
        latex += f"{r['t_statistic']:.2f} & "
        latex += f"{r['p_value']:.3f} & "
        latex += f"{r['cohens_dz']:.2f} ({r['effect_interpretation']}) \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return latex


def main():
    parser = argparse.ArgumentParser(description="Statistical analysis for temporal ablation")
    parser.add_argument('--results-dir', type=str,
                        default='/cluster/home/tmstorma/NINA_fordypningsoppgave/multi_temporal_experiments/outputs/experiments',
                        help='Directory containing experiment results')
    parser.add_argument('--manual', action='store_true',
                        help='Use manually specified results instead of loading from files')
    args = parser.parse_args()

    print("="*70)
    print("STATISTICAL ANALYSIS: Temporal Sampling Ablation")
    print("="*70)

    if args.manual:
        # Manual results (v2)
        print("\nUsing manually specified v2 results:")
        results = {
            'annual': np.array([52.46, 62.55, 69.88, 58.43, 37.14]),
            'bitemporal': np.array([53.99, 61.58, 53.64, 56.67, 38.47]),
            'biseasonal': np.array([41.44, 59.26, 56.06, 49.03, 36.44]),
        }
    else:
        print(f"\nLoading results from: {args.results_dir}")
        results = load_results_from_history(Path(args.results_dir))

    # Print loaded results
    print("\nLoaded per-fold IoU (%):")
    print(f"{'Fold':<6} {'Annual (T=7)':<15} {'Bi-temporal (T=2)':<18} {'Bi-seasonal (T=14)':<18}")
    print("-" * 60)
    for fold in range(5):
        print(f"{fold:<6} {results['annual'][fold]:<15.2f} {results['bitemporal'][fold]:<18.2f} {results['biseasonal'][fold]:<18.2f}")
    print("-" * 60)
    print(f"{'Mean':<6} {np.mean(results['annual']):<15.2f} {np.mean(results['bitemporal']):<18.2f} {np.mean(results['biseasonal']):<18.2f}")
    print(f"{'Std':<6} {np.std(results['annual'], ddof=1):<15.2f} {np.std(results['bitemporal'], ddof=1):<18.2f} {np.std(results['biseasonal'], ddof=1):<18.2f}")

    # Perform statistical comparisons
    all_stats = []

    # Annual vs Bi-temporal
    stats1 = paired_ttest_with_effect_size(
        results['annual'], results['bitemporal'],
        'Annual', 'Bi-temporal'
    )
    print_results(stats1)
    all_stats.append(stats1)

    # Annual vs Bi-seasonal
    stats2 = paired_ttest_with_effect_size(
        results['annual'], results['biseasonal'],
        'Annual', 'Bi-seasonal'
    )
    print_results(stats2)
    all_stats.append(stats2)

    # Generate LaTeX table
    print("\n" + "="*70)
    print("LaTeX Table:")
    print("="*70)
    print(generate_latex_table(all_stats))

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"""
Key findings:
1. Annual vs Bi-temporal: Δ = {stats1['mean_diff']:+.2f} pp, p = {stats1['p_value']:.3f}, dz = {stats1['cohens_dz']:.2f} ({stats1['effect_interpretation']})
   → {'Statistically significant' if stats1['p_value'] < 0.05 else 'NOT statistically significant'} at α=0.05

2. Annual vs Bi-seasonal: Δ = {stats2['mean_diff']:+.2f} pp, p = {stats2['p_value']:.3f}, dz = {stats2['cohens_dz']:.2f} ({stats2['effect_interpretation']})
   → {'Statistically significant' if stats2['p_value'] < 0.05 else 'NOT statistically significant'} at α=0.05

Formulas used:
- Paired t-test: t = mean(diff) / (std(diff) / sqrt(n)), df = n-1
- Cohen's dz: dz = mean(diff) / std(diff)
- std uses ddof=1 (sample standard deviation)
""")


if __name__ == "__main__":
    main()
