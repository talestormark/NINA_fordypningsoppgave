#!/usr/bin/env python3
"""
Statistical analysis for Part II spectral/spatial resolution experiments.

Performs:
1. Paired permutation tests (each condition vs block anchor)
2. Holm-Bonferroni correction for multiple comparisons
3. Effect sizes: mean delta-IoU with 95% bootstrap CI
4. Cohen's dz for paired designs

Usage:
    python statistical_tests.py
    python statistical_tests.py --experiments-dir /path/to/experiments
    python statistical_tests.py --metric iou  # default
    python statistical_tests.py --metric f1
"""

import argparse
import json
import numpy as np
from scipy import stats
from pathlib import Path
from itertools import combinations

PART2_DIR = Path(__file__).resolve().parent.parent.parent
EXPERIMENTS_DIR = PART2_DIR / "outputs" / "experiments"
OUTPUT_DIR = PART2_DIR / "outputs" / "analysis"

NUM_FOLDS = 5
N_PERMUTATIONS = 10000

# Block anchors (condition each is compared against)
BLOCK_ANCHORS = {
    "A": "A3_s2_9band",
    "B": "A3_s2_9band",
    "C": "A1_s2_rgb",
    "D": "A3_s2_9band",
}

# Which experiments belong to which block
BLOCK_EXPERIMENTS = {
    "A": ["A1_s2_rgb", "A2_s2_rgbnir", "A3_s2_9band", "A4_s2_indices",
          "A5_indices_only", "A6_temporal_diff"],
    "B": ["A3_s2_9band", "B2_s2_bandgroup"],
    "C": ["A1_s2_rgb", "C2_ps_rgb", "C2hm_ps_rgb_hm", "C3_s2_ps_fusion"],
    "D": ["A3_s2_9band", "D2_alphaearth", "D3_s2_ae_fusion"],
}


def load_fold_metrics(experiments_dir: Path, exp_name: str, metric: str = 'iou') -> np.ndarray:
    """Load per-fold best validation metric from history.json files."""
    values = []
    for fold in range(NUM_FOLDS):
        hist_path = experiments_dir / f"{exp_name}_fold{fold}" / "history.json"
        if not hist_path.exists():
            return np.array([])

        with open(hist_path) as f:
            history = json.load(f)

        val_epochs = history.get('val', [])
        if not val_epochs:
            return np.array([])

        best_val = max(epoch.get(metric, 0.0) for epoch in val_epochs)
        values.append(best_val * 100)  # Convert to percentage

    return np.array(values)


def load_test_fold_metrics(experiments_dir: Path, exp_name: str, metric: str = 'iou') -> np.ndarray:
    """Load per-fold test metrics from test_results.json."""
    test_path = experiments_dir / exp_name / "test_results.json"
    if not test_path.exists():
        return np.array([])

    with open(test_path) as f:
        data = json.load(f)

    values = []
    for fold_str in sorted(data.get('folds', {}).keys()):
        agg = data['folds'][fold_str].get('aggregated', {})
        if metric in agg:
            values.append(agg[metric] * 100)

    return np.array(values)


def paired_permutation_test(x: np.ndarray, y: np.ndarray, n_permutations: int = 10000,
                             seed: int = 42) -> float:
    """
    Two-sided paired permutation test.

    H0: The mean of x - y is zero.
    Test statistic: |mean(x - y)|

    Returns p-value.
    """
    rng = np.random.RandomState(seed)
    diff = x - y
    observed = abs(np.mean(diff))

    n = len(diff)
    count = 0
    for _ in range(n_permutations):
        signs = rng.choice([-1, 1], size=n)
        perm_stat = abs(np.mean(diff * signs))
        if perm_stat >= observed:
            count += 1

    return (count + 1) / (n_permutations + 1)  # +1 for continuity correction


def bootstrap_ci(x: np.ndarray, y: np.ndarray, n_bootstrap: int = 10000,
                  ci: float = 0.95, seed: int = 42) -> tuple:
    """
    Bootstrap confidence interval for mean(x - y).

    Returns (mean_diff, ci_low, ci_high).
    """
    rng = np.random.RandomState(seed)
    diff = x - y
    n = len(diff)

    boot_means = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_means[i] = np.mean(diff[idx])

    alpha = 1 - ci
    ci_low = np.percentile(boot_means, alpha / 2 * 100)
    ci_high = np.percentile(boot_means, (1 - alpha / 2) * 100)

    return float(np.mean(diff)), float(ci_low), float(ci_high)


def cohens_dz(x: np.ndarray, y: np.ndarray) -> float:
    """Cohen's dz for paired designs: mean(diff) / std(diff)."""
    diff = x - y
    std = np.std(diff, ddof=1)
    if std == 0:
        return 0.0
    return float(np.mean(diff) / std)


def interpret_effect_size(dz: float) -> str:
    """Interpret Cohen's dz magnitude."""
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


def holm_bonferroni(p_values: list) -> list:
    """
    Apply Holm-Bonferroni correction to a list of (comparison_name, p_value) tuples.

    Returns list of (comparison_name, raw_p, adjusted_p, significant) tuples.
    """
    n = len(p_values)
    # Sort by raw p-value
    sorted_pvals = sorted(p_values, key=lambda x: x[1])

    results = []
    for rank, (name, raw_p) in enumerate(sorted_pvals):
        adjusted_p = min(raw_p * (n - rank), 1.0)
        results.append((name, raw_p, adjusted_p))

    # Enforce monotonicity (adjusted p should be non-decreasing)
    for i in range(1, len(results)):
        if results[i][2] < results[i-1][2]:
            results[i] = (results[i][0], results[i][1], results[i-1][2])

    # Determine significance
    final = []
    for name, raw_p, adj_p in results:
        final.append((name, raw_p, adj_p, adj_p < 0.05))

    return final


def main():
    parser = argparse.ArgumentParser(description="Statistical tests for Part II experiments")
    parser.add_argument('--experiments-dir', type=str, default=str(EXPERIMENTS_DIR))
    parser.add_argument('--metric', type=str, default='iou',
                        choices=['iou', 'f1', 'precision', 'recall'])
    parser.add_argument('--use-test', action='store_true',
                        help='Use test set metrics instead of validation')
    parser.add_argument('--output-dir', type=str, default=str(OUTPUT_DIR))
    parser.add_argument('--n-permutations', type=int, default=N_PERMUTATIONS)
    args = parser.parse_args()

    experiments_dir = Path(args.experiments_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*70}")
    print(f"Statistical Analysis: Part II (metric={args.metric})")
    print(f"{'='*70}")

    # Load all available experiment metrics
    all_metrics = {}
    for block, experiments in BLOCK_EXPERIMENTS.items():
        for exp_name in experiments:
            if exp_name in all_metrics:
                continue
            if args.use_test:
                values = load_test_fold_metrics(experiments_dir, exp_name, args.metric)
            else:
                values = load_fold_metrics(experiments_dir, exp_name, args.metric)

            if len(values) == NUM_FOLDS:
                all_metrics[exp_name] = values
                print(f"  Loaded {exp_name}: {values} (mean={np.mean(values):.2f}%)")
            elif len(values) > 0:
                print(f"  WARNING: {exp_name} has {len(values)}/{NUM_FOLDS} folds, skipping")
            else:
                print(f"  {exp_name}: not found")

    if not all_metrics:
        print("\nNo experiments with complete fold data. Exiting.")
        return

    # Perform block-wise comparisons
    all_comparisons = []
    all_p_values = []

    for block, experiments in BLOCK_EXPERIMENTS.items():
        anchor = BLOCK_ANCHORS[block]
        if anchor not in all_metrics:
            print(f"\nSkipping Block {block}: anchor {anchor} not available")
            continue

        anchor_values = all_metrics[anchor]
        available = [e for e in experiments if e in all_metrics and e != anchor]

        if not available:
            continue

        print(f"\n{'='*70}")
        print(f"BLOCK {block} (anchor: {anchor})")
        print(f"{'='*70}")

        for exp_name in available:
            exp_values = all_metrics[exp_name]

            # Paired permutation test
            p_value = paired_permutation_test(
                exp_values, anchor_values,
                n_permutations=args.n_permutations
            )

            # Bootstrap CI for mean difference
            mean_diff, ci_low, ci_high = bootstrap_ci(exp_values, anchor_values)

            # Cohen's dz
            dz = cohens_dz(exp_values, anchor_values)

            # Also compute paired t-test for reference
            t_stat, t_p = stats.ttest_rel(exp_values, anchor_values)

            comparison = {
                'block': block,
                'experiment': exp_name,
                'anchor': anchor,
                'metric': args.metric,
                'n_folds': NUM_FOLDS,
                'exp_mean': float(np.mean(exp_values)),
                'exp_std': float(np.std(exp_values, ddof=1)),
                'anchor_mean': float(np.mean(anchor_values)),
                'anchor_std': float(np.std(anchor_values, ddof=1)),
                'mean_diff': mean_diff,
                'ci_95_low': ci_low,
                'ci_95_high': ci_high,
                'perm_p_value': p_value,
                'ttest_p_value': float(t_p),
                'cohens_dz': dz,
                'effect_size': interpret_effect_size(dz),
                'per_fold_diff': (exp_values - anchor_values).tolist(),
            }
            all_comparisons.append(comparison)
            all_p_values.append((f"{exp_name} vs {anchor}", p_value))

            # Print
            print(f"\n  {exp_name} vs {anchor}:")
            print(f"    {exp_name}: {np.mean(exp_values):.2f}% +/- {np.std(exp_values, ddof=1):.2f}%")
            print(f"    {anchor}: {np.mean(anchor_values):.2f}% +/- {np.std(anchor_values, ddof=1):.2f}%")
            print(f"    Mean diff: {mean_diff:+.2f} pp  [{ci_low:+.2f}, {ci_high:+.2f}]")
            print(f"    Per-fold:  {['%+.2f' % d for d in (exp_values - anchor_values)]}")
            print(f"    Permutation p = {p_value:.4f}  (t-test p = {t_p:.4f})")
            print(f"    Cohen's dz = {dz:.2f} ({interpret_effect_size(dz)})")

    # Holm-Bonferroni correction
    if all_p_values:
        print(f"\n{'='*70}")
        print("HOLM-BONFERRONI CORRECTION")
        print(f"{'='*70}")

        corrected = holm_bonferroni(all_p_values)
        print(f"\n{'Comparison':<40} {'Raw p':<10} {'Adj p':<10} {'Sig':<5}")
        print('-' * 65)
        for name, raw_p, adj_p, sig in corrected:
            sig_str = "***" if adj_p < 0.001 else "**" if adj_p < 0.01 else "*" if adj_p < 0.05 else "ns"
            print(f"{name:<40} {raw_p:<10.4f} {adj_p:<10.4f} {sig_str}")

        # Update comparisons with corrected p-values
        correction_map = {name: (adj_p, sig) for name, _, adj_p, sig in corrected}
        for comp in all_comparisons:
            key = f"{comp['experiment']} vs {comp['anchor']}"
            if key in correction_map:
                comp['adjusted_p_value'] = correction_map[key][0]
                comp['significant_corrected'] = correction_map[key][1]

    # Save results
    results = {
        'metric': args.metric,
        'source': 'test' if args.use_test else 'validation',
        'n_permutations': args.n_permutations,
        'comparisons': all_comparisons,
    }

    json_path = output_dir / f"statistical_tests_{args.metric}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {json_path}")

    # Save CSV summary
    csv_lines = [
        "block,experiment,anchor,metric,exp_mean,exp_std,anchor_mean,anchor_std,"
        "mean_diff,ci_low,ci_high,perm_p,adj_p,cohens_dz,effect_size,significant"
    ]
    for c in all_comparisons:
        csv_lines.append(
            f"{c['block']},{c['experiment']},{c['anchor']},{c['metric']},"
            f"{c['exp_mean']:.4f},{c['exp_std']:.4f},"
            f"{c['anchor_mean']:.4f},{c['anchor_std']:.4f},"
            f"{c['mean_diff']:.4f},{c['ci_95_low']:.4f},{c['ci_95_high']:.4f},"
            f"{c['perm_p_value']:.4f},{c.get('adjusted_p_value', ''):.4f},"
            f"{c['cohens_dz']:.4f},{c['effect_size']},"
            f"{c.get('significant_corrected', '')}"
        )

    csv_path = output_dir / f"statistical_tests_{args.metric}.csv"
    with open(csv_path, 'w') as f:
        f.write('\n'.join(csv_lines) + '\n')
    print(f"CSV saved to: {csv_path}")

    # Print LaTeX table
    print(f"\n{'='*70}")
    print("LATEX TABLE")
    print(f"{'='*70}")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Statistical comparisons (paired permutation test, $n=5$ folds).}")
    print(r"\label{tab:stats-p2}")
    print(r"\begin{tabular}{llcccc}")
    print(r"\toprule")
    print(r"\textbf{Block} & \textbf{Condition} & \textbf{$\Delta$IoU (pp)} & "
          r"\textbf{95\% CI} & \textbf{$p_{\text{adj}}$} & \textbf{$d_z$} \\")
    print(r"\midrule")

    for c in all_comparisons:
        adj_p = c.get('adjusted_p_value', c['perm_p_value'])
        sig = ""
        if adj_p < 0.001:
            sig = "***"
        elif adj_p < 0.01:
            sig = "**"
        elif adj_p < 0.05:
            sig = "*"

        exp_latex = c['experiment'].replace('_', r'\_')
        row_end = r"\\"
        print(f"{c['block']} & {exp_latex} & "
              f"${c['mean_diff']:+.2f}$ & "
              f"$[{c['ci_95_low']:+.2f}, {c['ci_95_high']:+.2f}]$ & "
              f"{adj_p:.3f}{sig} & "
              f"{c['cohens_dz']:.2f} {row_end}")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()
