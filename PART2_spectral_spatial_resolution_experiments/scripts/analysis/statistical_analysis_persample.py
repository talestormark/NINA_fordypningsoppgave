#!/usr/bin/env python3
"""
Per-tile statistical analysis for Part II spectral/spatial resolution experiments.

Uses per-tile IoU from test set evaluation (n=40 tiles) for paired comparisons,
mirroring the approach from Part I's statistical_analysis_persample.py.

For U-Net experiments: uses ensemble per-sample IoU from test_results.json
    (averaged probability across 5 fold models, then IoU computed).
For RF experiments: averages per-tile test IoU across 5 folds.

Statistical tests:
- Wilcoxon signed-rank test (paired, nonparametric)
- Permutation test (10,000 sign-flip permutations)
- Bootstrap 95% CI for mean difference
- Cliff's delta effect size

Multiple comparison correction: Holm-Bonferroni (applied within each family)

Usage:
    python statistical_analysis_persample.py
    python statistical_analysis_persample.py --n-permutations 50000
    python statistical_analysis_persample.py --metric f1
"""

from __future__ import annotations

import argparse
import json
import numpy as np
from pathlib import Path
from scipy import stats
import warnings
from typing import Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]
PART2_DIR = REPO_ROOT / "PART2_spectral_spatial_resolution_experiments"
AE_DIR = REPO_ROOT / "experiments" / "annotation_efficiency"

UNET_EXPERIMENTS_DIR = PART2_DIR / "outputs" / "experiments"
RF_OUTPUTS_DIR = AE_DIR / "outputs"
OUTPUT_DIR = PART2_DIR / "outputs" / "analysis"

NUM_FOLDS = 5
N_PERMUTATIONS = 10000
N_BOOTSTRAP = 10000
SEED = 42
ALPHA = 0.05


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_unet_pertile_iou(
    exp_name: str,
    metric: str = "iou",
    experiments_dir: Path = UNET_EXPERIMENTS_DIR,
) -> dict | None:
    """
    Load per-tile IoU from U-Net test_results.json.

    Prefers ensemble per_sample (probability-averaged across folds).
    Falls back to averaging per-fold per_sample IoU.

    Returns:
        dict mapping tile_id -> IoU value, or None if data missing.
    """
    test_path = experiments_dir / exp_name / "test_results.json"
    if not test_path.exists():
        return None

    with open(test_path) as f:
        data = json.load(f)

    # Prefer ensemble predictions (averaged probabilities -> single prediction)
    if "ensemble" in data and "per_sample" in data["ensemble"]:
        per_sample = data["ensemble"]["per_sample"]
        return {tile_id: m[metric] for tile_id, m in per_sample.items()}

    # Fallback: average per-tile metric across folds
    folds = data.get("folds", {})
    if not folds:
        return None

    tile_values = {}  # tile_id -> list of values across folds
    for fold_key, fold_data in folds.items():
        per_sample = fold_data.get("per_sample", {})
        for tile_id, m in per_sample.items():
            tile_values.setdefault(tile_id, []).append(m[metric])

    # Average across folds
    return {
        tile_id: np.mean(values)
        for tile_id, values in tile_values.items()
        if len(values) == len(folds)  # only tiles present in all folds
    }


def load_rf_pertile_iou(
    exp_name: str,
    metric: str = "iou",
    outputs_dir: Path = RF_OUTPUTS_DIR,
) -> dict | None:
    """
    Load per-tile test IoU from RF metrics.json files (one per fold).

    Averages per-tile IoU across folds.

    Returns:
        dict mapping tile_id -> IoU value, or None if data missing.
    """
    tile_values = {}
    n_folds_found = 0

    for fold in range(NUM_FOLDS):
        metrics_path = outputs_dir / exp_name / f"fold{fold}" / "metrics.json"
        if not metrics_path.exists():
            continue

        with open(metrics_path) as f:
            data = json.load(f)

        per_tile = data.get("test_metrics", {}).get("per_tile", {})
        if not per_tile:
            continue

        n_folds_found += 1
        for tile_id, m in per_tile.items():
            tile_values.setdefault(tile_id, []).append(m[metric])

    if n_folds_found == 0:
        return None

    # Average across folds (only tiles present in all found folds)
    return {
        tile_id: np.mean(values)
        for tile_id, values in tile_values.items()
        if len(values) == n_folds_found
    }


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------


def wilcoxon_signed_rank_test(differences: np.ndarray):
    """
    Wilcoxon signed-rank test (two-sided).

    Returns (statistic, p_value).
    """
    nonzero_diff = differences[differences != 0]
    if len(nonzero_diff) < 5:
        warnings.warn(
            f"Only {len(nonzero_diff)} non-zero differences. "
            "Wilcoxon test may be unreliable."
        )
        return np.nan, np.nan

    statistic, p_value = stats.wilcoxon(nonzero_diff, alternative="two-sided")
    return float(statistic), float(p_value)


def permutation_test(
    differences: np.ndarray,
    n_permutations: int = N_PERMUTATIONS,
    seed: int = SEED,
) -> tuple:
    """
    Two-sided sign-flip permutation test.

    Under H0, the sign of each difference is arbitrary.
    Test statistic: |mean(differences)|.

    Returns (observed_mean, p_value).
    """
    rng = np.random.RandomState(seed)
    observed_mean = np.mean(differences)
    n = len(differences)

    count = 0
    for _ in range(n_permutations):
        signs = rng.choice([-1, 1], size=n)
        perm_stat = abs(np.mean(differences * signs))
        if perm_stat >= abs(observed_mean):
            count += 1

    p_value = (count + 1) / (n_permutations + 1)
    return float(observed_mean), float(p_value)


def bootstrap_ci(
    differences: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    ci: float = 0.95,
    seed: int = SEED,
) -> tuple:
    """Bootstrap percentile CI for mean(differences)."""
    rng = np.random.RandomState(seed)
    n = len(differences)

    boot_means = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_means[i] = np.mean(differences[idx])

    alpha = 1 - ci
    lower = np.percentile(boot_means, alpha / 2 * 100)
    upper = np.percentile(boot_means, (1 - alpha / 2) * 100)
    return float(lower), float(upper)


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """
    Cliff's delta for paired data.

    Computes dominance: (# x_i > y_i - # x_i < y_i) / n.

    Interpretation (Romano et al., 2006):
        |d| < 0.147: negligible
        0.147 <= |d| < 0.33: small
        0.33  <= |d| < 0.474: medium
        |d| >= 0.474: large
    """
    differences = x - y
    n_greater = np.sum(differences > 0)
    n_less = np.sum(differences < 0)
    return float((n_greater - n_less) / len(differences))


def interpret_cliffs_delta(delta: float) -> str:
    """Interpret Cliff's delta magnitude."""
    d = abs(delta)
    if d < 0.147:
        return "negligible"
    elif d < 0.33:
        return "small"
    elif d < 0.474:
        return "medium"
    else:
        return "large"


def holm_bonferroni(p_values: list) -> list:
    """
    Holm-Bonferroni correction for a list of p-values.

    Returns adjusted p-values (same order as input).
    """
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])

    adjusted = [0.0] * n
    for rank, (orig_idx, p) in enumerate(indexed):
        adjusted[orig_idx] = min(1.0, p * (n - rank))

    # Enforce monotonicity (adjusted values should be non-decreasing in sorted order)
    for i in range(1, n):
        idx = indexed[i][0]
        prev_idx = indexed[i - 1][0]
        adjusted[idx] = max(adjusted[idx], adjusted[prev_idx])

    return adjusted


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Per-tile statistical analysis for Part II experiments"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="iou",
        choices=["iou", "f1", "precision", "recall"],
    )
    parser.add_argument("--n-permutations", type=int, default=N_PERMUTATIONS)
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument(
        "--unet-dir",
        type=str,
        default=str(UNET_EXPERIMENTS_DIR),
        help="Directory containing U-Net experiment outputs",
    )
    parser.add_argument(
        "--rf-dir",
        type=str,
        default=str(RF_OUTPUTS_DIR),
        help="Directory containing RF experiment outputs",
    )
    args = parser.parse_args()

    unet_dir = Path(args.unet_dir)
    rf_dir = Path(args.rf_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metric = args.metric

    print("=" * 70)
    print(f"PER-TILE STATISTICAL ANALYSIS -- Part 2 (metric={metric})")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load per-tile data for every experiment
    # ------------------------------------------------------------------

    # Experiment registry: name -> (loader_fn, *loader_args)
    experiments_spec = {
        # Block A (U-Net)
        "A1_s2_rgb":        ("unet", unet_dir),
        "A2_s2_rgbnir":     ("unet", unet_dir),
        "A3_s2_9band":      ("unet", unet_dir),
        "A4_s2_indices":    ("unet", unet_dir),
        # Block D (U-Net)
        "D2_alphaearth":    ("unet", unet_dir),
        # E4 (U-Net, sparse labels)
        "E4_ae_unet_sparse": ("unet", unet_dir),
        "E4_A3_s2_9band_sparse": ("unet", unet_dir),
        # E1, E2 (RF)
        "E1_ae_rf_sparse":  ("rf", rf_dir),
        "E2_s2_rf_sparse":  ("rf", rf_dir),
    }

    experiments = {}  # name -> dict{tile_id: iou}

    print("\nLoading per-tile test IoU values:")
    for exp_name, (kind, base_dir) in experiments_spec.items():
        if kind == "unet":
            data = load_unet_pertile_iou(exp_name, metric=metric, experiments_dir=base_dir)
        else:
            data = load_rf_pertile_iou(exp_name, metric=metric, outputs_dir=base_dir)

        if data is not None and len(data) > 0:
            experiments[exp_name] = data
            values = np.array(list(data.values()))
            print(
                f"  {exp_name}: {len(data)} tiles, "
                f"mean={np.mean(values)*100:.2f}%, "
                f"std={np.std(values, ddof=1)*100:.2f}%"
            )
        else:
            print(f"  {exp_name}: NOT FOUND")

    if len(experiments) < 2:
        print("\nFewer than 2 experiments loaded. Cannot perform comparisons.")
        return

    # ------------------------------------------------------------------
    # 2. Define comparison families
    # ------------------------------------------------------------------

    comparison_families = {
        "Block A: Spectral ablation (vs A3 anchor)": [
            ("A1 vs A3 (RGB vs 9-band)", "A1_s2_rgb", "A3_s2_9band"),
            ("A2 vs A3 (RGBNIR vs 9-band)", "A2_s2_rgbnir", "A3_s2_9band"),
            ("A4 vs A3 (indices vs 9-band)", "A4_s2_indices", "A3_s2_9band"),
        ],
        "RQ3e: Annotation efficiency": [
            ("E4 vs D2 (sparse vs dense, AE U-Net)", "E4_ae_unet_sparse", "D2_alphaearth"),
            ("E4-S2 vs A3 (sparse vs dense, S2 U-Net)", "E4_A3_s2_9band_sparse", "A3_s2_9band"),
            ("E4 vs A3 (sparse AE vs dense S2)", "E4_ae_unet_sparse", "A3_s2_9band"),
            ("E4-S2 vs E4 (S2 vs AE, both sparse)", "E4_A3_s2_9band_sparse", "E4_ae_unet_sparse"),
            ("D2 vs A3 (AE vs S2, both dense)", "D2_alphaearth", "A3_s2_9band"),
            ("E1 vs A3 (RF sparse vs U-Net dense)", "E1_ae_rf_sparse", "A3_s2_9band"),
        ],
    }

    # ------------------------------------------------------------------
    # 3. Run statistical tests
    # ------------------------------------------------------------------

    all_results = []

    for family_name, comparisons in comparison_families.items():
        print(f"\n{'='*70}")
        print(f"Family: {family_name}")
        print(f"{'='*70}")

        family_raw_p = []
        family_indices = []

        for comp_label, exp_a, exp_b in comparisons:
            if exp_a not in experiments or exp_b not in experiments:
                print(f"\n  {comp_label}: SKIPPED (missing data)")
                continue

            data_a = experiments[exp_a]
            data_b = experiments[exp_b]

            # Find common tiles
            common_tiles = sorted(set(data_a.keys()) & set(data_b.keys()))
            n = len(common_tiles)

            if n < 5:
                print(f"\n  {comp_label}: SKIPPED (only {n} common tiles)")
                continue

            arr_a = np.array([data_a[t] for t in common_tiles])
            arr_b = np.array([data_b[t] for t in common_tiles])
            differences = arr_a - arr_b

            # --- Descriptive statistics ---
            mean_diff = float(np.mean(differences))
            median_diff = float(np.median(differences))
            std_diff = float(np.std(differences, ddof=1))

            print(f"\n  {comp_label} (n={n} tiles):")
            print(f"    {exp_a}: mean={np.mean(arr_a)*100:.2f}%")
            print(f"    {exp_b}: mean={np.mean(arr_b)*100:.2f}%")
            print(f"    Mean diff:   {mean_diff*100:+.2f} pp")
            print(f"    Median diff: {median_diff*100:+.2f} pp")
            print(f"    Std diff:    {std_diff*100:.2f} pp")

            # --- Wilcoxon signed-rank test ---
            w_stat, w_pvalue = wilcoxon_signed_rank_test(differences)
            print(f"    Wilcoxon:    W={w_stat:.1f}, p={w_pvalue:.4f}")

            # --- Permutation test ---
            perm_mean, perm_p = permutation_test(
                differences,
                n_permutations=args.n_permutations,
                seed=args.seed,
            )
            print(f"    Permutation: p={perm_p:.4f}")

            # --- Bootstrap CI ---
            ci_lower, ci_upper = bootstrap_ci(
                differences,
                n_bootstrap=args.n_bootstrap,
                seed=args.seed,
            )
            print(f"    95% CI:      [{ci_lower*100:+.2f}, {ci_upper*100:+.2f}] pp")

            # --- Cliff's delta ---
            delta = cliffs_delta(arr_a, arr_b)
            delta_interp = interpret_cliffs_delta(delta)
            print(f"    Cliff's d:   {delta:+.3f} ({delta_interp})")

            result = {
                "family": family_name,
                "comparison": comp_label,
                "experiment": exp_a,
                "anchor": exp_b,
                "n_tiles": n,
                "exp_mean": float(np.mean(arr_a)),
                "anchor_mean": float(np.mean(arr_b)),
                "mean_diff": mean_diff,
                "median_diff": median_diff,
                "std_diff": std_diff,
                "ci_95_lower": ci_lower,
                "ci_95_upper": ci_upper,
                "wilcoxon_stat": w_stat,
                "wilcoxon_p": w_pvalue,
                "permutation_p": perm_p,
                "cliffs_delta": delta,
                "cliffs_delta_interp": delta_interp,
            }

            family_raw_p.append(perm_p)
            family_indices.append(len(all_results))
            all_results.append(result)

        # --- Holm-Bonferroni within family ---
        if family_raw_p:
            adjusted = holm_bonferroni(family_raw_p)
            print(f"\n  Holm-Bonferroni correction ({len(family_raw_p)} tests):")
            for j, idx in enumerate(family_indices):
                r = all_results[idx]
                r["adjusted_p"] = adjusted[j]
                r["significant"] = adjusted[j] < ALPHA
                sig = "SIGNIFICANT" if r["significant"] else "not significant"
                print(
                    f"    {r['comparison']}: "
                    f"raw={r['permutation_p']:.4f}, "
                    f"adj={adjusted[j]:.4f} ({sig})"
                )

    # ------------------------------------------------------------------
    # 4. Summary table
    # ------------------------------------------------------------------

    if all_results:
        print(f"\n{'='*70}")
        print("SUMMARY TABLE")
        print(f"{'='*70}")

        header = (
            f"{'Comparison':<42} {'n':>3} {'Mean d':>8} "
            f"{'95% CI':>20} {'p_perm':>8} {'p_adj':>8} "
            f"{'Cliff d':>8} {'Sig':>5}"
        )
        print(header)
        print("-" * len(header))

        for r in all_results:
            adj_p = r.get("adjusted_p", r["permutation_p"])
            sig_str = (
                "***" if adj_p < 0.001
                else "**" if adj_p < 0.01
                else "*" if adj_p < 0.05
                else "ns"
            )
            ci_str = f"[{r['ci_95_lower']*100:+.1f}, {r['ci_95_upper']*100:+.1f}]"
            print(
                f"{r['comparison']:<42} {r['n_tiles']:>3} "
                f"{r['mean_diff']*100:>+7.2f}  "
                f"{ci_str:>20} "
                f"{r['permutation_p']:>8.4f} {adj_p:>8.4f} "
                f"{r['cliffs_delta']:>+7.3f}  {sig_str:>4}"
            )

        # LaTeX table
        print(f"\n{'='*70}")
        print("LATEX TABLE")
        print(f"{'='*70}")
        print(r"\begin{table}[h]")
        print(r"\centering")
        print(
            r"\caption{Per-tile paired statistical comparisons "
            r"(permutation test, $n$ = test tiles).}"
        )
        print(r"\label{tab:stats-persample-p2}")
        print(r"\begin{tabular}{lcccccc}")
        print(r"\toprule")
        print(
            r"\textbf{Comparison} & \textbf{$n$} & "
            r"\textbf{$\Delta$IoU (pp)} & \textbf{95\% CI} & "
            r"\textbf{$p_{\text{adj}}$} & "
            r"\textbf{Cliff's $\delta$} \\"
        )
        print(r"\midrule")

        current_family = None
        for r in all_results:
            if r["family"] != current_family:
                if current_family is not None:
                    print(r"\addlinespace")
                current_family = r["family"]

            adj_p = r.get("adjusted_p", r["permutation_p"])
            sig = ""
            if adj_p < 0.001:
                sig = "^{***}"
            elif adj_p < 0.01:
                sig = "^{**}"
            elif adj_p < 0.05:
                sig = "^{*}"

            comp_latex = r["comparison"].replace("_", r"\_")
            print(
                f"{comp_latex} & {r['n_tiles']} & "
                f"${r['mean_diff']*100:+.2f}$ & "
                f"$[{r['ci_95_lower']*100:+.2f}, {r['ci_95_upper']*100:+.2f}]$ & "
                f"${adj_p:.3f}{sig}$ & "
                f"${r['cliffs_delta']:+.2f}$ ({r['cliffs_delta_interp']}) \\\\"
            )

        print(r"\bottomrule")
        print(r"\end{tabular}")
        print(r"\end{table}")

    # ------------------------------------------------------------------
    # 5. Save results
    # ------------------------------------------------------------------

    # JSON
    json_path = output_dir / f"statistical_analysis_persample_{metric}.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "metric": metric,
                "source": "test_set_pertile",
                "n_permutations": args.n_permutations,
                "n_bootstrap": args.n_bootstrap,
                "seed": args.seed,
                "comparisons": all_results,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to: {json_path}")

    # CSV
    csv_lines = [
        "family,comparison,experiment,anchor,n_tiles,"
        "exp_mean,anchor_mean,mean_diff,ci_lower,ci_upper,"
        "perm_p,adj_p,cliffs_delta,cliffs_delta_interp,significant"
    ]
    for r in all_results:
        adj_p = r.get("adjusted_p", r["permutation_p"])
        sig = r.get("significant", adj_p < ALPHA)
        csv_lines.append(
            f"\"{r['family']}\",\"{r['comparison']}\","
            f"{r['experiment']},{r['anchor']},{r['n_tiles']},"
            f"{r['exp_mean']:.6f},{r['anchor_mean']:.6f},"
            f"{r['mean_diff']:.6f},{r['ci_95_lower']:.6f},{r['ci_95_upper']:.6f},"
            f"{r['permutation_p']:.6f},{adj_p:.6f},"
            f"{r['cliffs_delta']:.6f},{r['cliffs_delta_interp']},{sig}"
        )
    csv_path = output_dir / f"statistical_analysis_persample_{metric}.csv"
    with open(csv_path, "w") as f:
        f.write("\n".join(csv_lines) + "\n")
    print(f"CSV saved to: {csv_path}")

    # Also save per-tile IoU values for reproducibility
    pertile_data = {}
    for exp_name, data in experiments.items():
        pertile_data[exp_name] = data
    pertile_path = output_dir / f"per_tile_iou_{metric}.json"
    with open(pertile_path, "w") as f:
        json.dump(pertile_data, f, indent=2)
    print(f"Per-tile data saved to: {pertile_path}")


if __name__ == "__main__":
    main()
