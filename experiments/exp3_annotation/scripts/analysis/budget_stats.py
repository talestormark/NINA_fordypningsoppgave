#!/usr/bin/env python3
"""
RQ3c: Pairwise statistics for the point-budget sweep.

Loads per-tile test IoU for each budget variant (n5, n10, n50, n100) and the
dense AlphaEarth U-Net (D2). Runs paired per-tile permutation tests for each
budget vs D2, with bootstrap 95% CI and Cliff's delta. Holm-Bonferroni
adjusts within the 4-comparison family.

Output: experiments/exp3_annotation/outputs/statistical_tests/budget_stats.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

from landtake.paths import REPO_ROOT
PART2_DIR = REPO_ROOT / "experiments/exp2_input_representation"
AE_DIR = REPO_ROOT / "experiments" / "exp3_annotation"

OUTPUT_PATH = AE_DIR / "outputs" / "statistical_tests" / "budget_stats.json"

N_PERMUTATIONS = 10000
N_BOOTSTRAP = 10000
SEED = 42
ALPHA = 0.05


# ---------------------------------------------------------------------------
# Stats helpers (mirrored from PART2 statistical_analysis_persample.py)
# ---------------------------------------------------------------------------

def permutation_test(differences, n_permutations=N_PERMUTATIONS, seed=SEED):
    rng = np.random.RandomState(seed)
    observed = abs(np.mean(differences))
    n = len(differences)
    count = 0
    for _ in range(n_permutations):
        signs = rng.choice([-1, 1], size=n)
        if abs(np.mean(differences * signs)) >= observed:
            count += 1
    return float(np.mean(differences)), (count + 1) / (n_permutations + 1)


def bootstrap_ci(differences, n_bootstrap=N_BOOTSTRAP, ci=0.95, seed=SEED):
    rng = np.random.RandomState(seed)
    n = len(differences)
    boot = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot[i] = np.mean(differences[idx])
    alpha = 1 - ci
    return float(np.percentile(boot, alpha / 2 * 100)), float(np.percentile(boot, (1 - alpha / 2) * 100))


def cliffs_delta(x, y):
    diff = x - y
    n_greater = np.sum(diff > 0)
    n_less = np.sum(diff < 0)
    return float((n_greater - n_less) / len(diff))


def interpret_cliffs(d):
    a = abs(d)
    if a < 0.147: return "negligible"
    if a < 0.33:  return "small"
    if a < 0.474: return "medium"
    return "large"


def holm_bonferroni(p_values):
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * n
    for rank, (orig_idx, p) in enumerate(indexed):
        adjusted[orig_idx] = min(1.0, p * (n - rank))
    for i in range(1, n):
        idx = indexed[i][0]
        prev_idx = indexed[i - 1][0]
        adjusted[idx] = max(adjusted[idx], adjusted[prev_idx])
    return adjusted


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_pertile_iou(test_results_path, metric="iou"):
    """Load per-tile IoU dict from a test_results.json (prefers ensemble)."""
    with open(test_results_path) as f:
        data = json.load(f)
    if "ensemble" in data and "per_sample" in data["ensemble"]:
        return {tid: m[metric] for tid, m in data["ensemble"]["per_sample"].items()}
    # Fallback: average across folds
    folds = data.get("folds", {})
    tile_values = {}
    for fold_data in folds.values():
        for tid, m in fold_data.get("per_sample", {}).items():
            tile_values.setdefault(tid, []).append(m[metric])
    return {tid: float(np.mean(v)) for tid, v in tile_values.items() if len(v) == len(folds)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

BUDGETS = [
    ("n5",   "E4_D2_alphaearth_sparse_n5",   AE_DIR / "outputs" / "E4_D2_alphaearth_sparse_n5"   / "test_results.json"),
    ("n10",  "E4_D2_alphaearth_sparse_n10",  AE_DIR / "outputs" / "E4_D2_alphaearth_sparse_n10"  / "test_results.json"),
    ("n15",  "E4_D2_alphaearth_sparse_n15",  AE_DIR / "outputs" / "E4_D2_alphaearth_sparse_n15"  / "test_results.json"),
    ("n50",  "E4_ae_unet_sparse",            PART2_DIR / "outputs" / "experiments" / "E4_ae_unet_sparse" / "test_results.json"),
    ("n100", "E4_D2_alphaearth_sparse_n100", AE_DIR / "outputs" / "E4_D2_alphaearth_sparse_n100" / "test_results.json"),
]
ANCHOR = ("D2", "D2_alphaearth", PART2_DIR / "outputs" / "experiments" / "D2_alphaearth" / "test_results.json")


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Load anchor
    if not ANCHOR[2].exists():
        print(f"ERROR: anchor results not found at {ANCHOR[2]}")
        sys.exit(1)
    anchor_iou = load_pertile_iou(ANCHOR[2])
    print(f"Anchor {ANCHOR[1]}: {len(anchor_iou)} tiles, mean={np.mean(list(anchor_iou.values()))*100:.2f}%")

    # Load each budget
    budget_iou = {}
    for budget_label, exp_name, path in BUDGETS:
        if not path.exists():
            print(f"  {budget_label}: MISSING ({path})")
            continue
        budget_iou[budget_label] = load_pertile_iou(path)
        mean = np.mean(list(budget_iou[budget_label].values())) * 100
        print(f"  {budget_label} ({exp_name}): {len(budget_iou[budget_label])} tiles, mean={mean:.2f}%")

    # Run pairwise tests vs anchor
    raw_p = []
    comparisons = []

    for budget_label, exp_name, _ in BUDGETS:
        if budget_label not in budget_iou:
            continue
        b = budget_iou[budget_label]
        common = sorted(set(b.keys()) & set(anchor_iou.keys()))
        x = np.array([b[t] for t in common])
        y = np.array([anchor_iou[t] for t in common])
        diff = x - y

        mean_diff, perm_p = permutation_test(diff)
        ci_low, ci_high = bootstrap_ci(diff)
        delta = cliffs_delta(x, y)

        result = {
            "budget": budget_label,
            "experiment": exp_name,
            "anchor": ANCHOR[1],
            "n_tiles": len(common),
            "exp_mean": float(np.mean(x)),
            "anchor_mean": float(np.mean(y)),
            "mean_diff": float(mean_diff),
            "ci_95_lower": float(ci_low),
            "ci_95_upper": float(ci_high),
            "permutation_p": float(perm_p),
            "cliffs_delta": float(delta),
            "cliffs_delta_interp": interpret_cliffs(delta),
        }
        comparisons.append(result)
        raw_p.append(perm_p)

    # Holm-Bonferroni
    adjusted = holm_bonferroni(raw_p)
    for r, adj in zip(comparisons, adjusted):
        r["adjusted_p"] = float(adj)
        r["significant"] = adj < ALPHA

    # Print summary
    print()
    print(f"{'Budget':<6} {'n':>3} {'Exp':>7} {'Anc':>7} {'Δpp':>7} {'95% CI':>20} {'p_perm':>8} {'p_adj':>8} {'Cliff δ':>8}")
    print("-" * 90)
    for r in comparisons:
        ci_str = f"[{r['ci_95_lower']*100:+.2f},{r['ci_95_upper']*100:+.2f}]"
        sig = "*" if r["significant"] else ""
        print(
            f"{r['budget']:<6} {r['n_tiles']:>3} "
            f"{r['exp_mean']*100:>6.2f}% {r['anchor_mean']*100:>6.2f}% "
            f"{r['mean_diff']*100:>+6.2f}  {ci_str:>20} "
            f"{r['permutation_p']:>8.4f} {r['adjusted_p']:>8.4f}{sig} "
            f"{r['cliffs_delta']:>+7.2f} ({r['cliffs_delta_interp']})"
        )

    # Save
    out = {
        "metric": "iou",
        "source": "test_set_pertile",
        "n_permutations": N_PERMUTATIONS,
        "n_bootstrap": N_BOOTSTRAP,
        "seed": SEED,
        "anchor": ANCHOR[1],
        "comparisons": comparisons,
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
