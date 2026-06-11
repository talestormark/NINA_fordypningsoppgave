#!/usr/bin/env python3
"""
M1: Pairwise statistics for the per-pixel MLP control vs the U-Net and the RF.

Two paired comparisons on the 40 test tiles:
  - MLP vs RF: tests whether neural-network end-to-end training matters when
    spatial context is absent. Both models are per-pixel classifiers.
  - MLP vs U-Net: tests whether spatial context explains the gap between the
    per-pixel NN and the convolutional U-Net. Both models share the same
    training procedure; only the receptive field differs.

Holm-Bonferroni adjustment across the two tests.

Output: experiments/exp3_annotation/outputs/statistical_tests/spatial_context_stats.json
"""

import json
from pathlib import Path

import numpy as np

from landtake.paths import REPO_ROOT as REPO
PART2_DIR = REPO / "experiments/exp2_input_representation"
AE_DIR = REPO / "experiments" / "exp3_annotation"

OUTPUT_PATH = AE_DIR / "outputs" / "statistical_tests" / "spatial_context_stats.json"

N_PERMUTATIONS = 10000
N_BOOTSTRAP = 10000
SEED = 42
ALPHA = 0.05


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


def load_unet_pertile(test_results_path, metric="iou"):
    with open(test_results_path) as f:
        data = json.load(f)
    return {tid: m[metric] for tid, m in data["ensemble"]["per_sample"].items()}


def load_rf_pertile(exp_dir, metric="iou"):
    tile_values = {}
    n_folds = 0
    for fold in range(5):
        path = exp_dir / f"fold{fold}" / "metrics.json"
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        per_tile = data.get("test_metrics", {}).get("per_tile", {})
        if not per_tile:
            continue
        n_folds += 1
        for tid, m in per_tile.items():
            tile_values.setdefault(tid, []).append(m[metric])
    return {tid: float(np.mean(v)) for tid, v in tile_values.items() if len(v) == n_folds}


# Sources
MLP_PATH = AE_DIR / "outputs" / "E5_D2_alphaearth_mlp_sparse" / "test_results.json"
UNET_PATH = PART2_DIR / "outputs" / "experiments" / "E4_ae_unet_sparse" / "test_results.json"
RF_DIR = AE_DIR / "outputs" / "E1_ae_rf_sparse"


COMPARISONS = [
    {"label": "MLP vs RF",   "exp_a": "E5 MLP",  "exp_b": "E1 RF"},
    {"label": "MLP vs U-Net", "exp_a": "E5 MLP", "exp_b": "E4 U-Net"},
]


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not MLP_PATH.exists():
        print(f"MLP results not found at {MLP_PATH}")
        return
    mlp = load_unet_pertile(MLP_PATH)
    unet = load_unet_pertile(UNET_PATH)
    rf = load_rf_pertile(RF_DIR)

    print(f"MLP:   n={len(mlp)},  mean={np.mean(list(mlp.values()))*100:.2f}%")
    print(f"U-Net: n={len(unet)}, mean={np.mean(list(unet.values()))*100:.2f}%")
    print(f"RF:    n={len(rf)},   mean={np.mean(list(rf.values()))*100:.2f}%")

    pairs = {
        "MLP vs RF":    (mlp, rf),
        "MLP vs U-Net": (mlp, unet),
    }

    raw_p = []
    results = []
    for label, (a, b) in pairs.items():
        common = sorted(set(a.keys()) & set(b.keys()))
        x = np.array([a[t] for t in common])
        y = np.array([b[t] for t in common])
        diff = x - y

        mean_diff, perm_p = permutation_test(diff)
        ci_low, ci_high = bootstrap_ci(diff)
        delta = cliffs_delta(x, y)

        r = {
            "label": label,
            "n_tiles": len(common),
            "mean_a": float(np.mean(x)),
            "mean_b": float(np.mean(y)),
            "mean_diff": float(mean_diff),
            "ci_95_lower": float(ci_low),
            "ci_95_upper": float(ci_high),
            "permutation_p": float(perm_p),
            "cliffs_delta": float(delta),
            "cliffs_delta_interp": interpret_cliffs(delta),
        }
        results.append(r)
        raw_p.append(perm_p)

    adjusted = holm_bonferroni(raw_p)
    for r, adj in zip(results, adjusted):
        r["adjusted_p"] = float(adj)
        r["significant"] = adj < ALPHA

    print()
    print(f"{'Comparison':<20} {'n':>3} {'a%':>7} {'b%':>7} {'Δpp':>7} {'95% CI':>20} {'p_perm':>8} {'p_adj':>8} {'Cliff δ':>8}")
    print("-" * 110)
    for r in results:
        ci_str = f"[{r['ci_95_lower']*100:+.2f},{r['ci_95_upper']*100:+.2f}]"
        sig = "*" if r["significant"] else ""
        print(
            f"{r['label']:<20} {r['n_tiles']:>3} "
            f"{r['mean_a']*100:>6.2f}% {r['mean_b']*100:>6.2f}% "
            f"{r['mean_diff']*100:>+6.2f}  {ci_str:>20} "
            f"{r['permutation_p']:>8.4f} {r['adjusted_p']:>8.4f}{sig} "
            f"{r['cliffs_delta']:>+7.2f} ({r['cliffs_delta_interp']})"
        )

    out = {
        "metric": "iou",
        "source": "test_set_pertile",
        "n_permutations": N_PERMUTATIONS,
        "n_bootstrap": N_BOOTSTRAP,
        "seed": SEED,
        "comparisons": results,
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
