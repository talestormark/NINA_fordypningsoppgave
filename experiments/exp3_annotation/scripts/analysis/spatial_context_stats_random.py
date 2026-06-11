#!/usr/bin/env python3
"""
Caveat 1 control: pairwise statistics for the per-pixel MLP control under
UNIFORM RANDOM sparse placement at 50 points/tile.

Three paired comparisons on the 40 test tiles:
  - MLP-random vs U-Net-random: settles Caveat 1 of Discussion §2. If the gap
    is not significant, the receptive field is irrelevant under random placement
    as well as under balanced.
  - MLP-random vs RF-random: sanity check. Expect a large, significant gap.
  - MLP-random vs MLP-balanced: characterises the random-placement penalty
    for the MLP itself.

Holm-Bonferroni adjustment across the three tests.

Mirrors spatial_context_stats.py (which handles the balanced regime).

Output: experiments/exp3_annotation/outputs/statistical_tests/spatial_context_stats_random.json
"""

import json
from pathlib import Path

import numpy as np

from landtake.paths import REPO_ROOT as REPO
AE_DIR = REPO / "experiments" / "exp3_annotation"

OUTPUT_PATH = AE_DIR / "outputs" / "statistical_tests" / "spatial_context_stats_random.json"

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
    """Load per-tile IoU from ensemble.per_sample in test_results.json (U-Net or MLP)."""
    with open(test_results_path) as f:
        data = json.load(f)
    return {tid: m[metric] for tid, m in data["ensemble"]["per_sample"].items()}


def load_rf_pertile(exp_dir, metric="iou"):
    """Average per-tile test IoU across 5 folds for an RF experiment."""
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
MLP_RANDOM_PATH = AE_DIR / "outputs" / "E5_D2_alphaearth_mlp_sparse_random" / "test_results.json"
MLP_BALANCED_PATH = AE_DIR / "outputs" / "E5_D2_alphaearth_mlp_sparse" / "test_results.json"
UNET_RANDOM_PATH = AE_DIR / "outputs" / "E4_rand_D2_alphaearth" / "test_results.json"
RF_RANDOM_DIR = AE_DIR / "outputs" / "E1_rand_ae_rf_sparse"


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not MLP_RANDOM_PATH.exists():
        print(f"MLP-random results not found at {MLP_RANDOM_PATH}")
        print("Did the MLP-random training and test inference complete?")
        return

    mlp_random = load_unet_pertile(MLP_RANDOM_PATH)
    unet_random = load_unet_pertile(UNET_RANDOM_PATH) if UNET_RANDOM_PATH.exists() else None
    mlp_balanced = load_unet_pertile(MLP_BALANCED_PATH) if MLP_BALANCED_PATH.exists() else None
    rf_random = load_rf_pertile(RF_RANDOM_DIR) if RF_RANDOM_DIR.exists() else None

    print(f"MLP-random:   n={len(mlp_random)},  mean={np.mean(list(mlp_random.values()))*100:.2f}%")
    if unet_random:
        print(f"U-Net-random: n={len(unet_random)}, mean={np.mean(list(unet_random.values()))*100:.2f}%")
    if mlp_balanced:
        print(f"MLP-balanced: n={len(mlp_balanced)}, mean={np.mean(list(mlp_balanced.values()))*100:.2f}%")
    if rf_random:
        print(f"RF-random:    n={len(rf_random)},   mean={np.mean(list(rf_random.values()))*100:.2f}%")
    print()

    pairs = {}
    if unet_random:
        pairs["MLP-random vs U-Net-random"] = (mlp_random, unet_random)
    if rf_random:
        pairs["MLP-random vs RF-random"] = (mlp_random, rf_random)
    if mlp_balanced:
        pairs["MLP-random vs MLP-balanced"] = (mlp_random, mlp_balanced)

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

    print(f"{'Comparison':<32} {'n':>3} {'a%':>7} {'b%':>7} {'Δpp':>7} {'95% CI':>22} {'p_perm':>8} {'p_adj':>8} {'Cliff δ':>16}")
    print("-" * 130)
    for r in results:
        ci_str = f"[{r['ci_95_lower']*100:+.2f},{r['ci_95_upper']*100:+.2f}]"
        sig = "*" if r["significant"] else ""
        print(
            f"{r['label']:<32} {r['n_tiles']:>3} "
            f"{r['mean_a']*100:>6.2f}% {r['mean_b']*100:>6.2f}% "
            f"{r['mean_diff']*100:>+6.2f}  {ci_str:>22} "
            f"{r['permutation_p']:>8.4f} {r['adjusted_p']:>8.4f}{sig} "
            f"{r['cliffs_delta']:>+7.2f} ({r['cliffs_delta_interp']})"
        )

    out = {
        "metric": "iou",
        "source": "test_set_pertile",
        "regime": "random_placement",
        "n_permutations": N_PERMUTATIONS,
        "n_bootstrap": N_BOOTSTRAP,
        "seed": SEED,
        "comparisons": results,
        "means": {
            "MLP-random": float(np.mean(list(mlp_random.values()))),
            **({"U-Net-random": float(np.mean(list(unet_random.values())))} if unet_random else {}),
            **({"MLP-balanced": float(np.mean(list(mlp_balanced.values())))} if mlp_balanced else {}),
            **({"RF-random": float(np.mean(list(rf_random.values())))} if rf_random else {}),
        },
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
