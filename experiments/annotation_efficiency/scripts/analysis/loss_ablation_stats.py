#!/usr/bin/env python3
"""
Loss-function ablation: paired permutation tests vs the combined (focal+dice) baseline.

Compares two ablations (focal-only and dice-only) to the combined-loss baseline
on the same A3_s2_9band data and architecture, using per-tile IoU on the 40
held-out test tiles (probability ensemble across the five fold models).

Output: experiments/annotation_efficiency/outputs/statistical_tests/loss_ablation.json
"""

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
PART2 = REPO / "PART2_spectral_spatial_resolution_experiments"
AE_DIR = REPO / "experiments" / "annotation_efficiency"

OUTPUT_PATH = AE_DIR / "outputs" / "statistical_tests" / "loss_ablation.json"

N_PERMUTATIONS = 10000
N_BOOTSTRAP = 10000
SEED = 42
ALPHA = 0.05


def permutation_test(diff, n_permutations=N_PERMUTATIONS, seed=SEED):
    rng = np.random.RandomState(seed)
    obs = abs(np.mean(diff))
    cnt = 0
    for _ in range(n_permutations):
        signs = rng.choice([-1, 1], size=len(diff))
        if abs(np.mean(diff * signs)) >= obs:
            cnt += 1
    return float(np.mean(diff)), (cnt + 1) / (n_permutations + 1)


def bootstrap_ci(diff, n=N_BOOTSTRAP, ci=0.95, seed=SEED):
    rng = np.random.RandomState(seed)
    boot = np.empty(n)
    for i in range(n):
        idx = rng.choice(len(diff), size=len(diff), replace=True)
        boot[i] = np.mean(diff[idx])
    a = 1 - ci
    return float(np.percentile(boot, a / 2 * 100)), float(np.percentile(boot, (1 - a / 2) * 100))


def cliffs_delta(x, y):
    diff = x - y
    return float((np.sum(diff > 0) - np.sum(diff < 0)) / len(diff))


def interp(d):
    a = abs(d)
    if a < 0.147: return "negligible"
    if a < 0.33: return "small"
    if a < 0.474: return "medium"
    return "large"


def holm(p_values):
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adj = [0.0] * n
    for r, (i, p) in enumerate(indexed):
        adj[i] = min(1.0, p * (n - r))
    for k in range(1, n):
        i = indexed[k][0]
        prev = indexed[k - 1][0]
        adj[i] = max(adj[i], adj[prev])
    return adj


def load_pertile(test_results_path):
    with open(test_results_path) as f:
        data = json.load(f)
    return {tid: m["iou"] for tid, m in data["ensemble"]["per_sample"].items()}


EXP_BASE = PART2 / "outputs" / "experiments"

ANCHOR = ("Combined focal+dice (A3_s2_9band)", EXP_BASE / "A3_s2_9band" / "test_results.json")

COMPARISONS = [
    {"label": "Focal-only vs combined", "exp": EXP_BASE / "A3_s2_9band_focal_only" / "test_results.json"},
    {"label": "Dice-only vs combined",  "exp": EXP_BASE / "A3_s2_9band_dice_only"  / "test_results.json"},
]


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not ANCHOR[1].exists():
        print(f"ERROR: anchor results not found at {ANCHOR[1]}")
        return
    anchor_iou = load_pertile(ANCHOR[1])
    print(f"Anchor {ANCHOR[0]}: {len(anchor_iou)} tiles, mean={np.mean(list(anchor_iou.values()))*100:.2f}%")

    raw_p = []
    results = []
    for comp in COMPARISONS:
        if not comp["exp"].exists():
            print(f"  {comp['label']}: MISSING ({comp['exp']})")
            continue
        a = load_pertile(comp["exp"])
        common = sorted(set(a.keys()) & set(anchor_iou.keys()))
        x = np.array([a[t] for t in common])
        y = np.array([anchor_iou[t] for t in common])
        diff = x - y
        mean_d, p = permutation_test(diff)
        lo, hi = bootstrap_ci(diff)
        d = cliffs_delta(x, y)
        r = {
            "label": comp["label"],
            "n_tiles": len(common),
            "mean_a": float(np.mean(x)),
            "mean_b": float(np.mean(y)),
            "mean_diff": float(mean_d),
            "ci_95_lower": float(lo),
            "ci_95_upper": float(hi),
            "permutation_p": float(p),
            "cliffs_delta": float(d),
            "cliffs_delta_interp": interp(d),
        }
        results.append(r)
        raw_p.append(p)

    adj = holm(raw_p)
    for r, ap in zip(results, adj):
        r["adjusted_p"] = float(ap)
        r["significant"] = ap < ALPHA

    print()
    print(f"{'Comparison':<28} {'n':>3} {'a%':>7} {'b%':>7} {'Δpp':>8} {'95% CI':>22} {'p_perm':>8} {'p_adj':>8} {'Cliff δ':>10}")
    print("-" * 112)
    for r in results:
        ci_str = f"[{r['ci_95_lower']*100:+.2f},{r['ci_95_upper']*100:+.2f}]"
        sig = "*" if r["significant"] else ""
        print(
            f"{r['label']:<28} {r['n_tiles']:>3} "
            f"{r['mean_a']*100:>6.2f}% {r['mean_b']*100:>6.2f}% "
            f"{r['mean_diff']*100:>+7.2f}  {ci_str:>22} "
            f"{r['permutation_p']:>8.4f} {r['adjusted_p']:>7.4f}{sig} "
            f"{r['cliffs_delta']:>+7.2f} ({r['cliffs_delta_interp']})"
        )

    out = {
        "metric": "iou",
        "source": "test_set_pertile",
        "n_permutations": N_PERMUTATIONS,
        "n_bootstrap": N_BOOTSTRAP,
        "seed": SEED,
        "anchor": ANCHOR[0],
        "comparisons": results,
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
