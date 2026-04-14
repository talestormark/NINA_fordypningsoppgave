#!/usr/bin/env python3
"""
Statistical comparison of Block A and RQ3e experiments.

Uses per-fold CV IoU values for paired comparisons (n=5 folds).
Permutation tests + bootstrap CIs, Holm-Bonferroni correction.
"""

import json
import numpy as np
from pathlib import Path
from itertools import combinations

REPO_ROOT = Path(__file__).resolve().parents[4]
P2_OUTPUTS = REPO_ROOT / "PART2_spectral_spatial_resolution_experiments" / "outputs" / "experiments"
AE_OUTPUTS = Path(__file__).resolve().parents[2] / "outputs"

N_PERMUTATIONS = 10000
N_BOOTSTRAP = 10000
SEED = 42
ALPHA = 0.05


def get_fold_ious_unet(exp_name, base_dir=P2_OUTPUTS):
    """Extract best val IoU per fold from history.json."""
    ious = []
    for fold in range(5):
        hp = base_dir / f"{exp_name}_fold{fold}" / "history.json"
        if not hp.exists():
            return None
        with open(hp) as f:
            h = json.load(f)
        best = max(entry['iou'] for entry in h['val'])
        ious.append(best)
    return np.array(ious)


def get_fold_ious_rf(exp_name, base_dir=AE_OUTPUTS):
    """Extract val IoU per fold from RF metrics.json."""
    ious = []
    for fold in range(5):
        mp = base_dir / exp_name / f"fold{fold}" / "metrics.json"
        if not mp.exists():
            return None
        with open(mp) as f:
            m = json.load(f)
        ious.append(m["val_metrics"]["macro_iou"])
    return np.array(ious)


def permutation_test(diffs, n_perm=N_PERMUTATIONS, seed=SEED):
    """Two-sided sign-flip permutation test."""
    rng = np.random.default_rng(seed)
    observed = np.mean(diffs)
    n = len(diffs)
    count = 0
    for _ in range(n_perm):
        signs = rng.choice([-1, 1], size=n)
        if abs(np.mean(diffs * signs)) >= abs(observed):
            count += 1
    return observed, count / n_perm


def bootstrap_ci(diffs, n_boot=N_BOOTSTRAP, ci=0.95, seed=SEED):
    """Bootstrap percentile CI for mean difference."""
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(diffs, size=len(diffs), replace=True)
        means.append(np.mean(sample))
    lower = np.percentile(means, (1 - ci) / 2 * 100)
    upper = np.percentile(means, (1 + ci) / 2 * 100)
    return lower, upper


def holm_bonferroni(p_values):
    """Holm-Bonferroni correction."""
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0] * n
    for rank, (orig_idx, p) in enumerate(indexed):
        adjusted[orig_idx] = min(1.0, p * (n - rank))
    # Enforce monotonicity
    for i in range(1, n):
        idx = indexed[i][0]
        prev_idx = indexed[i - 1][0]
        adjusted[idx] = max(adjusted[idx], adjusted[prev_idx])
    return adjusted


def main():
    print("=" * 70)
    print("STATISTICAL TESTS — Part 2 + RQ3e")
    print("=" * 70)

    # Collect all experiments
    experiments = {}

    # Block A (U-Net)
    for name in ["A3_s2_9band", "A1_s2_rgb", "A2_s2_rgbnir", "A4_s2_indices"]:
        ious = get_fold_ious_unet(name)
        if ious is not None:
            experiments[name] = ious
            print(f"  {name}: {ious.mean()*100:.1f}% ± {ious.std()*100:.1f}%")

    # Block D (U-Net)
    for name in ["D2_alphaearth"]:
        ious = get_fold_ious_unet(name)
        if ious is not None:
            experiments[name] = ious
            print(f"  {name}: {ious.mean()*100:.1f}% ± {ious.std()*100:.1f}%")

    # E4 (U-Net, masked loss) — check both possible locations
    e4_ious = get_fold_ious_unet("E4_ae_unet_sparse")
    if e4_ious is None:
        # Try annotation_efficiency outputs
        e4_ious = []
        for fold in range(5):
            hp = AE_OUTPUTS / "E4_ae_unet_sparse" / f"fold{fold}" / "history.json"
            if hp.exists():
                with open(hp) as f:
                    h = json.load(f)
                e4_ious.append(max(entry['iou'] for entry in h['val']))
        if e4_ious:
            e4_ious = np.array(e4_ious)
        else:
            # Fallback: use values from experiment log (E4 had no history.json due to timeout)
            e4_ious = np.array([0.535, 0.531, 0.532, 0.546, 0.548])
            print(f"  E4_ae_unet_sparse: {e4_ious.mean()*100:.1f}% ± {e4_ious.std()*100:.1f}% (from SLURM logs)")
    if e4_ious is not None:
        experiments["E4_ae_unet_sparse"] = e4_ious
        if "E4_ae_unet_sparse" not in [k for k in experiments if "E4" in k]:
            print(f"  E4_ae_unet_sparse: {e4_ious.mean()*100:.1f}% ± {e4_ious.std()*100:.1f}%")

    # E1 and E2 (RF)
    for name in ["E1_ae_rf_sparse", "E2_s2_rf_sparse"]:
        ious = get_fold_ious_rf(name)
        if ious is not None:
            experiments[name] = ious
            print(f"  {name}: {ious.mean()*100:.1f}% ± {ious.std()*100:.1f}%")

    print()

    # Define comparison families
    families = {
        "Block A: Spectral ablation (vs A3 anchor)": [
            ("A1_s2_rgb", "A3_s2_9band"),
            ("A2_s2_rgbnir", "A3_s2_9band"),
            ("A4_s2_indices", "A3_s2_9band"),
        ],
        "RQ3e: Annotation efficiency": [
            ("E4_ae_unet_sparse", "D2_alphaearth"),   # sparse vs dense (same model, same input)
            ("E4_ae_unet_sparse", "A3_s2_9band"),     # sparse AE vs dense S2
            ("E1_ae_rf_sparse", "A3_s2_9band"),       # RF sparse vs dense U-Net
            ("D2_alphaearth", "A3_s2_9band"),          # AE vs S2 (both dense)
        ],
    }

    all_results = {}

    for family_name, comparisons in families.items():
        print(f"\n{'='*60}")
        print(f"Family: {family_name}")
        print(f"{'='*60}")

        family_pvals = []
        family_labels = []

        for exp_a, exp_b in comparisons:
            if exp_a not in experiments or exp_b not in experiments:
                print(f"\n  {exp_a} vs {exp_b}: SKIPPED (missing data)")
                continue

            ious_a = experiments[exp_a]
            ious_b = experiments[exp_b]
            diffs = ious_a - ious_b

            mean_diff, p_value = permutation_test(diffs)
            ci_low, ci_high = bootstrap_ci(diffs)

            family_pvals.append(p_value)
            family_labels.append(f"{exp_a} vs {exp_b}")

            result = {
                "mean_diff_pp": float(mean_diff * 100),
                "ci_95": [float(ci_low * 100), float(ci_high * 100)],
                "p_value": float(p_value),
            }
            all_results[f"{exp_a} vs {exp_b}"] = result

            print(f"\n  {exp_a} vs {exp_b}:")
            print(f"    Mean Δ: {mean_diff*100:+.1f} pp")
            print(f"    95% CI: [{ci_low*100:+.1f}, {ci_high*100:+.1f}] pp")
            print(f"    p-value: {p_value:.4f}")

        # Holm-Bonferroni correction within family
        if family_pvals:
            adjusted = holm_bonferroni(family_pvals)
            print(f"\n  Holm-Bonferroni adjusted p-values:")
            for label, raw_p, adj_p in zip(family_labels, family_pvals, adjusted):
                sig = "SIGNIFICANT" if adj_p < ALPHA else "not significant"
                print(f"    {label}: raw={raw_p:.4f}, adj={adj_p:.4f} ({sig})")
                all_results[label]["p_adj"] = float(adj_p)

    # Save results
    out_dir = AE_OUTPUTS / "statistical_tests"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "statistical_tests.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_dir / 'statistical_tests.json'}")


if __name__ == "__main__":
    main()
