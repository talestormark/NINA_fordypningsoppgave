#!/usr/bin/env python3
"""
RQ3c figure: accuracy vs point budget for the masked-loss U-Net on AlphaEarth.

Combined single-panel plot (CV and Test on a shared per-tile macro IoU axis,
introduced after the CV-macro recompute since both columns now use the same
metric).

X-axis: point budget per tile (log scale: 10, 20, 30, 50, 200).
Y-axis: per-tile macro IoU (%).
Curves: CV mean (with ±std band) and Test (markers per budget).
Horizontal dotted references for the dense U-Net baseline (D2), one per curve.

Output: REPORT/Figures/6_RQ3_AnnotationEfficiency/budget_curve.pdf
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
PART2_DIR = REPO_ROOT / "PART2_spectral_spatial_resolution_experiments"
AE_DIR = REPO_ROOT / "experiments" / "annotation_efficiency"
FIGURE_DIR = REPO_ROOT / "REPORT" / "Figures" / "6_RQ3_AnnotationEfficiency"
FIGURE_OUT = FIGURE_DIR / "budget_curve.pdf"


# Per-fold CV per-tile macro IoU at the best-by-micro checkpoint.
# Source: cv_macro_summary.json per experiment (CV-macro recompute, 2026-05-16).
CV_FOLDS = {
    10:  [0.3896, 0.3770, 0.4378, 0.4362, 0.4139],
    20:  [0.4181, 0.4016, 0.4338, 0.4538, 0.4332],
    30:  [0.4252, 0.4137, 0.4423, 0.4497, 0.4137],
    50:  [0.4115, 0.3967, 0.4327, 0.4619, 0.4474],
    200: [0.4367, 0.3946, 0.4546, 0.4680, 0.4531],
}

# Test results paths (test_results.json with ensemble per_sample.iou).
TEST_RESULTS_PATHS = {
    10:  AE_DIR / "outputs" / "E4_D2_alphaearth_sparse_n5"   / "test_results.json",
    20:  AE_DIR / "outputs" / "E4_D2_alphaearth_sparse_n10"  / "test_results.json",
    30:  AE_DIR / "outputs" / "E4_D2_alphaearth_sparse_n15"  / "test_results.json",
    50:  PART2_DIR / "outputs" / "experiments" / "E4_ae_unet_sparse" / "test_results.json",
    200: AE_DIR / "outputs" / "E4_D2_alphaearth_sparse_n100" / "test_results.json",
}

# Dense reference (D2_alphaearth)
DENSE_TEST_PATH = PART2_DIR / "outputs" / "experiments" / "D2_alphaearth" / "test_results.json"
DENSE_CV_FOLDS = [0.4337, 0.4112, 0.4613, 0.4766, 0.4496]  # D2 per-tile macro IoU per fold (cv_macro_summary.json)


def load_test_iou(path):
    """Per-tile macro IoU on the ensemble (matches stats unit)."""
    if not path.exists():
        return None
    with open(path) as f:
        d = json.load(f)
    per_sample = d["ensemble"]["per_sample"]
    return float(np.mean([v["iou"] for v in per_sample.values()]))


def main():
    budgets = sorted(CV_FOLDS.keys())
    cv_means = np.array([np.mean(CV_FOLDS[b]) for b in budgets]) * 100
    cv_stds = np.array([np.std(CV_FOLDS[b]) for b in budgets]) * 100  # ddof=0 matches main-text tables

    test_ious = {}
    for b, p in TEST_RESULTS_PATHS.items():
        v = load_test_iou(p)
        if v is not None:
            test_ious[b] = v * 100

    dense_test = load_test_iou(DENSE_TEST_PATH)
    dense_cv = np.mean(DENSE_CV_FOLDS) * 100  # rough placeholder

    print("Point budget sweep summary:")
    for b in budgets:
        cv = cv_means[budgets.index(b)]
        std = cv_stds[budgets.index(b)]
        test = test_ious.get(b, None)
        test_str = f"{test:.2f}%" if test is not None else "TODO"
        print(f"  {b:3d} pts: CV {cv:.2f}% ± {std:.2f}%  Test {test_str}")
    if dense_test is not None:
        print(f"  Dense  : CV {dense_cv:.2f}%           Test {dense_test*100:.2f}%")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- Combined CV + Test panel ----------
    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    # CV curve with std band
    ax.plot(budgets, cv_means, marker='o', color='#1f77b4', linewidth=2,
            label='CV (mean ± std)')
    ax.fill_between(budgets, cv_means - cv_stds, cv_means + cv_stds,
                    color='#1f77b4', alpha=0.15)
    ax.axhline(dense_cv, color='#1f77b4', linestyle=':', linewidth=1.2, alpha=0.7,
               label=f'Dense U-Net CV ({dense_cv:.1f}%)')

    # Test curve
    if test_ious:
        test_x = sorted(test_ious.keys())
        test_y = [test_ious[b] for b in test_x]
        ax.plot(test_x, test_y, marker='s', color='#d62728', linewidth=2, label='Test')
    if dense_test is not None:
        ax.axhline(dense_test * 100, color='#d62728', linestyle=':', linewidth=1.2, alpha=0.7,
                   label=f'Dense U-Net Test ({dense_test*100:.1f}%)')

    ax.set_xscale('log')
    ax.set_xticks(budgets)
    ax.set_xticklabels([str(b) for b in budgets])
    ax.set_xlabel('Points per tile (balanced)')
    ax.set_ylabel('Per-tile macro IoU (%)')
    ax.grid(True, which='both', linestyle=':', alpha=0.4)
    ax.legend(loc='lower right', fontsize=9, framealpha=0.95)

    all_y = (list(cv_means) + list(cv_means - cv_stds) + list(cv_means + cv_stds)
             + [dense_cv] + list(test_ious.values()))
    if dense_test is not None:
        all_y.append(dense_test * 100)
    y_min, y_max = min(all_y), max(all_y)
    pad = (y_max - y_min) * 0.12
    ax.set_ylim(y_min - pad, y_max + pad)

    fig.tight_layout()
    fig.savefig(FIGURE_OUT, dpi=150, bbox_inches='tight')
    print(f"\nSaved {FIGURE_OUT}")


if __name__ == "__main__":
    main()
