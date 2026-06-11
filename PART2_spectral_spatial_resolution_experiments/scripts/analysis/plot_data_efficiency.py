#!/usr/bin/env python3
"""
RQ2b data-efficiency: AlphaEarth vs raw 9-band Sentinel-2 as the number of
labelled training tiles shrinks.

Figure (two panels, shared per-tile macro IoU axis):
  (a) Cross-validation: mean per-fold IoU +/- std band, both inputs.
  (b) Held-out test:     per-tile macro IoU on the 5-fold ensemble, both inputs.
X-axis: number of training tiles per fold (log scale: 9, 18, 35, 70, 176).
The 176-tile point is the full-data anchor (existing A3_s2_9band / D2_alphaearth).

Stats: per training size, paired per-tile permutation test (AlphaEarth - Sentinel-2)
on the 40 test tiles, Holm-Bonferroni adjusted across the five sizes. Reuses the
exact routines from statistical_analysis_persample.py (same as the RQ2b table).

Output:
  REPORT/Figures/5_RQ2_InputRepresentation/data_efficiency_curve.pdf
  PART2.../outputs/analysis/data_efficiency_stats.{json,csv}
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
PART2_DIR = REPO_ROOT / "PART2_spectral_spatial_resolution_experiments"
EXPDIR = PART2_DIR / "outputs" / "experiments"
ANALYSIS_DIR = PART2_DIR / "outputs" / "analysis"
FIGURE_DIR = REPO_ROOT / "REPORT" / "Figures" / "5_RQ2_InputRepresentation"
# Two standalone panels; the (a)/(b) subcaptions are written in LaTeX, not here.
FIG_CV = FIGURE_DIR / "data_efficiency_cv.pdf"
FIG_TEST = FIGURE_DIR / "data_efficiency_test.pdf"

# Reuse the per-tile statistics used for the RQ2b table.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from statistical_analysis_persample import (  # noqa: E402
    load_unet_pertile_iou,
    permutation_test,
    bootstrap_ci,
    cliffs_delta,
    interpret_cliffs_delta,
    holm_bonferroni,
)

SIZES = [9, 18, 35, 70, 176]  # training tiles per fold; 176 = full-data anchor
STAT_SIZES = [9, 18, 35, 70, 176]  # full curve family incl. full-data anchor

# Okabe-Ito, colourblind-safe
C_S2 = "#0072B2"  # blue
C_AE = "#E69F00"  # orange


def s2_name(n):
    return "A3_s2_9band" if n == 176 else f"A3_s2_9band_n{n}"


def ae_name(n):
    return "D2_alphaearth" if n == 176 else f"D2_alphaearth_n{n}"


def cv_folds(exp):
    """Per-fold per-tile macro IoU (length 5)."""
    p = EXPDIR / exp / "cv_macro_summary.json"
    j = json.load(open(p))
    pf = j["per_fold_mean_iou"]
    return np.array([pf[str(k)] for k in sorted(pf, key=lambda x: int(x))], dtype=float)


def main():
    # --- collect curve data ---
    cv = {"s2": {}, "ae": {}}      # n -> (mean, std) of per-fold IoU
    test = {"s2": {}, "ae": {}}    # n -> per-tile dict {tile: iou}
    for n in SIZES:
        cv["s2"][n] = cv_folds(s2_name(n))
        cv["ae"][n] = cv_folds(ae_name(n))
        test["s2"][n] = load_unet_pertile_iou(s2_name(n), experiments_dir=EXPDIR)
        test["ae"][n] = load_unet_pertile_iou(ae_name(n), experiments_dir=EXPDIR)

    x = np.array(SIZES, dtype=float)
    s2_cv_mean = np.array([cv["s2"][n].mean() for n in SIZES]) * 100
    s2_cv_std = np.array([cv["s2"][n].std() for n in SIZES]) * 100
    ae_cv_mean = np.array([cv["ae"][n].mean() for n in SIZES]) * 100
    ae_cv_std = np.array([cv["ae"][n].std() for n in SIZES]) * 100
    s2_test = np.array([np.mean(list(test["s2"][n].values())) for n in SIZES]) * 100
    ae_test = np.array([np.mean(list(test["ae"][n].values())) for n in SIZES]) * 100

    # --- figures: two standalone panels, no baked-in titles ---
    # Subcaptions "(a) Cross-validation" / "(b) Held-out test" are added in LaTeX.
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    lo = min((s2_cv_mean - s2_cv_std).min(), (ae_cv_mean - ae_cv_std).min(),
             s2_test.min(), ae_test.min()) - 1.5
    hi = max((s2_cv_mean + s2_cv_std).max(), (ae_cv_mean + ae_cv_std).max(),
             s2_test.max(), ae_test.max()) + 1.5

    def panel(path, curves):
        fig, ax = plt.subplots(figsize=(5.0, 4.0))
        for yy, err, color, marker, label in curves:
            ax.plot(x, yy, marker=marker, color=color, label=label)
            if err is not None:
                ax.fill_between(x, yy - err, yy + err, color=color, alpha=0.15)
        ax.set_xscale("log")
        ax.set_xticks(SIZES)
        ax.set_xticklabels([str(s) for s in SIZES])
        ax.set_xlabel("Training tiles per fold")
        ax.set_ylabel("Per-tile macro IoU (%)")
        ax.set_ylim(lo, hi)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="lower right", fontsize=9)
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)

    panel(FIG_CV, [
        (s2_cv_mean, s2_cv_std, C_S2, "o", "Sentinel-2 (9-band)"),
        (ae_cv_mean, ae_cv_std, C_AE, "s", "AlphaEarth"),
    ])
    panel(FIG_TEST, [
        (s2_test, None, C_S2, "o", "Sentinel-2 (9-band)"),
        (ae_test, None, C_AE, "s", "AlphaEarth"),
    ])
    print(f"Figures saved: {FIG_CV}, {FIG_TEST}")

    # --- stats: AE - S2 per-tile permutation test at each reduced size ---
    rows, raw_p = [], []
    for n in STAT_SIZES:
        a, b = test["ae"][n], test["s2"][n]
        common = sorted(set(a) & set(b))
        arr_ae = np.array([a[t] for t in common])
        arr_s2 = np.array([b[t] for t in common])
        diff = arr_ae - arr_s2
        mean_diff, p = permutation_test(diff, n_permutations=10000, seed=42)
        lo, hi = bootstrap_ci(diff, n_bootstrap=10000, seed=42)
        d = cliffs_delta(arr_ae, arr_s2)
        rows.append({
            "n_train": n, "n_tiles": len(common),
            "ae_mean": float(arr_ae.mean()), "s2_mean": float(arr_s2.mean()),
            "mean_diff": float(mean_diff), "ci_lo": float(lo), "ci_hi": float(hi),
            "perm_p": float(p), "cliffs_delta": float(d),
            "cliffs_interp": interpret_cliffs_delta(d),
        })
        raw_p.append(p)
    adj = holm_bonferroni(raw_p)
    for r, ap in zip(rows, adj):
        r["adj_p"] = float(ap)
        r["significant"] = bool(ap < 0.05)

    # --- print summary + LaTeX ---
    print("\n=== AlphaEarth - Sentinel-2, per training size (per-tile test, n=40) ===")
    print(f"{'N':>4} {'S2':>6} {'AE':>6} {'dIoU':>7} {'95% CI':>18} {'p_adj':>7} {'Cliff d':>9} {'sig':>4}")
    for r in rows:
        ci = f"[{r['ci_lo']*100:+.1f},{r['ci_hi']*100:+.1f}]"
        sig = "*" if r["significant"] else "ns"
        print(f"{r['n_train']:>4} {r['s2_mean']*100:>6.1f} {r['ae_mean']*100:>6.1f} "
              f"{r['mean_diff']*100:>+7.2f} {ci:>18} {r['adj_p']:>7.3f} "
              f"{r['cliffs_delta']:>+9.3f} {sig:>4}")

    print("\n=== LaTeX (tab:rq2b_dataeff_stats) ===")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{AlphaEarth vs 9-band Sentinel-2 by training-set size. Per-tile "
          r"macro test IoU and paired per-tile permutation tests on the 40 test tiles, "
          r"Holm-Bonferroni adjusted across the five sizes.}")
    print(r"\label{tab:rq2b_dataeff_stats}")
    print(r"\begin{tabular}{@{}rrrrlrl@{}}")
    print(r"\toprule")
    print(r"\textbf{Train tiles} & \textbf{S-2} & \textbf{AE} & "
          r"\textbf{$\Delta$ (pp)} & \textbf{95\% CI} & \textbf{$p_{\text{adj}}$} & "
          r"\textbf{Cliff's $\delta$} \\")
    print(r"\midrule")
    for r in rows:
        ci = f"$[{r['ci_lo']*100:+.1f},\\,{r['ci_hi']*100:+.1f}]$"
        print(f"{r['n_train']} & {r['s2_mean']*100:.1f}\\% & {r['ae_mean']*100:.1f}\\% & "
              f"${r['mean_diff']*100:+.1f}$ & {ci} & ${r['adj_p']:.3f}$ & "
              f"${r['cliffs_delta']:+.2f}$ ({r['cliffs_interp']}) \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    # --- save ---
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "sizes": SIZES,
        "s2_cv_mean": s2_cv_mean.tolist(), "s2_cv_std": s2_cv_std.tolist(),
        "ae_cv_mean": ae_cv_mean.tolist(), "ae_cv_std": ae_cv_std.tolist(),
        "s2_test": s2_test.tolist(), "ae_test": ae_test.tolist(),
        "comparisons": rows,
    }
    with open(ANALYSIS_DIR / "data_efficiency_stats.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nStats saved: {ANALYSIS_DIR / 'data_efficiency_stats.json'}")


if __name__ == "__main__":
    main()
