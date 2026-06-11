#!/usr/bin/env python3
"""
For each experiment in RQ1, RQ2, RQ3 compute the per-fold std of per-tile
macro test IoU. Used to populate the "$\\pm$ std" component of the Test IoU
column in body tables (M4 fix).

For each experiment, the central value is the ensemble per-tile macro IoU
(unchanged from `compute_macro_test_iou.py`). The std is the std (ddof=1) of
the 5 per-fold per-tile macro IoUs, where the per-fold value is the mean of
that fold's per-tile IoU on the test set.

This measures fold-to-fold variability of the trained model on the test set,
analogous to what the CV ± std reports on validation data.

Output: experiments/annotation_efficiency/outputs/test_iou_fold_std.csv
"""

import csv
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
PART1 = REPO / "PART1_multi_temporal_experiments"
PART2 = REPO / "PART2_spectral_spatial_resolution_experiments"
AE_DIR = REPO / "experiments" / "annotation_efficiency"

OUT_CSV = AE_DIR / "outputs" / "test_iou_fold_std.csv"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_unet_p2(exp_dir: Path):
    """U-Net (PART2 evaluate_test_set.py output). Returns ensemble mean and fold std."""
    p = exp_dir / "test_results.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    # Ensemble per-tile macro mean
    ens_macros = [v["iou"] for v in d["ensemble"]["per_sample"].values()]
    ensemble_mean = float(np.mean(ens_macros))
    n_tiles = len(ens_macros)
    # Per-fold per-tile macro
    fold_macros = []
    for fold_str, fold_data in d.get("folds", {}).items():
        per_sample = fold_data.get("per_sample", {})
        if not per_sample:
            continue
        per_tile = [m["iou"] for m in per_sample.values()]
        fold_macros.append(float(np.mean(per_tile)))
    if len(fold_macros) < 2:
        return None
    fold_std = float(np.std(fold_macros, ddof=1))
    return {
        "ensemble_mean": ensemble_mean,
        "fold_std": fold_std,
        "n_tiles": n_tiles,
        "n_folds": len(fold_macros),
    }


def load_unet_part1(test_eval_path: Path, condition: str):
    """RQ1 conditions. Ensemble mean from per_tile.iou.mean.
    For per-fold std, need per_sample_iou.json which has per-fold per-tile data."""
    d = json.load(open(test_eval_path))
    if condition not in d:
        return None
    entry = d[condition]
    ensemble_mean = entry["per_tile"]["iou"]["mean"]
    n_tiles = entry["n_test_tiles"]
    # Look for per-fold data
    per_sample_path = test_eval_path.parent / "per_sample_iou.json"
    if per_sample_path.exists():
        ps = json.load(open(per_sample_path))
        # Structure may vary; try to find per-fold per-tile data for the condition
        # The Part1 per_sample_iou.json typically has structure: {condition: {fold: {tile: iou}}}
        if condition in ps:
            cond_data = ps[condition]
            fold_macros = []
            if isinstance(cond_data, dict):
                # Check for per-fold structure
                for k, v in cond_data.items():
                    if isinstance(v, dict):
                        ious = [val for val in v.values() if isinstance(val, (int, float))]
                        if ious:
                            fold_macros.append(float(np.mean(ious)))
            if len(fold_macros) >= 2:
                return {
                    "ensemble_mean": float(ensemble_mean),
                    "fold_std": float(np.std(fold_macros, ddof=1)),
                    "n_tiles": int(n_tiles),
                    "n_folds": len(fold_macros),
                }
    # Fallback: no per-fold data accessible
    return {
        "ensemble_mean": float(ensemble_mean),
        "fold_std": None,
        "n_tiles": int(n_tiles),
        "n_folds": 0,
    }


def load_rf(rf_dir: Path, metric="iou"):
    """RF experiments. Per-fold per-tile macro from metrics.json."""
    fold_macros = []
    for fold in range(5):
        path = rf_dir / f"fold{fold}" / "metrics.json"
        if not path.exists():
            continue
        d = json.load(open(path))
        per_tile = d.get("test_metrics", {}).get("per_tile", {})
        if not per_tile:
            continue
        per_tile_iou = [m[metric] for m in per_tile.values()]
        fold_macros.append(float(np.mean(per_tile_iou)))
    if len(fold_macros) < 2:
        return None
    return {
        "ensemble_mean": float(np.mean(fold_macros)),  # RF "ensemble" = mean across folds
        "fold_std": float(np.std(fold_macros, ddof=1)),
        "n_tiles": None,
        "n_folds": len(fold_macros),
    }


# ---------------------------------------------------------------------------
# Experiment registry (mirrors compute_macro_test_iou.py)
# ---------------------------------------------------------------------------

RQ1_TEST_EVAL = PART1 / "outputs_v3" / "analysis" / "test_set_evaluation.json"

RQ1 = [
    ("LSTM-2", "bi_temporal"),
    ("LSTM-7", "annual"),
    ("LSTM-14", "bi_seasonal"),
    ("EarlyFusion", "early_fusion"),
    ("LateFusion", "late_fusion"),
    ("LSTM-2-lite", "lstm_lite"),
    ("Pool-7", "late_fusion_pool"),
    ("Conv3D-7", "conv3d_fusion"),
    ("LSTM-7-lite", "lstm7_lite"),
]

RQ2_UNET = [
    ("RGB", "A1_s2_rgb"),
    ("RGB+NIR", "A2_s2_rgbnir"),
    ("9-band", "A3_s2_9band"),
    ("9-band + indices", "A4_s2_indices"),
    ("AlphaEarth", "D2_alphaearth"),
]

RQ3_UNET = [
    ("D2 dense AE", "D2_alphaearth", PART2 / "outputs" / "experiments"),
    ("E4 sparse AE (50)", "E4_ae_unet_sparse", PART2 / "outputs" / "experiments"),
    ("A3 dense S-2", "A3_s2_9band", PART2 / "outputs" / "experiments"),
    ("E4-S2 sparse", "E4_A3_s2_9band_sparse", PART2 / "outputs" / "experiments"),
    ("E4 sparse AE (n5/10pts)", "E4_D2_alphaearth_sparse_n5", AE_DIR / "outputs"),
    ("E4 sparse AE (n10/20pts)", "E4_D2_alphaearth_sparse_n10", AE_DIR / "outputs"),
    ("E4 sparse AE (n15/30pts)", "E4_D2_alphaearth_sparse_n15", AE_DIR / "outputs"),
    ("E4 sparse AE (n100/200pts)", "E4_D2_alphaearth_sparse_n100", AE_DIR / "outputs"),
    ("E4-rand AE", "E4_rand_D2_alphaearth", AE_DIR / "outputs"),
    ("E4-rand S-2", "E4_rand_A3_s2_9band", AE_DIR / "outputs"),
    ("E5 MLP sparse AE", "E5_D2_alphaearth_mlp_sparse", AE_DIR / "outputs"),
]

RQ3_RF = [
    ("E1 RF balanced AE", "E1_ae_rf_sparse"),
    ("E1-rand RF", "E1_rand_ae_rf_sparse"),
    ("E2 RF balanced S-2", "E2_s2_rf_sparse"),
]


def main():
    rows = []

    # RQ1
    for label, cond in RQ1:
        r = load_unet_part1(RQ1_TEST_EVAL, cond)
        if r is None:
            continue
        rows.append(("RQ1", label, cond, r["ensemble_mean"], r["fold_std"], r["n_tiles"], r["n_folds"]))

    # RQ2
    for label, exp in RQ2_UNET:
        r = load_unet_p2(PART2 / "outputs" / "experiments" / exp)
        if r is None:
            continue
        rows.append(("RQ2", label, exp, r["ensemble_mean"], r["fold_std"], r["n_tiles"], r["n_folds"]))

    # RQ3 U-Net
    for label, exp, base in RQ3_UNET:
        r = load_unet_p2(base / exp)
        if r is None:
            print(f"  RQ3 {label}: MISSING")
            continue
        rows.append(("RQ3", label, exp, r["ensemble_mean"], r["fold_std"], r["n_tiles"], r["n_folds"]))

    # RQ3 RF
    for label, exp in RQ3_RF:
        r = load_rf(AE_DIR / "outputs" / exp)
        if r is None:
            print(f"  RQ3 RF {label}: MISSING")
            continue
        rows.append(("RQ3", label, exp, r["ensemble_mean"], r["fold_std"], r["n_tiles"], r["n_folds"]))

    # Print
    print(f"\n{'Chapter':<8} {'Label':<32} {'Experiment':<35} {'mean %':<8} {'std %':<8} {'n_tiles':<8} {'n_folds':<8}")
    print("-" * 110)
    for ch, lbl, exp, m, s, n, f in rows:
        mean_str = f"{m*100:.2f}"
        std_str = f"{s*100:.2f}" if s is not None else "N/A"
        n_str = str(n) if n is not None else "-"
        print(f"{ch:<8} {lbl:<32} {exp:<35} {mean_str:<8} {std_str:<8} {n_str:<8} {f:<8}")

    # Write CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f_out:
        w = csv.writer(f_out)
        w.writerow(["chapter", "label", "experiment", "ensemble_mean", "fold_std", "n_tiles", "n_folds"])
        for ch, lbl, exp, m, s, n, fc in rows:
            w.writerow([ch, lbl, exp,
                        f"{m:.6f}",
                        f"{s:.6f}" if s is not None else "",
                        n if n is not None else "",
                        fc])
    print(f"\nSaved {OUT_CSV}")


if __name__ == "__main__":
    main()
