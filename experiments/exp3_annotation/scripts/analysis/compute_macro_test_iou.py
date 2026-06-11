#!/usr/bin/env python3
"""
Compute per-tile macro Test IoU for every experiment in RQ1, RQ2, RQ3.

Single source of truth for the "Test IoU" column in the body chapters.
Outputs a CSV `outputs/macro_iou_summary.csv` with columns
    chapter, experiment, n_tiles, micro_iou, macro_iou.

Sources:
  - RQ1 (Part 1, full-window 25-tile test set): per-condition entries in
    PART1/outputs_v3/analysis/test_set_evaluation.json (already provides
    per_tile.iou.mean and a micro field).
  - RQ2 / RQ3 U-Net (Part 2, clamped 40-tile test set):
    PART2/outputs/experiments/<exp>/test_results.json
    (ensemble.aggregated.iou for micro, ensemble.per_sample for macro mean).
  - RQ3 RF (annotation_efficiency): per-fold metrics.json (test_metrics).

The CSV is the data backing the appendix table that shows both metrics.
"""

import csv
import json
from pathlib import Path

import numpy as np

from landtake.paths import REPO_ROOT as REPO
PART1 = REPO / "experiments/exp1_temporal_sampling"
PART2 = REPO / "experiments/exp2_input_representation"
AE_DIR = REPO / "experiments" / "exp3_annotation"

OUT_CSV = AE_DIR / "outputs" / "macro_iou_summary.csv"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_unet_p2(exp_dir: Path):
    """U-Net experiments evaluated by PART2's evaluate_test_set.py."""
    p = exp_dir / "test_results.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    micro = d["ensemble"]["aggregated"]["iou"]
    macros = [v["iou"] for v in d["ensemble"]["per_sample"].values()]
    return {
        "n_tiles": len(macros),
        "micro_iou": float(micro),
        "macro_iou": float(np.mean(macros)),
    }


def load_unet_part1(test_eval_path: Path, condition: str):
    """RQ1 conditions from PART1/outputs_v3/analysis/test_set_evaluation.json."""
    d = json.load(open(test_eval_path))
    if condition not in d:
        return None
    entry = d[condition]
    macro = entry["per_tile"]["iou"]["mean"]
    micro_field = entry.get("micro", None)
    if isinstance(micro_field, dict):
        micro = micro_field.get("iou")
        if isinstance(micro, dict):
            micro = micro.get("mean")
    else:
        micro = micro_field
    return {
        "n_tiles": entry["n_test_tiles"],
        "micro_iou": float(micro) if isinstance(micro, (int, float)) else None,
        "macro_iou": float(macro),
    }


def load_rf(rf_dir: Path):
    """RF experiments: average per-fold per-tile IoU across folds."""
    tile_values = {}
    folds = 0
    for fold in range(5):
        path = rf_dir / f"fold{fold}" / "metrics.json"
        if not path.exists():
            continue
        d = json.load(open(path))
        per_tile = d.get("test_metrics", {}).get("per_tile", {})
        if not per_tile:
            continue
        folds += 1
        for tid, m in per_tile.items():
            tile_values.setdefault(tid, []).append(m["iou"])
    if folds == 0:
        return None
    macro = np.mean([np.mean(v) for v in tile_values.values()])
    # micro: pool TP/FP/FN across tiles per fold, then average
    fold_micros = []
    for fold in range(5):
        path = rf_dir / f"fold{fold}" / "metrics.json"
        if not path.exists():
            continue
        d = json.load(open(path))
        fold_micros.append(d["test_metrics"]["micro_iou"])
    return {
        "n_tiles": len(tile_values),
        "micro_iou": float(np.mean(fold_micros)),
        "macro_iou": float(macro),
    }


# ---------------------------------------------------------------------------
# Experiment registry
# ---------------------------------------------------------------------------

# RQ1 condition → chapter label mapping
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
RQ1_TEST_EVAL = PART1 / "outputs_v3" / "analysis" / "test_set_evaluation.json"

# RQ2 / RQ3 U-Net experiments
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
    ("E4 sparse AE (n100/200pts)", "E4_D2_alphaearth_sparse_n100", AE_DIR / "outputs"),
    ("E4-rand AE", "E4_rand_D2_alphaearth", AE_DIR / "outputs"),
    ("E4-rand S-2", "E4_rand_A3_s2_9band", AE_DIR / "outputs"),
]

RQ3_RF = [
    ("E1 balanced AE", "E1_ae_rf_sparse"),
    ("E1-rand AE", "E1_rand_ae_rf_sparse"),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rows = []

    # RQ1
    for label, condition in RQ1:
        r = load_unet_part1(RQ1_TEST_EVAL, condition)
        if r is None:
            print(f"  [RQ1] {label}: MISSING")
            continue
        rows.append(("RQ1", label, condition, r["n_tiles"], r["micro_iou"], r["macro_iou"]))

    # RQ2
    for label, exp in RQ2_UNET:
        r = load_unet_p2(PART2 / "outputs" / "experiments" / exp)
        if r is None:
            print(f"  [RQ2] {label}: MISSING")
            continue
        rows.append(("RQ2", label, exp, r["n_tiles"], r["micro_iou"], r["macro_iou"]))

    # RQ3 U-Nets
    for label, exp, base in RQ3_UNET:
        r = load_unet_p2(base / exp)
        if r is None:
            print(f"  [RQ3] {label}: MISSING")
            continue
        rows.append(("RQ3", label, exp, r["n_tiles"], r["micro_iou"], r["macro_iou"]))

    # RQ3 RFs
    for label, exp in RQ3_RF:
        r = load_rf(AE_DIR / "outputs" / exp)
        if r is None:
            print(f"  [RQ3] {label}: MISSING")
            continue
        rows.append(("RQ3", label, exp, r["n_tiles"], r["micro_iou"], r["macro_iou"]))

    # Print
    print(f"\n{'Chapter':<8} {'Label':<32} {'Experiment':<35} {'n':<4} {'micro %':<10} {'macro %':<10}")
    print("-" * 100)
    for chapter, label, exp, n, micro, macro in rows:
        m = f"{micro*100:.2f}" if micro is not None else "N/A"
        print(f"{chapter:<8} {label:<32} {exp:<35} {n:<4} {m:<10} {macro*100:<10.2f}")

    # Write CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["chapter", "label", "experiment", "n_tiles", "micro_iou", "macro_iou"])
        for chapter, label, exp, n, micro, macro in rows:
            w.writerow([chapter, label, exp, n,
                        f"{micro:.6f}" if micro is not None else "",
                        f"{macro:.6f}"])
    print(f"\nSaved {OUT_CSV}")


if __name__ == "__main__":
    main()
