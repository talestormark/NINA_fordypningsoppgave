#!/usr/bin/env python3
"""
Aggregate CV-macro recompute outputs into a single comparison CSV.

Reads per-experiment cv_macro_summary.json files (or cv_summary.json for RF),
the historical macro_iou_summary.csv (which has Test macro / micro per
experiment), and produces one CSV with one row per experiment:

  chapter,label,experiment,n_val_per_fold,
  cv_macro_mean,cv_macro_std,
  cv_micro_mean,cv_micro_std,
  test_macro,test_micro,
  delta_cv_micro_vs_macro,
  delta_cv_macro_vs_test_macro
"""

import json
import csv
from pathlib import Path
from statistics import mean, pstdev

REPO_ROOT = Path(__file__).resolve().parents[2]
PART1_DIR = REPO_ROOT / "experiments/exp1_temporal_sampling"
PART2_DIR = REPO_ROOT / "experiments/exp2_input_representation"
AE_DIR = REPO_ROOT / "experiments" / "exp3_annotation"

OUT_PATH = REPO_ROOT / "outputs" / "analysis" / "cv_macro_recompute.csv"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Test-set values (already computed): chapter, label, experiment, n_tiles, micro, macro
TEST_CSV = AE_DIR / "outputs" / "macro_iou_summary.csv"


def load_cv_macro_json(path: Path):
    """Returns (cv_macro_mean, cv_macro_std, cv_micro_mean, cv_micro_std)."""
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    cv_macro_mean = d["cv_macro_mean"]
    cv_macro_std = d["cv_macro_std"]
    micros = [v["iou"] for v in d["per_fold_aggregated_micro"].values()]
    cv_micro_mean = mean(micros) if micros else 0.0
    cv_micro_std = pstdev(micros) if len(micros) > 1 else 0.0
    return cv_macro_mean, cv_macro_std, cv_micro_mean, cv_micro_std


def load_rf_cv_summary(path: Path):
    """RF cv_summary.json has only val IoU (macro). Returns same 4-tuple."""
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    return (
        d["cv_val_iou_mean"],
        d["cv_val_iou_std"],
        None,  # micro not recorded for RF
        None,
    )


def load_test_values():
    """Returns {experiment: (chapter, label, n_tiles, test_micro, test_macro)}."""
    result = {}
    with open(TEST_CSV) as f:
        for row in csv.DictReader(f):
            exp = row["experiment"]
            result[exp] = (
                row["chapter"],
                row["label"],
                int(row["n_tiles"]),
                float(row["micro_iou"]),
                float(row["macro_iou"]),
            )
    return result


# ---------------------------------------------------------------------------
# Experiment locations
# ---------------------------------------------------------------------------

# Part 1 — use temporal_sampling name for join with test CSV
PART1_EXPS = {
    "bi_temporal": ("exp003", PART1_DIR / "outputs_v3" / "exp003" / "cv_macro_summary.json"),
    "annual": ("exp001", PART1_DIR / "outputs_v3" / "exp001" / "cv_macro_summary.json"),
    "bi_seasonal": ("exp002", PART1_DIR / "outputs_v3" / "exp002" / "cv_macro_summary.json"),
    "early_fusion": ("exp005", PART1_DIR / "outputs_v3" / "exp005" / "cv_macro_summary.json"),
    "late_fusion": ("exp006", PART1_DIR / "outputs_v3" / "exp006" / "cv_macro_summary.json"),
    "late_fusion_pool": ("exp007", PART1_DIR / "outputs_v3" / "exp007" / "cv_macro_summary.json"),
    "conv3d_fusion": ("exp008", PART1_DIR / "outputs_v3" / "exp008" / "cv_macro_summary.json"),
    "lstm_lite": ("exp009", PART1_DIR / "outputs_v3" / "exp009" / "cv_macro_summary.json"),
    "lstm7_lite": ("exp010", PART1_DIR / "outputs_v3" / "exp010" / "cv_macro_summary.json"),
}

# Part 2 — direct match to test CSV experiment column
PART2_EXPS = ["A1_s2_rgb", "A2_s2_rgbnir", "A3_s2_9band", "A4_s2_indices", "D2_alphaearth"]

# Annotation efficiency U-Net
AE_UNET_EXPS = [
    "E4_ae_unet_sparse",
    "E4_A3_s2_9band_sparse",
    "E4_D2_alphaearth_sparse_n5",
    "E4_D2_alphaearth_sparse_n10",
    "E4_D2_alphaearth_sparse_n15",
    "E4_D2_alphaearth_sparse_n100",
    "E4_rand_D2_alphaearth",
    "E4_rand_A3_s2_9band",
    "E5_D2_alphaearth_mlp_sparse",
]

# Annotation efficiency RF
AE_RF_EXPS = ["E1_ae_rf_sparse", "E1_rand_ae_rf_sparse"]


def main():
    test_lookup = load_test_values()
    rows = []

    # Part 1
    for test_key, (exp_id, json_path) in PART1_EXPS.items():
        cv = load_cv_macro_json(json_path)
        if cv is None:
            print(f"  Missing Part 1 file: {json_path}")
            continue
        test_info = test_lookup.get(test_key)
        chapter = test_info[0] if test_info else "RQ1"
        label = test_info[1] if test_info else exp_id
        n_test = test_info[2] if test_info else None
        test_micro = test_info[3] if test_info else None
        test_macro = test_info[4] if test_info else None
        rows.append({
            "chapter": chapter, "label": label, "experiment": exp_id,
            "source": "Part1",
            "n_val_per_fold": 28,
            "cv_macro_mean": cv[0], "cv_macro_std": cv[1],
            "cv_micro_mean": cv[2], "cv_micro_std": cv[3],
            "n_test": n_test, "test_macro": test_macro, "test_micro": test_micro,
        })

    # Part 2
    for exp in PART2_EXPS:
        json_path = PART2_DIR / "outputs" / "experiments" / exp / "cv_macro_summary.json"
        cv = load_cv_macro_json(json_path)
        if cv is None:
            print(f"  Missing Part 2 file: {json_path}")
            continue
        test_info = test_lookup.get(exp)
        chapter = test_info[0] if test_info else "RQ2"
        label = test_info[1] if test_info else exp
        n_test = test_info[2] if test_info else None
        test_micro = test_info[3] if test_info else None
        test_macro = test_info[4] if test_info else None
        rows.append({
            "chapter": chapter, "label": label, "experiment": exp,
            "source": "Part2",
            "n_val_per_fold": 44,
            "cv_macro_mean": cv[0], "cv_macro_std": cv[1],
            "cv_micro_mean": cv[2], "cv_micro_std": cv[3],
            "n_test": n_test, "test_macro": test_macro, "test_micro": test_micro,
        })

    # Annotation efficiency U-Net
    for exp in AE_UNET_EXPS:
        json_path = AE_DIR / "outputs" / exp / "cv_macro_summary.json"
        cv = load_cv_macro_json(json_path)
        if cv is None:
            print(f"  Missing AE U-Net file: {json_path}")
            continue
        test_info = test_lookup.get(exp)
        chapter = test_info[0] if test_info else "RQ3"
        label = test_info[1] if test_info else exp
        n_test = test_info[2] if test_info else None
        test_micro = test_info[3] if test_info else None
        test_macro = test_info[4] if test_info else None
        rows.append({
            "chapter": chapter, "label": label, "experiment": exp,
            "source": "AE-UNet",
            "n_val_per_fold": 44,
            "cv_macro_mean": cv[0], "cv_macro_std": cv[1],
            "cv_micro_mean": cv[2], "cv_micro_std": cv[3],
            "n_test": n_test, "test_macro": test_macro, "test_micro": test_micro,
        })

    # Annotation efficiency RF
    for exp in AE_RF_EXPS:
        json_path = AE_DIR / "outputs" / exp / "cv_summary.json"
        cv = load_rf_cv_summary(json_path)
        if cv is None:
            print(f"  Missing AE RF file: {json_path}")
            continue
        test_info = test_lookup.get(exp)
        chapter = test_info[0] if test_info else "RQ3"
        label = test_info[1] if test_info else exp
        n_test = test_info[2] if test_info else None
        test_micro = test_info[3] if test_info else None
        test_macro = test_info[4] if test_info else None
        rows.append({
            "chapter": chapter, "label": label, "experiment": exp,
            "source": "AE-RF",
            "n_val_per_fold": 44,
            "cv_macro_mean": cv[0], "cv_macro_std": cv[1],
            "cv_micro_mean": cv[2], "cv_micro_std": cv[3],
            "n_test": n_test, "test_macro": test_macro, "test_micro": test_micro,
        })

    # Compute deltas
    for r in rows:
        if r["cv_micro_mean"] is not None:
            r["delta_micro_minus_macro_pp"] = (r["cv_micro_mean"] - r["cv_macro_mean"]) * 100
        else:
            r["delta_micro_minus_macro_pp"] = None
        if r["test_macro"] is not None:
            r["delta_cv_macro_minus_test_macro_pp"] = (r["cv_macro_mean"] - r["test_macro"]) * 100
        else:
            r["delta_cv_macro_minus_test_macro_pp"] = None

    # Sort: chapter, then label
    chapter_order = {"RQ1": 0, "RQ2": 1, "RQ3": 2}
    rows.sort(key=lambda r: (chapter_order.get(r["chapter"], 9), r["label"]))

    # Write CSV
    fieldnames = [
        "chapter", "label", "experiment", "source",
        "n_val_per_fold", "n_test",
        "cv_macro_mean", "cv_macro_std",
        "cv_micro_mean", "cv_micro_std",
        "test_macro", "test_micro",
        "delta_micro_minus_macro_pp",
        "delta_cv_macro_minus_test_macro_pp",
    ]
    with open(OUT_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            row_out = {k: r.get(k) for k in fieldnames}
            for k in ("cv_macro_mean", "cv_macro_std", "cv_micro_mean", "cv_micro_std",
                      "test_macro", "test_micro"):
                v = row_out[k]
                if v is not None:
                    row_out[k] = f"{v*100:.2f}"
            for k in ("delta_micro_minus_macro_pp", "delta_cv_macro_minus_test_macro_pp"):
                v = row_out[k]
                if v is not None:
                    row_out[k] = f"{v:+.2f}"
            w.writerow(row_out)

    print(f"\nWrote {OUT_PATH}")
    print(f"  {len(rows)} experiments aggregated")

    # Print pretty table
    print("\n" + "=" * 100)
    print(f"{'Ch':<4}{'Label':<22}{'Experiment':<32}"
          f"{'CV mac':>9}{'CV mic':>9}{'Tst mac':>9}"
          f"{'mic-mac':>10}{'CV-Tst':>9}")
    print("-" * 100)
    for r in rows:
        cv_mac = f"{r['cv_macro_mean']*100:.2f}" if r["cv_macro_mean"] is not None else "—"
        cv_mic = f"{r['cv_micro_mean']*100:.2f}" if r["cv_micro_mean"] is not None else "—"
        tst_mac = f"{r['test_macro']*100:.2f}" if r["test_macro"] is not None else "—"
        d1 = f"{r['delta_micro_minus_macro_pp']:+.2f}" if r["delta_micro_minus_macro_pp"] is not None else "—"
        d2 = f"{r['delta_cv_macro_minus_test_macro_pp']:+.2f}" if r["delta_cv_macro_minus_test_macro_pp"] is not None else "—"
        print(f"{r['chapter']:<4}{r['label']:<22}{r['experiment']:<32}"
              f"{cv_mac:>9}{cv_mic:>9}{tst_mac:>9}{d1:>10}{d2:>9}")


if __name__ == "__main__":
    main()
