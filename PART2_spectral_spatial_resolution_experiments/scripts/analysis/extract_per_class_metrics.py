#!/usr/bin/env python3
"""
Extract per-class IoU, precision, recall, F1 from existing test results.

Handles two formats:
  - U-Net: test_results.json with ensemble.per_sample containing {tp,fp,fn,tn}
  - RF:    fold*/metrics.json with test_metrics.per_tile containing {tp,fp,fn,tn}

For U-Net experiments, uses the ensemble predictions (averaged probabilities across folds).
For RF experiments, averages the per-tile confusion counts across folds, then aggregates.

Usage:
    python extract_per_class_metrics.py
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

# ── Configuration ──────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[3]
PART2_DIR = REPO_ROOT / "PART2_spectral_spatial_resolution_experiments" / "outputs" / "experiments"
ANNOT_DIR = REPO_ROOT / "experiments" / "annotation_efficiency" / "outputs"
OUTPUT_DIR = REPO_ROOT / "PART2_spectral_spatial_resolution_experiments" / "outputs" / "analysis"

# Experiments to process: (name, type, path)
UNET_NAMES = [
    "A1_s2_rgb",
    "A2_s2_rgbnir",
    "A3_s2_9band",
    "A4_s2_indices",
    "D2_alphaearth",
    "E4_ae_unet_sparse",
    "E4_A3_s2_9band_sparse",
]

RF_NAMES = [
    "E1_ae_rf_sparse",
    "E2_s2_rf_sparse",
]


# ── Metric computation ─────────────────────────────────────────────────────

def compute_metrics_from_counts(tp, fp, fn, tn):
    """Compute per-class and summary metrics from aggregated confusion counts."""
    change_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    change_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    change_f1 = (2 * change_precision * change_recall /
                 (change_precision + change_recall)
                 if (change_precision + change_recall) > 0 else 0.0)
    change_iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    nochange_precision = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    nochange_recall = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    nochange_f1 = (2 * nochange_precision * nochange_recall /
                   (nochange_precision + nochange_recall)
                   if (nochange_precision + nochange_recall) > 0 else 0.0)
    nochange_iou = tn / (tn + fp + fn) if (tn + fp + fn) > 0 else 0.0

    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total > 0 else 0.0

    return {
        "change_iou": change_iou,
        "change_precision": change_precision,
        "change_recall": change_recall,
        "change_f1": change_f1,
        "nochange_iou": nochange_iou,
        "nochange_precision": nochange_precision,
        "nochange_recall": nochange_recall,
        "nochange_f1": nochange_f1,
        "accuracy": accuracy,
    }


def compute_per_tile_iou(tile_data):
    tp, fp, fn = tile_data["tp"], tile_data["fp"], tile_data["fn"]
    denom = tp + fp + fn
    return tp / denom if denom > 0 else 0.0


def compute_positive_ratio(tile_data):
    tp, fp, fn, tn = tile_data["tp"], tile_data["fp"], tile_data["fn"], tile_data["tn"]
    total = tp + fp + fn + tn
    return (tp + fn) / total if total > 0 else 0.0


def process_unet(path):
    """Process a U-Net test_results.json using the ensemble section."""
    with open(path) as f:
        data = json.load(f)

    per_sample = data["ensemble"]["per_sample"]

    total_tp = sum(t["tp"] for t in per_sample.values())
    total_fp = sum(t["fp"] for t in per_sample.values())
    total_fn = sum(t["fn"] for t in per_sample.values())
    total_tn = sum(t["tn"] for t in per_sample.values())

    metrics = compute_metrics_from_counts(total_tp, total_fp, total_fn, total_tn)

    per_tile_ious = [compute_per_tile_iou(t) for t in per_sample.values()]
    metrics["macro_change_iou"] = sum(per_tile_ious) / len(per_tile_ious)

    pos_ratios = [compute_positive_ratio(t) for t in per_sample.values()]
    metrics["mean_positive_ratio"] = sum(pos_ratios) / len(pos_ratios)
    metrics["min_positive_ratio"] = min(pos_ratios)
    metrics["max_positive_ratio"] = max(pos_ratios)
    metrics["n_tiles"] = len(per_sample)

    per_tile_detail = {}
    for tile_id, t in per_sample.items():
        per_tile_detail[tile_id] = {
            "change_iou": compute_per_tile_iou(t),
            "positive_ratio": compute_positive_ratio(t),
            "tp": t["tp"], "fp": t["fp"], "fn": t["fn"], "tn": t["tn"],
        }
    metrics["per_tile"] = per_tile_detail
    return metrics


def process_rf(exp_dir):
    """Process an RF experiment with fold*/metrics.json, averaging across folds."""
    tile_counts = defaultdict(lambda: {"tp": [], "fp": [], "fn": [], "tn": []})
    n_folds = 0

    for fold_dir in sorted(exp_dir.glob("fold*")):
        metrics_path = fold_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        with open(metrics_path) as f:
            fold_data = json.load(f)
        if "test_metrics" not in fold_data:
            continue
        n_folds += 1
        for tile_id, t in fold_data["test_metrics"]["per_tile"].items():
            tile_counts[tile_id]["tp"].append(t["tp"])
            tile_counts[tile_id]["fp"].append(t["fp"])
            tile_counts[tile_id]["fn"].append(t["fn"])
            tile_counts[tile_id]["tn"].append(t["tn"])

    averaged_tiles = {}
    for tile_id, counts in tile_counts.items():
        averaged_tiles[tile_id] = {
            "tp": sum(counts["tp"]) / len(counts["tp"]),
            "fp": sum(counts["fp"]) / len(counts["fp"]),
            "fn": sum(counts["fn"]) / len(counts["fn"]),
            "tn": sum(counts["tn"]) / len(counts["tn"]),
        }

    total_tp = sum(t["tp"] for t in averaged_tiles.values())
    total_fp = sum(t["fp"] for t in averaged_tiles.values())
    total_fn = sum(t["fn"] for t in averaged_tiles.values())
    total_tn = sum(t["tn"] for t in averaged_tiles.values())

    metrics = compute_metrics_from_counts(total_tp, total_fp, total_fn, total_tn)

    per_tile_ious = [compute_per_tile_iou(t) for t in averaged_tiles.values()]
    metrics["macro_change_iou"] = sum(per_tile_ious) / len(per_tile_ious)

    pos_ratios = [compute_positive_ratio(t) for t in averaged_tiles.values()]
    metrics["mean_positive_ratio"] = sum(pos_ratios) / len(pos_ratios)
    metrics["min_positive_ratio"] = min(pos_ratios)
    metrics["max_positive_ratio"] = max(pos_ratios)
    metrics["n_tiles"] = len(averaged_tiles)
    metrics["n_folds_averaged"] = n_folds

    per_tile_detail = {}
    for tile_id, t in averaged_tiles.items():
        per_tile_detail[tile_id] = {
            "change_iou": compute_per_tile_iou(t),
            "positive_ratio": compute_positive_ratio(t),
            "tp": t["tp"], "fp": t["fp"], "fn": t["fn"], "tn": t["tn"],
        }
    metrics["per_tile"] = per_tile_detail
    return metrics


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    experiments = []
    for name in UNET_NAMES:
        p = PART2_DIR / name / "test_results.json"
        if p.exists():
            experiments.append((name, "unet", p))
        else:
            print(f"WARNING: {p} not found, skipping {name}")
    for name in RF_NAMES:
        d = ANNOT_DIR / name
        if d.exists() and (d / "fold0" / "metrics.json").exists():
            experiments.append((name, "rf", d))
        else:
            print(f"WARNING: {d} not found, skipping {name}")

    results = {}
    for name, exp_type, path in experiments:
        try:
            if exp_type == "unet":
                metrics = process_unet(path)
            else:
                metrics = process_rf(path)
            metrics["type"] = exp_type
            results[name] = metrics
        except Exception as e:
            print(f"ERROR processing {name}: {e}")

    # ── Print summary table ─────────────────────────────────────────────
    print()
    header = (f"{'Experiment':<28s} "
              f"{'ChgIoU':>7s} {'Prec':>7s} {'Rec':>7s} {'F1':>7s} "
              f"{'NChgIoU':>7s} {'MacroIoU':>8s} "
              f"{'PosRatio':>8s} {'Min':>6s} {'Max':>6s}")
    sep = "-" * len(header)
    print(header)
    print(sep)

    for name, _, _ in experiments:
        if name not in results:
            continue
        m = results[name]
        print(f"{name:<28s} "
              f"{m['change_iou']*100:>6.1f}% {m['change_precision']*100:>6.1f}% "
              f"{m['change_recall']*100:>6.1f}% {m['change_f1']*100:>6.1f}% "
              f"{m['nochange_iou']*100:>6.1f}% {m['macro_change_iou']*100:>7.1f}% "
              f"{m['mean_positive_ratio']*100:>7.2f}% "
              f"{m['min_positive_ratio']*100:>5.1f}% "
              f"{m['max_positive_ratio']*100:>5.1f}%")

    print(sep)
    print()

    # ── Save to JSON ────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {"summary": {}, "per_tile_detail": {}}
    for name, m in results.items():
        per_tile = m.pop("per_tile", {})
        output["summary"][name] = m
        output["per_tile_detail"][name] = per_tile

    out_path = OUTPUT_DIR / "per_class_metrics.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
