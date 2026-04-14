#!/usr/bin/env python3
"""
Stratified analysis by change level (low/moderate/high) for Part 2 test results.

Uses pre-computed per-tile IoU from statistical_analysis_persample.py output
and change_level metadata from split_info.csv. No GPU needed.

Usage:
    python stratified_by_change_level.py
"""

import json
import csv
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PER_TILE_PATH = REPO_ROOT / "PART2_spectral_spatial_resolution_experiments" / "outputs" / "analysis" / "per_tile_iou_iou.json"
SPLITS_CSV = REPO_ROOT / "preprocessing" / "outputs" / "splits" / "unified" / "split_info.csv"
OUTPUT_DIR = REPO_ROOT / "PART2_spectral_spatial_resolution_experiments" / "outputs" / "analysis"

EXPERIMENTS = [
    "A1_s2_rgb", "A2_s2_rgbnir", "A3_s2_9band", "A4_s2_indices",
    "D2_alphaearth", "E4_ae_unet_sparse", "E4_A3_s2_9band_sparse",
    "E1_ae_rf_sparse", "E2_s2_rf_sparse",
]


def main():
    with open(PER_TILE_PATH) as f:
        per_tile = json.load(f)

    meta = {}
    with open(SPLITS_CSV) as f:
        for row in csv.DictReader(f):
            meta[row["refid"]] = row

    # Get test tile IDs from first available experiment
    first_exp = next(e for e in EXPERIMENTS if e in per_tile)
    test_refids = list(per_tile[first_exp].keys())

    results = {}

    for level in ["low", "moderate", "high"]:
        subset = [r for r in test_refids if meta.get(r, {}).get("change_level") == level]
        n = len(subset)
        print(f"=== {level.upper()} change (n={n}) ===")

        level_results = {}
        for exp in EXPERIMENTS:
            if exp not in per_tile:
                continue
            vals = np.array([per_tile[exp][r] for r in subset if r in per_tile[exp]])
            if len(vals) == 0:
                continue
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            print(f"  {exp:28s} {mean*100:5.1f}% ± {std*100:4.1f}%")
            level_results[exp] = {"mean": mean, "std": std, "n": len(vals),
                                   "values": vals.tolist()}

        results[level] = {"n_tiles": n, "experiments": level_results}
        print()

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "stratified_by_change_level.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
