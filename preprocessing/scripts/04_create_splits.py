#!/usr/bin/env python3
"""
Create stratified train/val/test splits for data_v2.

Generates two split sets:
- Part 1 (temporal experiments): 163 full-window tiles only (startYear<=2018, endYear>=2024)
- Part 2 (spectral/spatial experiments): all 260 trainable tiles

Both preserve v1 test tiles and stratify by change_level.

Usage:
    python preprocessing/scripts/04_create_splits.py
"""

import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_V2 = PROJECT_ROOT / "data_v2"
MASK_ANALYSIS_CSV = PROJECT_ROOT / "preprocessing" / "outputs" / "mask_analysis.csv"
INVENTORY_CSV = PROJECT_ROOT / "preprocessing" / "outputs" / "data_inventory.csv"
YEAR_RANGE_CSV = PROJECT_ROOT / "preprocessing" / "outputs" / "year_range_analysis.csv"
GEOJSON_PATH = DATA_V2 / "land_take_bboxes_650m_v1_filtered.geojson"
V1_SPLITS_DIR = PROJECT_ROOT / "outputs_v1" / "splits"
OUTPUT_BASE = PROJECT_ROOT / "preprocessing" / "outputs" / "splits"

RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

LOW_THRESHOLD = 5.0
HIGH_THRESHOLD = 30.0


def load_v1_test_refids() -> list:
    """Load the 8 v1 test REFIDs."""
    path = V1_SPLITS_DIR / "test_refids.txt"
    if not path.exists():
        print(f"  WARNING: v1 test refids not found at {path}")
        return []
    return [line.strip() for line in path.read_text().strip().splitlines() if line.strip()]


def load_geojson_metadata(path: Path) -> dict:
    """Load country and change_type from filtered geojson."""
    if not path.exists():
        print(f"  WARNING: GeoJSON not found at {path}")
        return {}
    with open(path) as f:
        data = json.load(f)
    metadata = {}
    for feature in data["features"]:
        props = feature["properties"]
        plotid = props.get("PLOTID", "")
        if plotid and plotid not in metadata:
            metadata[plotid] = {
                "country": props.get("country", ""),
                "change_type": props.get("change_type", ""),
            }
    return metadata


def categorize_change_level(ratio):
    if ratio < LOW_THRESHOLD:
        return "low"
    elif ratio < HIGH_THRESHOLD:
        return "moderate"
    else:
        return "high"


def stratified_split(df, train_ratio, val_ratio, seed):
    """Stratified split by change_level. Returns train_df, val_df, test_df."""
    np.random.seed(seed)

    train_parts, val_parts, test_parts = [], [], []

    for level in ["low", "moderate", "high"]:
        level_df = df[df["change_level"] == level].copy()
        n = len(level_df)
        if n == 0:
            continue

        n_train = int(np.round(n * train_ratio))
        n_val = int(np.round(n * val_ratio))

        shuffled = level_df.sample(frac=1, random_state=seed).reset_index(drop=True)
        train_parts.append(shuffled.iloc[:n_train])
        val_parts.append(shuffled.iloc[n_train : n_train + n_val])
        test_parts.append(shuffled.iloc[n_train + n_val :])

    return (
        pd.concat(train_parts, ignore_index=True),
        pd.concat(val_parts, ignore_index=True),
        pd.concat(test_parts, ignore_index=True),
    )


def generate_splits(coarse_df, v1_test_available, geo_meta, output_dir, label):
    """Generate and save splits for a given tile set."""
    print(f"\n{'=' * 70}")
    print(f"GENERATING SPLITS: {label}")
    print(f"{'=' * 70}")
    print(f"  Total tiles: {len(coarse_df)}")

    # Separate locked test tiles from rest
    locked_test = coarse_df[coarse_df["refid"].isin(v1_test_available)].copy()
    remaining = coarse_df[~coarse_df["refid"].isin(v1_test_available)].copy()
    print(f"  Locked test tiles (v1): {len(locked_test)}")
    print(f"  Remaining for stratified split: {len(remaining)}")

    # Stratified split on remaining
    train_df, val_df, new_test_df = stratified_split(
        remaining, TRAIN_RATIO, VAL_RATIO, RANDOM_SEED
    )

    # Combine test sets
    test_df = pd.concat([locked_test, new_test_df], ignore_index=True)

    # Print stats
    for split_name, split_df in [("TRAIN", train_df), ("VAL", val_df), ("TEST", test_df)]:
        print(f"\n  {split_name}: {len(split_df)} tiles")
        print(f"    Change ratio: mean={split_df['change_ratio'].mean():.2f}%, "
              f"median={split_df['change_ratio'].median():.2f}%")
        for level in ["low", "moderate", "high"]:
            count = (split_df["change_level"] == level).sum()
            print(f"    {level}: {count}")

    total = len(train_df) + len(val_df) + len(test_df)
    print(f"\n  Total: {total}")
    print(f"  Ratios: train={len(train_df)/total:.1%}, "
          f"val={len(val_df)/total:.1%}, test={len(test_df)/total:.1%}")

    # Build output
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    # Mark v1 test tiles
    test_df["v1_test"] = test_df["refid"].isin(v1_test_available)

    all_splits = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # Enrich with geojson metadata
    all_splits["country"] = all_splits["refid"].map(
        lambda r: geo_meta.get(r, {}).get("country", "")
    )
    all_splits["change_type"] = all_splits["refid"].map(
        lambda r: geo_meta.get(r, {}).get("change_type", "")
    )

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        path = output_dir / f"{split_name}_refids.txt"
        split_df["refid"].to_csv(path, index=False, header=False)
        print(f"  Saved {path.name}: {len(split_df)} tiles")

    out_cols = ["refid", "split", "change_ratio", "change_level", "country", "change_type"]
    if "v1_test" in all_splits.columns:
        out_cols.append("v1_test")
    split_info_path = output_dir / "split_info.csv"
    all_splits[out_cols].to_csv(split_info_path, index=False)
    print(f"  Saved {split_info_path.name}")

    # Verify no leakage
    train_set = set(train_df["refid"])
    val_set = set(val_df["refid"])
    test_set = set(test_df["refid"])
    assert len(train_set & val_set) == 0, "Train/val overlap!"
    assert len(train_set & test_set) == 0, "Train/test overlap!"
    assert len(val_set & test_set) == 0, "Val/test overlap!"
    print("  No data leakage detected.")

    for r in v1_test_available:
        if r in set(coarse_df["refid"]):
            assert r in test_set, f"V1 test tile {r} not in test set!"
    v1_in_test = sum(1 for r in v1_test_available if r in test_set)
    print(f"  {v1_in_test} v1 test tiles preserved in test set.")

    return train_df, val_df, test_df


def main():
    print("=" * 70)
    print("CREATE SPLITS -- data_v2")
    print("=" * 70)

    # Load mask analysis (coarse masks only)
    mask_df = pd.read_csv(MASK_ANALYSIS_CSV)
    coarse = mask_df[(mask_df["mask_type"] == "coarse") & (mask_df["success"] == True)].copy()
    print(f"\nCoarse masks with successful analysis: {len(coarse)}")

    # Load inventory to check S2 availability
    inv_df = pd.read_csv(INVENTORY_CSV)
    trainable = set(inv_df[inv_df["has_s2"] & inv_df["has_mask_coarse"]]["refid"].values)
    print(f"Tiles with S2 + coarse mask (trainable): {len(trainable)}")

    # Keep only trainable tiles
    coarse = coarse[coarse["refid"].isin(trainable)].copy()
    coarse["change_level"] = coarse["change_ratio"].apply(categorize_change_level)
    print(f"Tiles for splitting: {len(coarse)}")

    # Load year range analysis for full-window filtering
    year_df = pd.read_csv(YEAR_RANGE_CSV)
    full_window_refids = set(year_df[year_df["full_window"] == True]["refid"].values)
    print(f"Full-window tiles (startYear<=2018, endYear>=2024): {len(full_window_refids)}")

    # Load v1 test REFIDs
    v1_test = load_v1_test_refids()
    v1_test_available = [r for r in v1_test if r in set(coarse["refid"].values)]
    v1_test_missing = [r for r in v1_test if r not in set(coarse["refid"].values)]
    print(f"\nV1 test tiles: {len(v1_test)} total, {len(v1_test_available)} available in v2")
    if v1_test_missing:
        print(f"  Missing v1 test tiles:")
        for r in v1_test_missing:
            print(f"    - {r}")

    # Check how many v1 test tiles are full-window
    v1_test_full_window = [r for r in v1_test_available if r in full_window_refids]
    v1_test_not_full = [r for r in v1_test_available if r not in full_window_refids]
    print(f"  V1 test tiles with full window: {len(v1_test_full_window)}/{len(v1_test_available)}")
    if v1_test_not_full:
        print(f"  V1 test tiles WITHOUT full window:")
        for r in v1_test_not_full:
            row = year_df[year_df["refid"] == r]
            if len(row) > 0:
                print(f"    - {r} (startYear={int(row.iloc[0]['startYear'])}, "
                      f"endYear={int(row.iloc[0]['endYear'])})")

    # Load geojson metadata
    geo_meta = load_geojson_metadata(GEOJSON_PATH)

    # -------------------------------------------------------------------------
    # Part 1 splits: full-window tiles only (163 tiles)
    # -------------------------------------------------------------------------
    coarse_p1 = coarse[coarse["refid"].isin(full_window_refids)].copy()
    p1_train, p1_val, p1_test = generate_splits(
        coarse_p1, v1_test_full_window, geo_meta,
        OUTPUT_BASE / "part1", "Part 1 (full-window, 2018-2024)"
    )

    # -------------------------------------------------------------------------
    # Part 2 splits: all trainable tiles (260 tiles)
    # -------------------------------------------------------------------------
    p2_train, p2_val, p2_test = generate_splits(
        coarse, v1_test_available, geo_meta,
        OUTPUT_BASE / "part2", "Part 2 (all trainable)"
    )

    # -------------------------------------------------------------------------
    # Cross-check: Part 1 splits should be a subset of Part 2 splits
    # -------------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("CROSS-CHECK")
    print(f"{'=' * 70}")
    p1_test_set = set(p1_test["refid"])
    p2_test_set = set(p2_test["refid"])
    p1_in_p2_test = p1_test_set & p2_test_set
    p1_test_not_in_p2_test = p1_test_set - p2_test_set
    print(f"  Part 1 test tiles also in Part 2 test: {len(p1_in_p2_test)}/{len(p1_test_set)}")
    if p1_test_not_in_p2_test:
        print(f"  WARNING: {len(p1_test_not_in_p2_test)} Part 1 test tiles are NOT in Part 2 test set")
        print(f"  (This is expected since splits are generated independently)")

    print(f"\n{'=' * 70}")
    print("Done!")


if __name__ == "__main__":
    main()
