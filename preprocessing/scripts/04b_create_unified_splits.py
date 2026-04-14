#!/usr/bin/env python3
"""
Create unified stratified train/val/test splits for all experiments.

Two-stage approach:
1. Split the 163 full-window tiles (startYear<=2018, endYear>=2024) at 70/15/15
2. Split the remaining 97 misaligned tiles at 70/15/15

Both Part 1 and Part 2 use the same split assignments. Part 1 filters to
the 163 full-window subset; Part 2 uses all 260.

No v1 test tile locking — clean stratified split only.

Usage:
    python preprocessing/scripts/04b_create_unified_splits.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_V2 = PROJECT_ROOT / "data_v2"
MASK_ANALYSIS_CSV = PROJECT_ROOT / "preprocessing" / "outputs" / "mask_analysis.csv"
INVENTORY_CSV = PROJECT_ROOT / "preprocessing" / "outputs" / "data_inventory.csv"
YEAR_RANGE_CSV = PROJECT_ROOT / "preprocessing" / "outputs" / "year_range_analysis.csv"
ANNOTATION_META_CSV = DATA_V2 / "annotations_metadata_final.csv"
GEOJSON_PATH = DATA_V2 / "land_take_bboxes_650m_v1_filtered.geojson"
OUTPUT_DIR = PROJECT_ROOT / "preprocessing" / "outputs" / "splits" / "unified"

RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

LOW_THRESHOLD = 5.0
HIGH_THRESHOLD = 30.0


def load_geojson_metadata(path: Path) -> dict:
    if not path.exists():
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
    """Pure stratified split by change_level. No locked tiles."""
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
        val_parts.append(shuffled.iloc[n_train:n_train + n_val])
        test_parts.append(shuffled.iloc[n_train + n_val:])

    return (
        pd.concat(train_parts, ignore_index=True),
        pd.concat(val_parts, ignore_index=True),
        pd.concat(test_parts, ignore_index=True),
    )


def main():
    print("=" * 70)
    print("CREATE UNIFIED SPLITS (no v1 lock)")
    print("=" * 70)

    # Load data
    mask_df = pd.read_csv(MASK_ANALYSIS_CSV)
    coarse = mask_df[(mask_df["mask_type"] == "coarse") & (mask_df["success"] == True)].copy()

    inv_df = pd.read_csv(INVENTORY_CSV)
    trainable = set(inv_df[inv_df["has_s2"] & inv_df["has_mask_coarse"]]["refid"].values)
    coarse = coarse[coarse["refid"].isin(trainable)].copy()
    coarse["change_level"] = coarse["change_ratio"].apply(categorize_change_level)
    print(f"Trainable tiles: {len(coarse)}")

    # Full-window filter
    year_df = pd.read_csv(YEAR_RANGE_CSV)
    full_window_refids = set(year_df[year_df["full_window"] == True]["refid"].values)
    # Intersect with trainable
    full_window_refids = full_window_refids & set(coarse["refid"])
    print(f"Full-window trainable tiles: {len(full_window_refids)}")

    # Annotation metadata
    anno = pd.read_csv(ANNOTATION_META_CSV).rename(columns={"REFID": "refid"})
    anno = anno[["refid", "startYear", "endYear"]].dropna()
    anno["startYear"] = anno["startYear"].astype(int)
    anno["endYear"] = anno["endYear"].astype(int)

    # Geojson metadata
    geo_meta = load_geojson_metadata(GEOJSON_PATH)

    # =========================================================================
    # STAGE 1: Split 163 full-window tiles
    # =========================================================================
    print(f"\n{'=' * 70}")
    print(f"STAGE 1: Split {len(full_window_refids)} full-window tiles at 70/15/15")
    print(f"{'=' * 70}")

    fw_df = coarse[coarse["refid"].isin(full_window_refids)].copy()
    fw_train, fw_val, fw_test = stratified_split(fw_df, TRAIN_RATIO, VAL_RATIO, RANDOM_SEED)

    n_fw = len(fw_train) + len(fw_val) + len(fw_test)
    print(f"  Train: {len(fw_train)} ({len(fw_train)/n_fw:.1%})")
    print(f"  Val:   {len(fw_val)} ({len(fw_val)/n_fw:.1%})")
    print(f"  Test:  {len(fw_test)} ({len(fw_test)/n_fw:.1%})")

    # =========================================================================
    # STAGE 2: Split remaining misaligned tiles
    # =========================================================================
    mis_df = coarse[~coarse["refid"].isin(full_window_refids)].copy()

    print(f"\n{'=' * 70}")
    print(f"STAGE 2: Split {len(mis_df)} misaligned tiles at 70/15/15")
    print(f"{'=' * 70}")

    mis_train, mis_val, mis_test = stratified_split(mis_df, TRAIN_RATIO, VAL_RATIO, RANDOM_SEED + 1)

    n_mis = len(mis_train) + len(mis_val) + len(mis_test)
    print(f"  Train: {len(mis_train)} ({len(mis_train)/n_mis:.1%})")
    print(f"  Val:   {len(mis_val)} ({len(mis_val)/n_mis:.1%})")
    print(f"  Test:  {len(mis_test)} ({len(mis_test)/n_mis:.1%})")

    # =========================================================================
    # COMBINE
    # =========================================================================
    for df, label in [(fw_train, "train"), (fw_val, "val"), (fw_test, "test"),
                      (mis_train, "train"), (mis_val, "val"), (mis_test, "test")]:
        df["split"] = label

    all_df = pd.concat([fw_train, fw_val, fw_test, mis_train, mis_val, mis_test], ignore_index=True)

    # Add metadata
    all_df["full_window"] = all_df["refid"].isin(full_window_refids)
    all_df["country"] = all_df["refid"].map(lambda r: geo_meta.get(r, {}).get("country", ""))
    all_df["change_type"] = all_df["refid"].map(lambda r: geo_meta.get(r, {}).get("change_type", ""))
    all_df = all_df.merge(anno, on="refid", how="left")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print(f"\n{'=' * 70}")
    print("COMBINED SPLITS")
    print(f"{'=' * 70}")

    total = len(all_df)
    print(f"\nAll {total} tiles:")
    for split in ["train", "val", "test"]:
        n = (all_df["split"] == split).sum()
        fw = ((all_df["split"] == split) & all_df["full_window"]).sum()
        print(f"  {split}: {n} ({n/total:.1%}) — {fw} full-window, {n-fw} misaligned")

    fw_sub = all_df[all_df["full_window"]]
    n_fw_total = len(fw_sub)
    print(f"\n{n_fw_total} full-window subset:")
    for split in ["train", "val", "test"]:
        n = (fw_sub["split"] == split).sum()
        print(f"  {split}: {n} ({n/n_fw_total:.1%})")

    # Verify no leakage
    for s1, s2 in [("train", "val"), ("train", "test"), ("val", "test")]:
        set1 = set(all_df[all_df["split"] == s1]["refid"])
        set2 = set(all_df[all_df["split"] == s2]["refid"])
        assert len(set1 & set2) == 0, f"{s1}/{s2} overlap!"
    print(f"\nNo leakage detected.")

    # Change level distribution
    print(f"\nChange level distribution:")
    for split in ["train", "val", "test"]:
        sub = all_df[all_df["split"] == split]
        counts = sub["change_level"].value_counts().to_dict()
        print(f"  {split}: {counts}")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["refid", "split", "change_ratio", "change_level", "full_window",
            "startYear", "endYear", "country", "change_type"]
    all_df_out = all_df[[c for c in cols if c in all_df.columns]]

    for split_name in ["train", "val", "test"]:
        split_df = all_df_out[all_df_out["split"] == split_name]
        path = OUTPUT_DIR / f"{split_name}_refids.txt"
        split_df["refid"].to_csv(path, index=False, header=False)
        print(f"  {split_name}: {len(split_df)} tiles → {path.name}")

    out_path = OUTPUT_DIR / "split_info.csv"
    all_df_out.to_csv(out_path, index=False)
    print(f"  Full CSV → {out_path.name}")
    print(f"\nDone! Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
