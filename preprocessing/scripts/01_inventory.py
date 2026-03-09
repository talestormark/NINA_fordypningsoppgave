#!/usr/bin/env python3
"""
Data inventory for data_v2: catalog all files, cross-reference with metadata,
identify gaps, and check old v1 split availability.

Usage:
    python preprocessing/scripts/01_inventory.py
"""

import re
import sys
from pathlib import Path
import pandas as pd

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_V2 = PROJECT_ROOT / "data_v2"
METADATA_CSV = DATA_V2 / "annotations_metadata_final.csv"
V1_SPLITS_DIR = PROJECT_ROOT / "outputs_v1" / "splits"
OUTPUT_CSV = PROJECT_ROOT / "preprocessing" / "outputs" / "data_inventory.csv"

# Source folder config: (folder_name, file_suffix)
SOURCES = {
    "s2":            ("Sentinel",              "_RGBNIRRSWIRQ_Mosaic.tif"),
    "ps":            ("PlanetScope",           "_RGBQ_Mosaic.tif"),
    "vhr":           ("VHR_google",            "_RGBY_Mosaic.tif"),
    "ae":            ("AlphaEarth",            "_VEY_Mosaic.tif"),
    "mask_coarse":   ("Land_take_masks_coarse", "_mask.tif"),
    "mask_detailed": ("Land_take_masks_detailed", "_mask.tif"),
}


def extract_refids_from_folder(folder: Path, suffix: str) -> set:
    """Extract REFIDs from .tif files in a folder by stripping the known suffix."""
    if not folder.exists():
        print(f"  WARNING: folder not found: {folder}")
        return set()
    refids = set()
    for f in folder.glob("*.tif"):
        if f.name.endswith(suffix):
            refid = f.name[: -len(suffix)]
            refids.add(refid)
    return refids


def load_v1_splits() -> dict:
    """Load old v1 split assignments. Returns {refid: split_name}."""
    splits = {}
    for split_name in ["train", "val", "test"]:
        path = V1_SPLITS_DIR / f"{split_name}_refids.txt"
        if path.exists():
            for line in path.read_text().strip().splitlines():
                refid = line.strip()
                if refid:
                    splits[refid] = split_name
    return splits


def main():
    print("=" * 70)
    print("DATA INVENTORY — data_v2")
    print("=" * 70)

    # 1. Load metadata CSV
    meta_df = pd.read_csv(METADATA_CSV)
    meta_refids = set(meta_df["REFID"].values)
    print(f"\nMetadata CSV: {len(meta_refids)} REFIDs")

    # 2. Extract REFIDs per source
    source_refids = {}
    for source_key, (folder_name, suffix) in SOURCES.items():
        folder = DATA_V2 / folder_name
        refids = extract_refids_from_folder(folder, suffix)
        source_refids[source_key] = refids
        print(f"  {source_key:<15} ({folder_name}): {len(refids)} files")

    # 3. Load v1 splits
    v1_splits = load_v1_splits()
    v1_refids = set(v1_splits.keys())
    print(f"\nV1 splits: {len(v1_refids)} REFIDs "
          f"(train={sum(1 for v in v1_splits.values() if v=='train')}, "
          f"val={sum(1 for v in v1_splits.values() if v=='val')}, "
          f"test={sum(1 for v in v1_splits.values() if v=='test')})")

    # 4. Build inventory: union of all REFIDs
    all_refids = set(meta_refids)
    for refids in source_refids.values():
        all_refids.update(refids)
    all_refids.update(v1_refids)

    print(f"\nTotal unique REFIDs across all sources: {len(all_refids)}")

    # 5. Build inventory DataFrame
    rows = []
    for refid in sorted(all_refids):
        meta_row = meta_df[meta_df["REFID"] == refid]
        row = {
            "refid": refid,
            "type": meta_row["type"].values[0] if len(meta_row) > 0 else "",
            "source": meta_row["source"].values[0] if len(meta_row) > 0 else "",
            "startYear": meta_row["startYear"].values[0] if len(meta_row) > 0 else "",
            "endYear": meta_row["endYear"].values[0] if len(meta_row) > 0 else "",
            "has_s2": refid in source_refids["s2"],
            "has_ps": refid in source_refids["ps"],
            "has_vhr": refid in source_refids["vhr"],
            "has_ae": refid in source_refids["ae"],
            "has_mask_coarse": refid in source_refids["mask_coarse"],
            "has_mask_detailed": refid in source_refids["mask_detailed"],
            "in_metadata": refid in meta_refids,
            "in_v1_split": refid in v1_refids,
            "v1_split": v1_splits.get(refid, ""),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # 6. Save
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nInventory saved to: {OUTPUT_CSV}")

    # 7. Summary stats
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    # Tiles with all core sources (S2 + coarse mask = trainable)
    has_core = df["has_s2"] & df["has_mask_coarse"]
    print(f"  Tiles with S2 + coarse mask (trainable): {has_core.sum()}")

    has_all_4 = has_core & df["has_ps"] & df["has_vhr"] & df["has_ae"]
    print(f"  Tiles with all 4 imagery sources + coarse mask: {has_all_4.sum()}")

    # Metadata coverage
    in_meta_but_missing = df[df["in_metadata"] & ~df["has_s2"]]
    print(f"\n  In metadata but missing S2: {len(in_meta_but_missing)}")
    if len(in_meta_but_missing) > 0:
        for _, r in in_meta_but_missing.iterrows():
            print(f"    - {r['refid']}")

    has_file_not_meta = df[~df["in_metadata"] & (df["has_s2"] | df["has_ps"] | df["has_vhr"] | df["has_ae"])]
    print(f"  Has files but not in metadata: {len(has_file_not_meta)}")
    if len(has_file_not_meta) > 0:
        for _, r in has_file_not_meta.iterrows():
            print(f"    - {r['refid']}")

    # V1 availability
    print(f"\n  V1 REFIDs found in v2 S2: "
          f"{sum(1 for r in v1_refids if r in source_refids['s2'])}/{len(v1_refids)}")
    v1_missing_s2 = [r for r in v1_refids if r not in source_refids["s2"]]
    if v1_missing_s2:
        print(f"  V1 REFIDs missing S2 in v2:")
        for r in sorted(v1_missing_s2):
            print(f"    - {r} (was {v1_splits[r]})")

    v1_test_refids = [r for r, s in v1_splits.items() if s == "test"]
    v1_test_in_v2 = [r for r in v1_test_refids if r in source_refids["s2"]]
    print(f"\n  V1 test tiles available in v2: {len(v1_test_in_v2)}/{len(v1_test_refids)}")

    # Year range stats
    years_df = df[df["in_metadata"]].copy()
    years_df["startYear"] = pd.to_numeric(years_df["startYear"], errors="coerce")
    years_df["endYear"] = pd.to_numeric(years_df["endYear"], errors="coerce")
    na_years = years_df[years_df["startYear"].isna() | years_df["endYear"].isna()]
    print(f"\n  Tiles with NA years: {len(na_years)}")
    if len(na_years) > 0:
        for _, r in na_years.iterrows():
            print(f"    - {r['refid']}")
    valid_years = years_df.dropna(subset=["startYear", "endYear"])
    if len(valid_years) > 0:
        print(f"  startYear range: {int(valid_years['startYear'].min())}-{int(valid_years['startYear'].max())}")
        print(f"  endYear range:   {int(valid_years['endYear'].min())}-{int(valid_years['endYear'].max())}")

    # Mask type distribution
    coarse_only = df["has_mask_coarse"] & ~df["has_mask_detailed"]
    detailed = df["has_mask_detailed"]
    neither = ~df["has_mask_coarse"] & ~df["has_mask_detailed"]
    print(f"\n  Mask distribution:")
    print(f"    Coarse only:  {coarse_only.sum()}")
    print(f"    Detailed:     {detailed.sum()} (also have coarse: {(df['has_mask_coarse'] & df['has_mask_detailed']).sum()})")
    print(f"    No mask:      {neither.sum()}")

    print(f"\n{'=' * 70}")
    print("Done!")


if __name__ == "__main__":
    main()
