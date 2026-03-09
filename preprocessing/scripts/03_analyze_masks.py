#!/usr/bin/env python3
"""
Mask analysis for data_v2: compute change ratios for all tiles.
Needed for stratified splitting.

Usage:
    python preprocessing/scripts/03_analyze_masks.py
"""

import sys
import numpy as np
import rasterio
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_V2 = PROJECT_ROOT / "data_v2"
OUTPUT_CSV = PROJECT_ROOT / "preprocessing" / "outputs" / "mask_analysis.csv"

# Change level thresholds
LOW_THRESHOLD = 5.0    # < 5%
HIGH_THRESHOLD = 30.0  # >= 30%

MASK_CONFIGS = [
    ("coarse", "Land_take_masks_coarse", "_mask.tif"),
    ("detailed", "Land_take_masks_detailed", "_mask.tif"),
]


def categorize_change_level(change_ratio: float) -> str:
    if change_ratio < LOW_THRESHOLD:
        return "low"
    elif change_ratio < HIGH_THRESHOLD:
        return "moderate"
    else:
        return "high"


def analyze_mask(path: Path) -> dict:
    """Analyze a single mask file. Returns stats dict."""
    try:
        with rasterio.open(path) as src:
            data = src.read(1)
            total = data.size
            nodata_val = src.nodata

            # Count nodata
            if nodata_val is not None:
                nodata_mask = data == nodata_val
            else:
                nodata_mask = np.zeros_like(data, dtype=bool)

            if np.issubdtype(data.dtype, np.floating):
                nodata_mask |= np.isnan(data)

            nodata_pixels = int(nodata_mask.sum())
            valid_pixels = total - nodata_pixels

            if valid_pixels == 0:
                return {
                    "total_pixels": total,
                    "valid_pixels": 0,
                    "change_pixels": 0,
                    "change_ratio": 0.0,
                    "nodata_pixels": nodata_pixels,
                    "success": False,
                    "error": "No valid pixels",
                }

            # Count change pixels (value == 1)
            valid_data = data[~nodata_mask]
            change_pixels = int((valid_data == 1).sum())
            change_ratio = (change_pixels / valid_pixels) * 100.0

            # Check binary validity
            unique_vals = set(np.unique(valid_data))
            is_binary = unique_vals.issubset({0, 1})

            return {
                "total_pixels": total,
                "valid_pixels": valid_pixels,
                "change_pixels": change_pixels,
                "change_ratio": round(change_ratio, 4),
                "nodata_pixels": nodata_pixels,
                "is_binary": is_binary,
                "height": src.height,
                "width": src.width,
                "success": True,
                "error": "",
            }
    except Exception as e:
        return {
            "total_pixels": 0,
            "valid_pixels": 0,
            "change_pixels": 0,
            "change_ratio": 0.0,
            "nodata_pixels": 0,
            "success": False,
            "error": str(e),
        }


def main():
    print("=" * 70)
    print("MASK ANALYSIS — data_v2")
    print("=" * 70)

    all_rows = []

    for mask_type, folder_name, suffix in MASK_CONFIGS:
        folder = DATA_V2 / folder_name
        if not folder.exists():
            print(f"\nWARNING: {folder} not found, skipping")
            continue

        # Discover masks
        mask_files = sorted(folder.glob("*.tif"))
        refids = []
        for f in mask_files:
            if f.name.endswith(suffix):
                refids.append(f.name[: -len(suffix)])

        print(f"\n{mask_type} masks ({folder_name}): {len(refids)} files")

        for i, refid in enumerate(refids):
            if (i + 1) % 50 == 0:
                print(f"  Progress: {i + 1}/{len(refids)}")

            path = folder / f"{refid}{suffix}"
            stats = analyze_mask(path)
            stats["refid"] = refid
            stats["mask_type"] = mask_type
            stats["change_level"] = categorize_change_level(stats["change_ratio"])
            all_rows.append(stats)

    df = pd.DataFrame(all_rows)

    # Save
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "refid", "mask_type", "total_pixels", "valid_pixels", "change_pixels",
        "change_ratio", "change_level", "nodata_pixels", "height", "width",
        "is_binary", "success", "error",
    ]
    existing_cols = [c for c in cols if c in df.columns]
    df[existing_cols].to_csv(OUTPUT_CSV, index=False)
    print(f"\nMask analysis saved to: {OUTPUT_CSV}")

    # Summary per mask type
    for mask_type in df["mask_type"].unique():
        sub = df[df["mask_type"] == mask_type]
        successful = sub[sub["success"]]
        failed = sub[~sub["success"]]

        print(f"\n{'=' * 70}")
        print(f"{mask_type.upper()} MASKS SUMMARY")
        print(f"{'=' * 70}")
        print(f"  Total: {len(sub)}, Success: {len(successful)}, Failed: {len(failed)}")

        if len(successful) > 0:
            print(f"  Change ratio: mean={successful['change_ratio'].mean():.2f}%, "
                  f"median={successful['change_ratio'].median():.2f}%, "
                  f"max={successful['change_ratio'].max():.2f}%")
            print(f"  Change level distribution:")
            for level in ["low", "moderate", "high"]:
                count = (successful["change_level"] == level).sum()
                pct = count / len(successful) * 100
                print(f"    {level:<10} {count:4d} ({pct:.1f}%)")

            if "is_binary" in successful.columns:
                non_binary = (~successful["is_binary"]).sum()
                if non_binary > 0:
                    print(f"  WARNING: {non_binary} masks are not strictly binary!")

        if len(failed) > 0:
            print(f"  Failed tiles:")
            for _, r in failed.iterrows():
                print(f"    - {r['refid']}: {r['error']}")

    print(f"\n{'=' * 70}")
    print("Done!")


if __name__ == "__main__":
    main()
