#!/usr/bin/env python3
"""
Analyze year ranges across tiles and their implications for temporal alignment.

The annotation masks are derived from VHR imagery at (startYear, endYear), but
S2/PS/AE temporal stacks always cover 2018-2024. This script quantifies the
mismatch and its impact on training.

Usage:
    python preprocessing/scripts/05_analyze_year_ranges.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_V2 = PROJECT_ROOT / "data_v2"
METADATA_CSV = DATA_V2 / "annotations_metadata_final.csv"
INVENTORY_CSV = PROJECT_ROOT / "preprocessing" / "outputs" / "data_inventory.csv"
OUTPUT_CSV = PROJECT_ROOT / "preprocessing" / "outputs" / "year_range_analysis.csv"

# S2 temporal coverage (fixed by GEE export)
S2_START = 2018
S2_END = 2024
S2_YEARS = list(range(S2_START, S2_END + 1))  # [2018, 2019, ..., 2024]


def compute_temporal_alignment(start_year, end_year):
    """
    Compute temporal alignment metrics for a tile.

    Returns dict with:
    - valid_years: S2 years within annotation window
    - n_valid_years: count of valid years
    - extra_before: S2 years before annotation start (pre-change baseline, low risk)
    - extra_after: S2 years after annotation end (unlabeled change, HIGH RISK)
    - missing_before: annotation years before S2 start (lost baseline)
    - missing_after: annotation years after S2 end (invisible change)
    """
    s2_start = max(start_year, S2_START)
    s2_end = min(end_year, S2_END)

    valid_years = [y for y in S2_YEARS if start_year <= y <= end_year]
    extra_before = [y for y in S2_YEARS if y < start_year]
    extra_after = [y for y in S2_YEARS if y > end_year]
    missing_before = list(range(start_year, min(start_year, S2_START)))
    missing_after = list(range(max(end_year, S2_END) + 1, end_year + 1)) if end_year > S2_END else []

    return {
        "valid_years": valid_years,
        "n_valid_years": len(valid_years),
        "n_extra_before": len(extra_before),
        "n_extra_after": len(extra_after),
        "n_missing_before": len(missing_before),
        "n_missing_after": 1 if end_year > S2_END else 0,
        "annotation_span": end_year - start_year,
        "s2_overlap_span": s2_end - s2_start if s2_end >= s2_start else 0,
        "full_window": start_year <= S2_START and end_year >= S2_END,
        "risk_level": classify_risk(start_year, end_year),
    }


def classify_risk(start_year, end_year):
    """Classify temporal alignment risk for training."""
    if start_year <= S2_START and end_year >= S2_END:
        return "none"  # Annotation window covers full S2 range
    elif end_year < S2_END:
        if S2_END - end_year >= 2:
            return "high"  # 2+ years of S2 data beyond annotation
        else:
            return "moderate"  # 1 year beyond
    elif start_year > S2_START:
        return "low"  # Pre-annotation S2 is just baseline, not harmful
    else:
        return "low"  # endYear > S2_END means mask has info S2 can't see


def main():
    print("=" * 70)
    print("YEAR RANGE ANALYSIS — Temporal Alignment")
    print("=" * 70)

    # Load metadata
    meta = pd.read_csv(METADATA_CSV)
    meta["startYear"] = pd.to_numeric(meta["startYear"], errors="coerce")
    meta["endYear"] = pd.to_numeric(meta["endYear"], errors="coerce")

    # Load inventory for S2 availability
    inv = pd.read_csv(INVENTORY_CSV)
    trainable_refids = set(inv[inv["has_s2"] & inv["has_mask_coarse"]]["refid"])

    # Filter to valid tiles
    na_tiles = meta[meta["startYear"].isna() | meta["endYear"].isna()]
    valid = meta.dropna(subset=["startYear", "endYear"]).copy()
    valid["startYear"] = valid["startYear"].astype(int)
    valid["endYear"] = valid["endYear"].astype(int)

    print(f"\nTotal tiles in metadata: {len(meta)}")
    print(f"Tiles with NA years: {len(na_tiles)}")
    if len(na_tiles) > 0:
        for _, r in na_tiles.iterrows():
            print(f"  - {r['REFID']}")
    print(f"Tiles with valid years: {len(valid)}")
    print(f"Trainable tiles (has S2 + coarse mask): {len(trainable_refids)}")

    # Year distributions
    print(f"\n{'=' * 70}")
    print("YEAR DISTRIBUTIONS")
    print(f"{'=' * 70}")

    print("\nstartYear:")
    for year, count in valid["startYear"].value_counts().sort_index().items():
        bar = "#" * count
        print(f"  {year}: {count:4d}  {bar}")

    print("\nendYear:")
    for year, count in valid["endYear"].value_counts().sort_index().items():
        bar = "#" * count
        print(f"  {year}: {count:4d}  {bar}")

    # Compute alignment per tile
    rows = []
    for _, row in valid.iterrows():
        alignment = compute_temporal_alignment(row["startYear"], row["endYear"])
        alignment["refid"] = row["REFID"]
        alignment["startYear"] = row["startYear"]
        alignment["endYear"] = row["endYear"]
        alignment["trainable"] = row["REFID"] in trainable_refids
        rows.append(alignment)

    df = pd.DataFrame(rows)

    # Save
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_cols = [
        "refid", "startYear", "endYear", "annotation_span",
        "n_valid_years", "s2_overlap_span",
        "n_extra_before", "n_extra_after",
        "n_missing_before", "n_missing_after",
        "full_window", "risk_level", "trainable",
    ]
    df[out_cols].to_csv(OUTPUT_CSV, index=False)
    print(f"\nYear range analysis saved to: {OUTPUT_CSV}")

    # Summary
    trainable_df = df[df["trainable"]]

    print(f"\n{'=' * 70}")
    print("TEMPORAL ALIGNMENT SUMMARY (trainable tiles only)")
    print(f"{'=' * 70}")
    print(f"  Total trainable: {len(trainable_df)}")

    # Risk distribution
    print(f"\n  Risk level distribution:")
    for level in ["none", "low", "moderate", "high"]:
        count = (trainable_df["risk_level"] == level).sum()
        pct = count / len(trainable_df) * 100
        print(f"    {level:<10} {count:4d} ({pct:.1f}%)")

    # Full window
    full = trainable_df[trainable_df["full_window"]]
    print(f"\n  Full window (annotation covers all S2 years): {len(full)}")

    # Extra after (dangerous)
    has_extra_after = trainable_df[trainable_df["n_extra_after"] > 0]
    print(f"  Tiles with S2 years AFTER annotation end: {len(has_extra_after)}")
    if len(has_extra_after) > 0:
        print(f"    Extra years breakdown:")
        for n in sorted(has_extra_after["n_extra_after"].unique()):
            c = (has_extra_after["n_extra_after"] == n).sum()
            print(f"      {n} extra year(s): {c} tiles")

    # Valid years stats
    print(f"\n  Valid S2 years per tile:")
    print(f"    min={trainable_df['n_valid_years'].min()}, "
          f"max={trainable_df['n_valid_years'].max()}, "
          f"mean={trainable_df['n_valid_years'].mean():.1f}")

    # Annotation span stats
    print(f"\n  Annotation span (endYear - startYear):")
    print(f"    min={trainable_df['annotation_span'].min()}, "
          f"max={trainable_df['annotation_span'].max()}, "
          f"mean={trainable_df['annotation_span'].mean():.1f}")

    # Year-pair distribution
    print(f"\n{'=' * 70}")
    print("YEAR-PAIR DISTRIBUTION (startYear, endYear)")
    print(f"{'=' * 70}")
    pair_counts = trainable_df.groupby(["startYear", "endYear"]).size().reset_index(name="count")
    pair_counts = pair_counts.sort_values("count", ascending=False)
    for _, r in pair_counts.iterrows():
        risk = classify_risk(r["startYear"], r["endYear"])
        valid_yrs = len([y for y in S2_YEARS if r["startYear"] <= y <= r["endYear"]])
        print(f"  ({int(r['startYear'])}, {int(r['endYear'])}): "
              f"{r['count']:4d} tiles, {valid_yrs} valid S2 years, risk={risk}")

    # Impact on experiment configurations
    print(f"\n{'=' * 70}")
    print("IMPACT ON EXPERIMENTS")
    print(f"{'=' * 70}")

    # T=2 bi-temporal (2018 vs 2024)
    t2_ok = trainable_df[(trainable_df["startYear"] <= 2018) & (trainable_df["endYear"] >= 2024)]
    print(f"\n  T=2 (2018 vs 2024):")
    print(f"    Tiles with both endpoints in annotation window: {len(t2_ok)}/{len(trainable_df)}")

    # T=7 annual
    print(f"\n  T=7 (annual, 2018-2024):")
    print(f"    Tiles with all 7 years in window: {len(full)}/{len(trainable_df)}")
    print(f"    Mean valid years: {trainable_df['n_valid_years'].mean():.1f}/7")

    # A6 temporal diff: mean(2022-2024) - mean(2018-2020)
    a6_late_ok = trainable_df[trainable_df["endYear"] >= 2024]
    a6_early_ok = trainable_df[trainable_df["startYear"] <= 2018]
    a6_both_ok = trainable_df[(trainable_df["startYear"] <= 2018) & (trainable_df["endYear"] >= 2024)]
    print(f"\n  A6 temporal diff (mean(2022-2024) - mean(2018-2020)):")
    print(f"    Late window (2022-2024) in annotation: {len(a6_late_ok)}/{len(trainable_df)}")
    print(f"    Early window (2018-2020) in annotation: {len(a6_early_ok)}/{len(trainable_df)}")
    print(f"    Both windows in annotation: {len(a6_both_ok)}/{len(trainable_df)}")

    # Option analysis
    print(f"\n{'=' * 70}")
    print("OPTION ANALYSIS")
    print(f"{'=' * 70}")

    print(f"\n  Option A (per-tile clipping to valid years):")
    print(f"    All {len(trainable_df)} tiles usable")
    print(f"    Variable T: {trainable_df['n_valid_years'].value_counts().sort_index().to_dict()}")

    print(f"\n  Option B (full-window tiles only):")
    print(f"    {len(full)} tiles usable ({len(full)/len(trainable_df)*100:.0f}%)")

    common_start = int(trainable_df["startYear"].max())
    common_end = int(trainable_df["endYear"].min())
    common_years = [y for y in S2_YEARS if common_start <= y <= common_end]
    print(f"\n  Option C (clip to common overlap {common_start}-{common_end}):")
    print(f"    All {len(trainable_df)} tiles usable")
    print(f"    Uniform T={len(common_years)} ({common_years})")

    print(f"\n  Option D (accept noise, use all 2018-2024):")
    print(f"    All {len(trainable_df)} tiles usable, T=7")
    print(f"    ~{len(has_extra_after)} tiles ({len(has_extra_after)/len(trainable_df)*100:.0f}%) "
          f"have label noise from post-annotation S2 years")

    print(f"\n{'=' * 70}")
    print("Done!")


if __name__ == "__main__":
    main()
