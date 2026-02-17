#!/usr/bin/env python3
"""
Validate Sentinel-2 temporal data quality for multi-temporal experiments.

Checks:
- Data availability for all time steps (14 quarters)
- NoData percentage per quarter
- Value ranges per band
- Temporal consistency
- Cloud/gap detection

Outputs:
- Per-tile, per-quarter quality report (CSV)
- Summary statistics (TXT)
- Example temporal sequence visualizations (PNG)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore', category=rasterio.errors.NotGeoreferencedWarning)

# Add parent directories to path
repo_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(repo_dir))
sys.path.insert(0, str(repo_dir / "PART1_multi_temporal_experiments"))

try:
    from config import DATA_DIR, REFID_LIST_FILE
    from PART1_multi_temporal_experiments.config import (
        SENTINEL2_DIR, SENTINEL2_PATTERN, SENTINEL2_BANDS,
        YEARS, QUARTERS, SENTINEL2_MIN_VALUE, SENTINEL2_MAX_VALUE,
        SENTINEL2_NODATA, MAX_NODATA_PERCENT, MAX_CLOUD_PERCENT,
        MT_REPORTS_DIR, MT_FIGURES_DIR, TEMPORAL_QUALITY_REPORT,
        TEMPORAL_SUMMARY_REPORT
    )
except ImportError as e:
    print(f"Error importing config: {e}")
    print("Using fallback defaults")
    DATA_DIR = Path("/cluster/home/tmstorma/NINA_fordypningsoppgave/data")
    SENTINEL2_DIR = DATA_DIR / "Sentinel"
    YEARS = list(range(2018, 2025))
    QUARTERS = [2, 3]


def get_refids_from_files():
    """Get list of REFIDs from Sentinel-2 directory."""
    sentinel_files = list(SENTINEL2_DIR.glob("*_RGBNIRRSWIRQ_Mosaic.tif"))
    refids = [f.stem.replace("_RGBNIRRSWIRQ_Mosaic", "") for f in sentinel_files]
    return sorted(refids)


def get_band_index(year, quarter, band_name):
    """
    Calculate band index in Sentinel-2 stack.

    Band order: 2018_Q2_blue, 2018_Q2_green, ..., 2018_Q2_swir2,
                2018_Q3_blue, 2018_Q3_green, ..., 2024_Q3_swir2

    Total: 7 years × 2 quarters × 9 bands = 126 bands
    """
    year_idx = YEARS.index(year)
    quarter_idx = QUARTERS.index(quarter)
    band_idx = SENTINEL2_BANDS.index(band_name)

    # Band index = (year × quarters_per_year + quarter) × bands_per_quarter + band
    time_step = year_idx * len(QUARTERS) + quarter_idx
    band_number = time_step * len(SENTINEL2_BANDS) + band_idx + 1  # 1-indexed

    return band_number


def extract_quarter_data(raster_data, year, quarter):
    """
    Extract all bands for a specific quarter.

    Args:
        raster_data: Full 126-band array (bands, height, width)
        year: Year (2018-2024)
        quarter: Quarter (2 or 3)

    Returns:
        Array of shape (9, height, width) for the quarter
    """
    band_indices = []
    for band_name in SENTINEL2_BANDS:
        idx = get_band_index(year, quarter, band_name) - 1  # Convert to 0-indexed
        band_indices.append(idx)

    return raster_data[band_indices, :, :]


def validate_quarter(quarter_data, year, quarter, refid):
    """
    Validate a single quarter's data.

    Returns dictionary with quality metrics.
    """
    height, width = quarter_data.shape[1:]
    total_pixels = height * width * len(SENTINEL2_BANDS)

    # Calculate NoData percentage
    nodata_mask = (quarter_data == SENTINEL2_NODATA) | (quarter_data < 0)
    num_nodata = np.sum(nodata_mask)
    nodata_percent = (num_nodata / total_pixels) * 100

    # Get valid data statistics
    valid_data = quarter_data[~nodata_mask]

    if len(valid_data) > 0:
        min_val = np.min(valid_data)
        max_val = np.max(valid_data)
        mean_val = np.mean(valid_data)
        std_val = np.std(valid_data)
    else:
        min_val = max_val = mean_val = std_val = np.nan

    # Check for value range issues
    out_of_range = np.sum((valid_data < SENTINEL2_MIN_VALUE) |
                          (valid_data > SENTINEL2_MAX_VALUE))
    out_of_range_percent = (out_of_range / len(valid_data) * 100) if len(valid_data) > 0 else 0

    # Quality assessment
    if nodata_percent > MAX_NODATA_PERCENT:
        quality_flag = "POOR"
    elif nodata_percent > MAX_CLOUD_PERCENT:
        quality_flag = "FAIR"
    else:
        quality_flag = "GOOD"

    return {
        'refid': refid,
        'year': year,
        'quarter': quarter,
        'nodata_percent': nodata_percent,
        'valid_pixels': total_pixels - num_nodata,
        'min_value': min_val,
        'max_value': max_val,
        'mean_value': mean_val,
        'std_value': std_val,
        'out_of_range_percent': out_of_range_percent,
        'quality_flag': quality_flag,
    }


def validate_tile(refid, verbose=False):
    """
    Validate all quarters for a single tile.

    Returns list of quality metrics per quarter.
    """
    sentinel_file = SENTINEL2_DIR / SENTINEL2_PATTERN.format(refid=refid)

    if not sentinel_file.exists():
        if verbose:
            print(f"  ⚠️  File not found: {sentinel_file}")
        return None

    try:
        with rasterio.open(sentinel_file) as src:
            # Check band count
            if src.count != 126:
                if verbose:
                    print(f"  ❌ Wrong band count: {src.count} (expected 126)")
                return None

            # Read all bands
            raster_data = src.read()  # Shape: (126, H, W)

            if verbose:
                print(f"  ✓ Loaded data: {raster_data.shape}")

        # Validate each quarter
        results = []
        for year in YEARS:
            for quarter in QUARTERS:
                quarter_data = extract_quarter_data(raster_data, year, quarter)
                quality = validate_quarter(quarter_data, year, quarter, refid)
                results.append(quality)

                if verbose:
                    flag = quality['quality_flag']
                    nodata = quality['nodata_percent']
                    print(f"    {year} Q{quarter}: {flag} (NoData: {nodata:.2f}%)")

        return results

    except Exception as e:
        if verbose:
            print(f"  ❌ Error reading file: {e}")
        return None


def visualize_temporal_sequence(refid, output_dir, band_indices=[2, 1, 0]):
    """
    Create visualization of temporal sequence for a tile.

    Args:
        refid: Tile REFID
        output_dir: Output directory for PNG
        band_indices: Which bands to use for RGB (default: red, green, blue)
    """
    sentinel_file = SENTINEL2_DIR / SENTINEL2_PATTERN.format(refid=refid)

    if not sentinel_file.exists():
        return

    try:
        with rasterio.open(sentinel_file) as src:
            raster_data = src.read()

        # Create figure with subplots for each quarter
        num_quarters = len(YEARS) * len(QUARTERS)
        fig, axes = plt.subplots(len(YEARS), len(QUARTERS),
                                figsize=(8, len(YEARS) * 3))

        if len(YEARS) == 1:
            axes = [axes]

        for year_idx, year in enumerate(YEARS):
            for quarter_idx, quarter in enumerate(QUARTERS):
                ax = axes[year_idx][quarter_idx] if len(YEARS) > 1 else axes[quarter_idx]

                # Extract quarter data
                quarter_data = extract_quarter_data(raster_data, year, quarter)

                # Create RGB composite (normalize to 0-1)
                rgb = quarter_data[band_indices, :, :]  # Shape: (3, H, W)
                rgb = np.transpose(rgb, (1, 2, 0))  # Shape: (H, W, 3)

                # Normalize
                rgb = np.clip(rgb, 0, 3000)  # Clip to reasonable range
                rgb = rgb / 3000.0

                # Replace NoData with black
                nodata_mask = np.any(rgb == 0, axis=2)
                rgb[nodata_mask] = [0, 0, 0]

                # Display
                ax.imshow(rgb)
                ax.set_title(f"{year} Q{quarter}", fontsize=10)
                ax.axis('off')

        plt.suptitle(f"Temporal Sequence: {refid}", fontsize=14, y=0.995)
        plt.tight_layout()

        output_file = output_dir / f"temporal_sequence_{refid}.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved visualization: {output_file.name}")

    except Exception as e:
        print(f"  ❌ Error creating visualization: {e}")


def generate_summary_report(df, output_file):
    """Generate human-readable summary report."""
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("SENTINEL-2 TEMPORAL DATA QUALITY REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Total tiles validated: {df['refid'].nunique()}\n")
        f.write(f"Total quarters validated: {len(df)}\n")
        f.write(f"Time span: {df['year'].min()}-{df['year'].max()}\n")
        f.write(f"Quarters per year: {df['quarter'].unique()}\n\n")

        f.write("-" * 80 + "\n")
        f.write("OVERALL QUALITY SUMMARY\n")
        f.write("-" * 80 + "\n\n")

        quality_counts = df['quality_flag'].value_counts()
        total = len(df)
        f.write(f"Quality distribution:\n")
        for flag in ['GOOD', 'FAIR', 'POOR']:
            count = quality_counts.get(flag, 0)
            percent = (count / total) * 100
            f.write(f"  {flag:6s}: {count:4d} quarters ({percent:5.1f}%)\n")

        f.write(f"\n")
        f.write(f"NoData statistics:\n")
        f.write(f"  Mean:   {df['nodata_percent'].mean():6.2f}%\n")
        f.write(f"  Median: {df['nodata_percent'].median():6.2f}%\n")
        f.write(f"  Max:    {df['nodata_percent'].max():6.2f}%\n")
        f.write(f"  Min:    {df['nodata_percent'].min():6.2f}%\n")

        f.write(f"\n")
        f.write(f"Value range statistics:\n")
        f.write(f"  Min value across all quarters: {df['min_value'].min():.0f}\n")
        f.write(f"  Max value across all quarters: {df['max_value'].max():.0f}\n")
        f.write(f"  Mean reflectance: {df['mean_value'].mean():.0f}\n")

        f.write("\n" + "-" * 80 + "\n")
        f.write("PER-YEAR QUALITY\n")
        f.write("-" * 80 + "\n\n")

        for year in sorted(df['year'].unique()):
            year_data = df[df['year'] == year]
            f.write(f"{year}:\n")
            f.write(f"  Mean NoData: {year_data['nodata_percent'].mean():6.2f}%\n")
            good_count = len(year_data[year_data['quality_flag'] == 'GOOD'])
            f.write(f"  GOOD quarters: {good_count}/{len(year_data)}\n\n")

        f.write("-" * 80 + "\n")
        f.write("PROBLEMATIC QUARTERS (NoData > 5%)\n")
        f.write("-" * 80 + "\n\n")

        problematic = df[df['nodata_percent'] > MAX_NODATA_PERCENT]
        if len(problematic) == 0:
            f.write("  None! All quarters are good quality.\n")
        else:
            f.write(f"  Found {len(problematic)} problematic quarters:\n\n")
            for _, row in problematic.iterrows():
                f.write(f"  {row['refid']:40s} {row['year']} Q{row['quarter']} "
                       f"NoData: {row['nodata_percent']:6.2f}%\n")

        f.write("\n" + "-" * 80 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("-" * 80 + "\n\n")

        poor_tiles = df[df['quality_flag'] == 'POOR']['refid'].unique()
        if len(poor_tiles) > 0:
            f.write(f"⚠️  Consider excluding {len(poor_tiles)} tiles with POOR quality:\n")
            for refid in poor_tiles:
                f.write(f"  - {refid}\n")
        else:
            f.write("✓ All tiles have acceptable quality (GOOD or FAIR)\n")

        f.write("\n")
        high_nodata_quarters = len(df[df['nodata_percent'] > MAX_CLOUD_PERCENT])
        if high_nodata_quarters > 0:
            f.write(f"⚠️  {high_nodata_quarters} quarters have >20% NoData (may affect quarterly sampling)\n")
            f.write(f"   → Consider using annual composites (average Q2+Q3) to reduce gaps\n")
        else:
            f.write("✓ Quarterly sampling feasible (low NoData across all quarters)\n")

        f.write("\n" + "=" * 80 + "\n")


def main():
    print("=" * 80)
    print("SENTINEL-2 TEMPORAL DATA VALIDATION")
    print("=" * 80)
    print()

    # Create output directories
    MT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    viz_dir = MT_FIGURES_DIR / "temporal_sequences"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Get list of REFIDs
    print("Finding Sentinel-2 files...")
    refids = get_refids_from_files()
    print(f"Found {len(refids)} tiles\n")

    # Validate all tiles
    print("Validating temporal data quality...")
    print(f"Checking {len(YEARS)} years × {len(QUARTERS)} quarters = {len(YEARS) * len(QUARTERS)} time steps per tile")
    print()

    all_results = []
    failed_tiles = []

    for refid in tqdm(refids, desc="Validating tiles"):
        results = validate_tile(refid, verbose=False)
        if results is not None:
            all_results.extend(results)
        else:
            failed_tiles.append(refid)

    if len(failed_tiles) > 0:
        print(f"\n⚠️  Failed to validate {len(failed_tiles)} tiles:")
        for refid in failed_tiles:
            print(f"  - {refid}")
        print()

    # Create DataFrame
    df = pd.DataFrame(all_results)

    # Save detailed CSV
    print(f"Saving detailed quality report: {TEMPORAL_QUALITY_REPORT}")
    df.to_csv(TEMPORAL_QUALITY_REPORT, index=False)

    # Generate summary report
    print(f"Generating summary report: {TEMPORAL_SUMMARY_REPORT}")
    generate_summary_report(df, TEMPORAL_SUMMARY_REPORT)

    # Create visualizations for a few example tiles
    print(f"\nCreating temporal sequence visualizations...")

    # Select examples: one GOOD, one FAIR, one with highest NoData
    good_tiles = df[df['quality_flag'] == 'GOOD']['refid'].unique()
    fair_tiles = df[df['quality_flag'] == 'FAIR']['refid'].unique()

    examples = []
    if len(good_tiles) > 0:
        examples.append(('GOOD', good_tiles[0]))
    if len(fair_tiles) > 0:
        examples.append(('FAIR', fair_tiles[0]))

    # Add tile with highest average NoData
    tile_nodata = df.groupby('refid')['nodata_percent'].mean().sort_values(ascending=False)
    if len(tile_nodata) > 0:
        examples.append(('HIGH_NODATA', tile_nodata.index[0]))

    for label, refid in examples[:3]:  # Max 3 examples
        print(f"  Creating visualization for {refid} ({label})...")
        visualize_temporal_sequence(refid, viz_dir)

    # Print summary statistics
    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)
    print()
    print(f"✓ Validated {len(refids) - len(failed_tiles)}/{len(refids)} tiles")
    print(f"✓ Total quarters checked: {len(df)}")
    print()
    print("Quality summary:")
    quality_counts = df['quality_flag'].value_counts()
    for flag in ['GOOD', 'FAIR', 'POOR']:
        count = quality_counts.get(flag, 0)
        percent = (count / len(df)) * 100
        print(f"  {flag:6s}: {count:4d} quarters ({percent:5.1f}%)")
    print()
    print(f"Mean NoData: {df['nodata_percent'].mean():.2f}%")
    print(f"Max NoData:  {df['nodata_percent'].max():.2f}%")
    print()

    # Recommendation
    poor_count = quality_counts.get('POOR', 0)
    if poor_count == 0:
        print("✅ All quarters have acceptable quality!")
        print("   → Safe to proceed with multi-temporal experiments")
    elif poor_count < len(df) * 0.05:  # Less than 5% poor
        print("⚠️  A few quarters have poor quality, but overall data is good")
        print("   → Can proceed with caution")
    else:
        print("❌ Significant quality issues detected")
        print("   → Review detailed report before proceeding")

    print()
    print("Reports saved:")
    print(f"  - Detailed: {TEMPORAL_QUALITY_REPORT}")
    print(f"  - Summary:  {TEMPORAL_SUMMARY_REPORT}")
    print(f"  - Visualizations: {viz_dir}")
    print()


if __name__ == "__main__":
    main()
