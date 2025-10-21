"""
Script to check data quality: NoData values, value ranges, and outliers
Includes Sentinel-2, VHR Google, and Mask quality checks
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import rasterio
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import DATA_ROOT, FOLDERS, REPORTS_DIR


def check_sentinel_quality(filepath):
    """
    Check Sentinel-2 data quality for first 9 bands (2018 Q2)

    Args:
        filepath: Path to Sentinel-2 raster

    Returns:
        dict: Quality metrics
    """
    try:
        with rasterio.open(filepath) as src:
            # Read first 9 bands (2018 Q2: bands 1-9)
            data = src.read(list(range(1, 10)))  # Reads bands 1-9

        # Flatten the data
        data_flat = data.reshape(9, -1)  # Shape: (9, total_pixels)

        # Count NaN/NoData
        nodata_mask = np.isnan(data_flat) | np.isinf(data_flat) | (data_flat < 0)
        total_pixels = data_flat.size
        nodata_pixels = np.sum(nodata_mask)
        nodata_pct = (nodata_pixels / total_pixels) * 100

        # Get valid data only
        valid_data = data_flat[~nodata_mask]

        if len(valid_data) > 0:
            min_val = float(np.min(valid_data))
            max_val = float(np.max(valid_data))
            mean_val = float(np.mean(valid_data))
            std_val = float(np.std(valid_data))
        else:
            min_val = max_val = mean_val = std_val = None

        # Check for issues
        issues = []
        if nodata_pct > 5:
            issues.append(f"High NoData: {nodata_pct:.1f}%")
        if min_val is not None and min_val < 0:
            issues.append(f"Negative values: min={min_val}")
        if max_val is not None and max_val > 15000:
            issues.append(f"Excessive values: max={max_val}")

        return {
            'nodata_pct': nodata_pct,
            'min': min_val,
            'max': max_val,
            'mean': mean_val,
            'std': std_val,
            'issues': issues
        }

    except Exception as e:
        return {
            'nodata_pct': None,
            'min': None,
            'max': None,
            'mean': None,
            'std': None,
            'issues': [f"Error reading file: {e}"]
        }


def check_vhr_quality(filepath):
    """
    Check VHR Google data quality for all 6 bands (2018 + 2025 RGB)

    Args:
        filepath: Path to VHR raster

    Returns:
        dict: Quality metrics
    """
    try:
        with rasterio.open(filepath) as src:
            # Read all 6 bands: [2018_R, 2018_G, 2018_B, 2025_R, 2025_G, 2025_B]
            data = src.read()  # Shape: (6, height, width)

        # Separate 2018 and 2025
        rgb_2018 = data[0:3]  # First 3 bands
        rgb_2025 = data[3:6]  # Last 3 bands

        # Flatten for analysis
        rgb_2018_flat = rgb_2018.reshape(3, -1)
        rgb_2025_flat = rgb_2025.reshape(3, -1)

        # Check for NoData (0,0,0 black pixels or 255,255,255 white pixels)
        total_pixels = rgb_2018_flat.shape[1]

        # 2018 checks
        black_2018 = np.all(rgb_2018_flat == 0, axis=0).sum()
        white_2018 = np.all(rgb_2018_flat == 255, axis=0).sum()
        nodata_2018_pct = ((black_2018 + white_2018) / total_pixels) * 100

        # 2025 checks
        black_2025 = np.all(rgb_2025_flat == 0, axis=0).sum()
        white_2025 = np.all(rgb_2025_flat == 255, axis=0).sum()
        nodata_2025_pct = ((black_2025 + white_2025) / total_pixels) * 100

        # Statistics for 2018
        stats_2018 = {
            'min': float(np.min(rgb_2018)),
            'max': float(np.max(rgb_2018)),
            'mean': float(np.mean(rgb_2018)),
            'std': float(np.std(rgb_2018)),
            'mean_r': float(np.mean(rgb_2018_flat[0])),
            'mean_g': float(np.mean(rgb_2018_flat[1])),
            'mean_b': float(np.mean(rgb_2018_flat[2]))
        }

        # Statistics for 2025
        stats_2025 = {
            'min': float(np.min(rgb_2025)),
            'max': float(np.max(rgb_2025)),
            'mean': float(np.mean(rgb_2025)),
            'std': float(np.std(rgb_2025)),
            'mean_r': float(np.mean(rgb_2025_flat[0])),
            'mean_g': float(np.mean(rgb_2025_flat[1])),
            'mean_b': float(np.mean(rgb_2025_flat[2]))
        }

        # Check for issues
        issues = []
        if nodata_2018_pct > 5:
            issues.append(f"2018: High NoData {nodata_2018_pct:.1f}%")
        if nodata_2025_pct > 5:
            issues.append(f"2025: High NoData {nodata_2025_pct:.1f}%")

        # Check if images are too dark or too bright
        if stats_2018['mean'] < 30:
            issues.append(f"2018: Very dark (mean={stats_2018['mean']:.1f})")
        if stats_2025['mean'] < 30:
            issues.append(f"2025: Very dark (mean={stats_2025['mean']:.1f})")
        if stats_2018['mean'] > 225:
            issues.append(f"2018: Very bright (mean={stats_2018['mean']:.1f})")
        if stats_2025['mean'] > 225:
            issues.append(f"2025: Very bright (mean={stats_2025['mean']:.1f})")

        # Check color balance (channels should be reasonably similar)
        rgb_diff_2018 = max(abs(stats_2018['mean_r'] - stats_2018['mean_g']),
                            abs(stats_2018['mean_g'] - stats_2018['mean_b']),
                            abs(stats_2018['mean_r'] - stats_2018['mean_b']))
        rgb_diff_2025 = max(abs(stats_2025['mean_r'] - stats_2025['mean_g']),
                            abs(stats_2025['mean_g'] - stats_2025['mean_b']),
                            abs(stats_2025['mean_r'] - stats_2025['mean_b']))

        if rgb_diff_2018 > 80:
            issues.append(f"2018: Strong color cast (diff={rgb_diff_2018:.1f})")
        if rgb_diff_2025 > 80:
            issues.append(f"2025: Strong color cast (diff={rgb_diff_2025:.1f})")

        return {
            'nodata_2018_pct': nodata_2018_pct,
            'nodata_2025_pct': nodata_2025_pct,
            'min_2018': stats_2018['min'],
            'max_2018': stats_2018['max'],
            'mean_2018': stats_2018['mean'],
            'std_2018': stats_2018['std'],
            'min_2025': stats_2025['min'],
            'max_2025': stats_2025['max'],
            'mean_2025': stats_2025['mean'],
            'std_2025': stats_2025['std'],
            'issues': issues
        }

    except Exception as e:
        return {
            'nodata_2018_pct': None,
            'nodata_2025_pct': None,
            'min_2018': None,
            'max_2018': None,
            'mean_2018': None,
            'std_2018': None,
            'min_2025': None,
            'max_2025': None,
            'mean_2025': None,
            'std_2025': None,
            'issues': [f"Error reading file: {e}"]
        }


def check_mask_quality(filepath):
    """
    Check mask data quality

    Args:
        filepath: Path to mask raster

    Returns:
        dict: Quality metrics
    """
    try:
        with rasterio.open(filepath) as src:
            mask = src.read(1)  # Read single band

        # Get unique values
        unique_values = np.unique(mask)
        unique_values_str = ','.join([str(int(v)) for v in unique_values])

        # Check if only 0 and 1
        valid_values = set([0, 1])
        actual_values = set(unique_values)
        is_valid = actual_values.issubset(valid_values)

        # Count change pixels
        total_pixels = mask.size
        change_pixels = np.sum(mask == 1)
        change_pct = (change_pixels / total_pixels) * 100

        # Check for issues
        issues = []
        if not is_valid:
            issues.append(f"Invalid mask values: {unique_values_str}")

        return {
            'unique_values': unique_values_str,
            'change_pct': change_pct,
            'is_valid': is_valid,
            'issues': issues
        }

    except Exception as e:
        return {
            'unique_values': None,
            'change_pct': None,
            'is_valid': False,
            'issues': [f"Error reading file: {e}"]
        }


if __name__ == "__main__":
    try:
        # Load REFID list
        refid_list_file = Path(REPORTS_DIR) / "refid_list.txt"
        with open(refid_list_file, 'r') as f:
            refids = [line.strip() for line in f if line.strip()]

        print(f"\nLoaded {len(refids)} REFIDs")
        print(f"Checking data quality (Sentinel-2, VHR Google, Masks) for first 10 REFIDs...\n")
        print("=" * 80)

        # Process first 10 REFIDs
        results = []
        refids_to_process = refids[:10]

        data_root = Path(DATA_ROOT)

        for refid in tqdm(refids_to_process, desc="Quality checks"):
            row = {'refid': refid}

            # Construct file paths
            sentinel_path = data_root / FOLDERS['sentinel'] / f"{refid}_RGBNIRRSWIRQ_Mosaic.tif"
            vhr_path = data_root / FOLDERS['vhr'] / f"{refid}_RGBY_Mosaic.tif"
            mask_path = data_root / FOLDERS['masks'] / f"{refid}_mask.tif"

            # Check Sentinel quality
            if sentinel_path.exists():
                s_quality = check_sentinel_quality(sentinel_path)
                row['sentinel_nodata_pct'] = s_quality['nodata_pct']
                row['sentinel_min'] = s_quality['min']
                row['sentinel_max'] = s_quality['max']
                row['sentinel_mean'] = s_quality['mean']
                row['sentinel_std'] = s_quality['std']
                sentinel_issues = s_quality['issues']
            else:
                row['sentinel_nodata_pct'] = None
                row['sentinel_min'] = None
                row['sentinel_max'] = None
                row['sentinel_mean'] = None
                row['sentinel_std'] = None
                sentinel_issues = ["Sentinel file not found"]

            # Check VHR quality
            if vhr_path.exists():
                v_quality = check_vhr_quality(vhr_path)
                row['vhr_nodata_2018_pct'] = v_quality['nodata_2018_pct']
                row['vhr_nodata_2025_pct'] = v_quality['nodata_2025_pct']
                row['vhr_mean_2018'] = v_quality['mean_2018']
                row['vhr_std_2018'] = v_quality['std_2018']
                row['vhr_mean_2025'] = v_quality['mean_2025']
                row['vhr_std_2025'] = v_quality['std_2025']
                vhr_issues = v_quality['issues']
            else:
                row['vhr_nodata_2018_pct'] = None
                row['vhr_nodata_2025_pct'] = None
                row['vhr_mean_2018'] = None
                row['vhr_std_2018'] = None
                row['vhr_mean_2025'] = None
                row['vhr_std_2025'] = None
                vhr_issues = ["VHR file not found"]

            # Check mask quality
            if mask_path.exists():
                m_quality = check_mask_quality(mask_path)
                row['mask_unique_values'] = m_quality['unique_values']
                row['mask_change_pct'] = m_quality['change_pct']
                mask_issues = m_quality['issues']
            else:
                row['mask_unique_values'] = None
                row['mask_change_pct'] = None
                mask_issues = ["Mask file not found"]

            # Combine all issues
            all_issues = sentinel_issues + vhr_issues + mask_issues
            row['quality_issues'] = '; '.join(all_issues) if all_issues else 'None'

            results.append(row)

        # Create DataFrame
        df = pd.DataFrame(results)

        # Save to CSV
        output_file = Path(REPORTS_DIR) / "data_quality.csv"
        df.to_csv(output_file, index=False)
        print(f"\n✓ Data quality report saved to: {output_file}")

        # Print summary
        print("\n" + "=" * 80)
        print("\n📊 Data Quality Summary:\n")

        # Sentinel summary
        sentinel_valid = df['sentinel_nodata_pct'].notna()
        if sentinel_valid.any():
            avg_nodata = df.loc[sentinel_valid, 'sentinel_nodata_pct'].mean()
            high_nodata = (df.loc[sentinel_valid, 'sentinel_nodata_pct'] > 5).sum()
            very_high_nodata = (df.loc[sentinel_valid, 'sentinel_nodata_pct'] > 10).sum()

            print(f"Sentinel-2 Quality:")
            print(f"  Average NoData percentage: {avg_nodata:.2f}%")
            print(f"  Tiles with >5% NoData: {high_nodata}/{sentinel_valid.sum()}")
            print(f"  Tiles with >10% NoData: {very_high_nodata}/{sentinel_valid.sum()}")

            min_overall = df.loc[sentinel_valid, 'sentinel_min'].min()
            max_overall = df.loc[sentinel_valid, 'sentinel_max'].max()
            print(f"  Value range: [{min_overall:.0f}, {max_overall:.0f}] (expected: 0-10,000)")

            out_of_range = ((df.loc[sentinel_valid, 'sentinel_min'] < 0) |
                           (df.loc[sentinel_valid, 'sentinel_max'] > 15000)).sum()
            print(f"  Tiles with out-of-range values: {out_of_range}/{sentinel_valid.sum()}")

        # VHR summary
        vhr_valid = df['vhr_nodata_2018_pct'].notna()
        if vhr_valid.any():
            print(f"\nVHR Google Quality:")
            avg_nodata_2018 = df.loc[vhr_valid, 'vhr_nodata_2018_pct'].mean()
            avg_nodata_2025 = df.loc[vhr_valid, 'vhr_nodata_2025_pct'].mean()
            print(f"  Average NoData 2018: {avg_nodata_2018:.2f}%")
            print(f"  Average NoData 2025: {avg_nodata_2025:.2f}%")

            avg_brightness_2018 = df.loc[vhr_valid, 'vhr_mean_2018'].mean()
            avg_brightness_2025 = df.loc[vhr_valid, 'vhr_mean_2025'].mean()
            print(f"  Average brightness 2018: {avg_brightness_2018:.1f}/255")
            print(f"  Average brightness 2025: {avg_brightness_2025:.1f}/255")

            high_nodata = ((df.loc[vhr_valid, 'vhr_nodata_2018_pct'] > 5) |
                          (df.loc[vhr_valid, 'vhr_nodata_2025_pct'] > 5)).sum()
            print(f"  Tiles with >5% NoData: {high_nodata}/{vhr_valid.sum()}")

        # Mask summary
        mask_valid = df['mask_unique_values'].notna()
        if mask_valid.any():
            print(f"\nMask Quality:")
            invalid_masks = df.loc[mask_valid, 'mask_unique_values'].apply(
                lambda x: not set(str(x).split(',')).issubset({'0', '1'})
            ).sum()

            if invalid_masks == 0:
                print(f"  All masks contain only {{0, 1}}: ✓")
            else:
                print(f"  Invalid masks (values other than 0,1): {invalid_masks}/{mask_valid.sum()}")

            avg_change = df.loc[mask_valid, 'mask_change_pct'].mean()
            print(f"  Average change percentage: {avg_change:.2f}%")

        # Overall issues
        tiles_with_issues = (df['quality_issues'] != 'None').sum()
        print(f"\nOverall:")
        print(f"  Tiles processed: {len(df)}")
        print(f"  Tiles with quality issues: {tiles_with_issues}/{len(df)}")

        if tiles_with_issues == 0:
            print(f"\n✅ No quality issues detected!")
        else:
            print(f"\n⚠️  {tiles_with_issues} tile(s) have quality issues")
            print(f"\nIssues by tile:")
            for idx, row in df[df['quality_issues'] != 'None'].iterrows():
                print(f"  {row['refid'][:30]}...")
                print(f"    {row['quality_issues']}")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
