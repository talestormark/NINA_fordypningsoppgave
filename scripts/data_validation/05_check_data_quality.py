"""
Comprehensive data quality check script
Checks ALL 53 REFIDs across ALL 5 data sources:
- Sentinel-2 (start + end quarters)
- PlanetScope (start + end quarters)
- VHR Google (start + end years)
- AlphaEarth (2018 embeddings)
- Masks (binary labels)
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


def check_sentinel_temporal(filepath):
    """
    Check Sentinel-2 quality for BOTH start (2018 Q2) and end (2024 Q3) quarters

    Args:
        filepath: Path to Sentinel-2 raster (126 bands)

    Returns:
        dict: Quality metrics for both temporal endpoints
    """
    try:
        with rasterio.open(filepath) as src:
            # Read first 9 bands (2018 Q2: bands 1-9)
            start_data = src.read(list(range(1, 10)))
            # Read last 9 bands (2024 Q3: bands 118-126)
            end_data = src.read(list(range(118, 127)))

        results = {}

        # Process start year (2018)
        start_flat = start_data.reshape(9, -1)
        nodata_mask_start = np.isnan(start_flat) | np.isinf(start_flat) | (start_flat < 0)
        nodata_pct_start = (np.sum(nodata_mask_start) / start_flat.size) * 100
        valid_start = start_flat[~nodata_mask_start]

        if len(valid_start) > 0:
            results['s2_start_nodata_pct'] = nodata_pct_start
            results['s2_start_min'] = float(np.min(valid_start))
            results['s2_start_max'] = float(np.max(valid_start))
            results['s2_start_mean'] = float(np.mean(valid_start))
            results['s2_start_std'] = float(np.std(valid_start))
        else:
            results['s2_start_nodata_pct'] = 100.0
            results['s2_start_min'] = None
            results['s2_start_max'] = None
            results['s2_start_mean'] = None
            results['s2_start_std'] = None

        # Process end year (2024)
        end_flat = end_data.reshape(9, -1)
        nodata_mask_end = np.isnan(end_flat) | np.isinf(end_flat) | (end_flat < 0)
        nodata_pct_end = (np.sum(nodata_mask_end) / end_flat.size) * 100
        valid_end = end_flat[~nodata_mask_end]

        if len(valid_end) > 0:
            results['s2_end_nodata_pct'] = nodata_pct_end
            results['s2_end_min'] = float(np.min(valid_end))
            results['s2_end_max'] = float(np.max(valid_end))
            results['s2_end_mean'] = float(np.mean(valid_end))
            results['s2_end_std'] = float(np.std(valid_end))
        else:
            results['s2_end_nodata_pct'] = 100.0
            results['s2_end_min'] = None
            results['s2_end_max'] = None
            results['s2_end_mean'] = None
            results['s2_end_std'] = None

        # Check for issues
        issues = []
        if nodata_pct_start > 5:
            issues.append(f"S2 2018: High NoData {nodata_pct_start:.1f}%")
        if nodata_pct_end > 5:
            issues.append(f"S2 2024: High NoData {nodata_pct_end:.1f}%")
        if results['s2_start_min'] is not None and results['s2_start_min'] < 0:
            issues.append(f"S2 2018: Negative values (min={results['s2_start_min']:.0f})")
        if results['s2_start_max'] is not None and results['s2_start_max'] > 15000:
            issues.append(f"S2 2018: Excessive values (max={results['s2_start_max']:.0f})")
        if results['s2_end_min'] is not None and results['s2_end_min'] < 0:
            issues.append(f"S2 2024: Negative values (min={results['s2_end_min']:.0f})")
        if results['s2_end_max'] is not None and results['s2_end_max'] > 15000:
            issues.append(f"S2 2024: Excessive values (max={results['s2_end_max']:.0f})")

        results['issues'] = issues
        return results

    except Exception as e:
        return {
            's2_start_nodata_pct': None, 's2_start_min': None, 's2_start_max': None,
            's2_start_mean': None, 's2_start_std': None,
            's2_end_nodata_pct': None, 's2_end_min': None, 's2_end_max': None,
            's2_end_mean': None, 's2_end_std': None,
            'issues': [f"S2 Error: {str(e)}"]
        }


def check_planetscope_temporal(filepath):
    """
    Check PlanetScope quality for BOTH start (2018 Q2) and end (2024 Q3) quarters

    Args:
        filepath: Path to PlanetScope raster (42 bands)

    Returns:
        dict: Quality metrics for both temporal endpoints
    """
    try:
        with rasterio.open(filepath) as src:
            # Read first 3 bands (2018 Q2 RGB: bands 1-3)
            start_data = src.read([1, 2, 3])
            # Read last 3 bands (2024 Q3 RGB: bands 40-42)
            end_data = src.read([40, 41, 42])

        results = {}

        # Process start year (2018)
        start_flat = start_data.reshape(3, -1)
        nodata_mask_start = np.isnan(start_flat) | np.isinf(start_flat) | (start_flat < 0)
        nodata_pct_start = (np.sum(nodata_mask_start) / start_flat.size) * 100
        valid_start = start_flat[~nodata_mask_start]

        if len(valid_start) > 0:
            results['ps_start_nodata_pct'] = nodata_pct_start
            results['ps_start_min'] = float(np.min(valid_start))
            results['ps_start_max'] = float(np.max(valid_start))
            results['ps_start_mean'] = float(np.mean(valid_start))
            results['ps_start_std'] = float(np.std(valid_start))
        else:
            results['ps_start_nodata_pct'] = 100.0
            results['ps_start_min'] = None
            results['ps_start_max'] = None
            results['ps_start_mean'] = None
            results['ps_start_std'] = None

        # Process end year (2024)
        end_flat = end_data.reshape(3, -1)
        nodata_mask_end = np.isnan(end_flat) | np.isinf(end_flat) | (end_flat < 0)
        nodata_pct_end = (np.sum(nodata_mask_end) / end_flat.size) * 100
        valid_end = end_flat[~nodata_mask_end]

        if len(valid_end) > 0:
            results['ps_end_nodata_pct'] = nodata_pct_end
            results['ps_end_min'] = float(np.min(valid_end))
            results['ps_end_max'] = float(np.max(valid_end))
            results['ps_end_mean'] = float(np.mean(valid_end))
            results['ps_end_std'] = float(np.std(valid_end))
        else:
            results['ps_end_nodata_pct'] = 100.0
            results['ps_end_min'] = None
            results['ps_end_max'] = None
            results['ps_end_mean'] = None
            results['ps_end_std'] = None

        # Check for issues
        issues = []
        if nodata_pct_start > 5:
            issues.append(f"PS 2018: High NoData {nodata_pct_start:.1f}%")
        if nodata_pct_end > 5:
            issues.append(f"PS 2024: High NoData {nodata_pct_end:.1f}%")
        if results['ps_start_max'] is not None and results['ps_start_max'] > 15000:
            issues.append(f"PS 2018: Excessive values (max={results['ps_start_max']:.0f})")
        if results['ps_end_max'] is not None and results['ps_end_max'] > 15000:
            issues.append(f"PS 2024: Excessive values (max={results['ps_end_max']:.0f})")

        results['issues'] = issues
        return results

    except Exception as e:
        return {
            'ps_start_nodata_pct': None, 'ps_start_min': None, 'ps_start_max': None,
            'ps_start_mean': None, 'ps_start_std': None,
            'ps_end_nodata_pct': None, 'ps_end_min': None, 'ps_end_max': None,
            'ps_end_mean': None, 'ps_end_std': None,
            'issues': [f"PS Error: {str(e)}"]
        }


def check_vhr_temporal(filepath):
    """
    Check VHR Google quality for BOTH start and end years

    Args:
        filepath: Path to VHR raster (6 bands: 3 start + 3 end)

    Returns:
        dict: Quality metrics for both temporal endpoints
    """
    try:
        with rasterio.open(filepath) as src:
            # Read all 6 bands
            data = src.read()

        # Split into start (bands 1-3) and end (bands 4-6)
        start_data = data[0:3]
        end_data = data[3:6]

        results = {}

        # Process start year
        start_flat = start_data.reshape(3, -1)
        # Detect NoData as fully black (0,0,0) or fully white (255,255,255) pixels
        black_start = np.all(start_flat == 0, axis=0).sum()
        white_start = np.all(start_flat == 255, axis=0).sum()
        total_pixels = start_flat.shape[1]
        nodata_pct_start = ((black_start + white_start) / total_pixels) * 100

        results['vhr_start_nodata_pct'] = nodata_pct_start
        results['vhr_start_min'] = float(np.min(start_data))
        results['vhr_start_max'] = float(np.max(start_data))
        results['vhr_start_mean'] = float(np.mean(start_data))
        results['vhr_start_std'] = float(np.std(start_data))

        # Process end year
        end_flat = end_data.reshape(3, -1)
        black_end = np.all(end_flat == 0, axis=0).sum()
        white_end = np.all(end_flat == 255, axis=0).sum()
        nodata_pct_end = ((black_end + white_end) / total_pixels) * 100

        results['vhr_end_nodata_pct'] = nodata_pct_end
        results['vhr_end_min'] = float(np.min(end_data))
        results['vhr_end_max'] = float(np.max(end_data))
        results['vhr_end_mean'] = float(np.mean(end_data))
        results['vhr_end_std'] = float(np.std(end_data))

        # Check for issues
        issues = []
        if nodata_pct_start > 5:
            issues.append(f"VHR start: High NoData {nodata_pct_start:.1f}%")
        if nodata_pct_end > 5:
            issues.append(f"VHR end: High NoData {nodata_pct_end:.1f}%")
        if results['vhr_start_mean'] < 30:
            issues.append(f"VHR start: Very dark (mean={results['vhr_start_mean']:.1f})")
        if results['vhr_end_mean'] < 30:
            issues.append(f"VHR end: Very dark (mean={results['vhr_end_mean']:.1f})")
        if results['vhr_start_mean'] > 225:
            issues.append(f"VHR start: Very bright (mean={results['vhr_start_mean']:.1f})")
        if results['vhr_end_mean'] > 225:
            issues.append(f"VHR end: Very bright (mean={results['vhr_end_mean']:.1f})")

        results['issues'] = issues
        return results

    except Exception as e:
        return {
            'vhr_start_nodata_pct': None, 'vhr_start_min': None, 'vhr_start_max': None,
            'vhr_start_mean': None, 'vhr_start_std': None,
            'vhr_end_nodata_pct': None, 'vhr_end_min': None, 'vhr_end_max': None,
            'vhr_end_mean': None, 'vhr_end_std': None,
            'issues': [f"VHR Error: {str(e)}"]
        }


def check_alphaearth(filepath):
    """
    Check AlphaEarth quality for first 64 bands (2018 embeddings)

    Args:
        filepath: Path to AlphaEarth raster (448 bands)

    Returns:
        dict: Quality metrics for embeddings
    """
    try:
        with rasterio.open(filepath) as src:
            # Read first 64 bands (2018 embeddings)
            data = src.read(list(range(1, 65)))

        data_flat = data.reshape(64, -1)
        nodata_mask = np.isnan(data_flat) | np.isinf(data_flat)
        nodata_pct = (np.sum(nodata_mask) / data_flat.size) * 100
        valid_data = data_flat[~nodata_mask]

        if len(valid_data) > 0:
            results = {
                'ae_nodata_pct': nodata_pct,
                'ae_min': float(np.min(valid_data)),
                'ae_max': float(np.max(valid_data)),
                'ae_mean': float(np.mean(valid_data)),
                'ae_std': float(np.std(valid_data))
            }
        else:
            results = {
                'ae_nodata_pct': 100.0,
                'ae_min': None,
                'ae_max': None,
                'ae_mean': None,
                'ae_std': None
            }

        issues = []
        if nodata_pct > 5:
            issues.append(f"AE: High NoData {nodata_pct:.1f}%")

        results['issues'] = issues
        return results

    except Exception as e:
        return {
            'ae_nodata_pct': None,
            'ae_min': None,
            'ae_max': None,
            'ae_mean': None,
            'ae_std': None,
            'issues': [f"AE Error: {str(e)}"]
        }


def check_mask(filepath):
    """
    Check mask quality (binary labels)

    Args:
        filepath: Path to mask raster (1 band)

    Returns:
        dict: Quality metrics for mask
    """
    try:
        with rasterio.open(filepath) as src:
            mask = src.read(1)

        unique_values = np.unique(mask)
        unique_str = ','.join([str(int(v)) for v in unique_values])

        # Check if only 0 and 1
        valid_values = set([0, 1])
        actual_values = set(unique_values)
        is_valid = actual_values.issubset(valid_values)

        # Compute change percentage
        total_pixels = mask.size
        change_pixels = np.sum(mask == 1)
        change_pct = (change_pixels / total_pixels) * 100

        issues = []
        if not is_valid:
            issues.append(f"Mask: Invalid values {unique_str}")

        return {
            'mask_unique_values': unique_str,
            'mask_change_pct': change_pct,
            'issues': issues
        }

    except Exception as e:
        return {
            'mask_unique_values': None,
            'mask_change_pct': None,
            'issues': [f"Mask Error: {str(e)}"]
        }


def load_refids(refid_file):
    """
    Load REFIDs from enhanced refid_list.txt format (with metadata table)

    Args:
        refid_file: Path to refid_list.txt

    Returns:
        list: REFIDs
    """
    refids = []
    in_data_section = False

    with open(refid_file, 'r') as f:
        for line in f:
            line_stripped = line.strip()

            # Skip empty lines
            if not line_stripped:
                continue

            # Skip header lines
            if line_stripped.startswith('=') or line_stripped.startswith('#') or \
               'Land-Take Detection Dataset' in line_stripped or \
               'European countries' in line_stripped or \
               'Generated:' in line_stripped or \
               'Total REFIDs:' in line_stripped or \
               'These REFIDs have complete data' in line_stripped or \
               'REFID List with Metadata' in line_stripped:
                continue

            # Detect start of data section (header row with "REFID Country Loss Type...")
            if line_stripped.startswith('REFID') and 'Country' in line_stripped:
                in_data_section = True
                continue

            # Skip separator lines
            if line_stripped.startswith('-'):
                continue

            # Parse data lines: REFID is first whitespace-separated token
            if in_data_section:
                # Split on whitespace and take first token
                parts = line_stripped.split()
                if parts:
                    refid = parts[0]
                    # Validate REFID format: starts with 'a', 'R', or 'r' and contains '_'
                    if refid and len(refid) > 10 and refid[0] in 'aRr' and '_' in refid:
                        refids.append(refid)

    return refids


def generate_summary_report(df, output_path):
    """
    Generate a human-readable text summary report from quality check results

    Args:
        df: DataFrame with quality check results
        output_path: Path to save the summary report
    """
    from datetime import datetime

    with open(output_path, 'w') as f:
        # Header
        f.write("=" * 80 + "\n")
        f.write("DATA QUALITY CHECK SUMMARY REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total tiles assessed: {len(df)}\n")
        f.write("=" * 80 + "\n\n")

        # Overall status
        tiles_with_issues = (df['quality_issues'] != 'None').sum()
        tiles_clean = len(df) - tiles_with_issues

        f.write("OVERALL STATUS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Tiles passing all checks: {tiles_clean}/{len(df)}\n")
        f.write(f"  Tiles with quality issues: {tiles_with_issues}/{len(df)}\n")

        if tiles_with_issues == 0:
            f.write("\n  ✓ No quality issues detected across all sources!\n")
        else:
            f.write(f"\n  ⚠ {tiles_with_issues} tile(s) have quality issues\n")

        f.write("\n")

        # Sentinel-2 summary
        s2_valid = df['s2_start_nodata_pct'].notna()
        if s2_valid.any():
            f.write("=" * 80 + "\n")
            f.write("SENTINEL-2 (10m resolution)\n")
            f.write("=" * 80 + "\n")
            f.write(f"Tiles processed: {s2_valid.sum()}/{len(df)}\n\n")

            avg_nodata_start = df.loc[s2_valid, 's2_start_nodata_pct'].mean()
            avg_nodata_end = df.loc[s2_valid, 's2_end_nodata_pct'].mean()
            f.write(f"NoData Analysis:\n")
            f.write(f"  Average NoData 2018 Q2: {avg_nodata_start:.2f}%\n")
            f.write(f"  Average NoData 2024 Q3: {avg_nodata_end:.2f}%\n")

            high_nodata = ((df.loc[s2_valid, 's2_start_nodata_pct'] > 5) |
                          (df.loc[s2_valid, 's2_end_nodata_pct'] > 5)).sum()
            f.write(f"  Tiles with >5% NoData: {high_nodata}/{s2_valid.sum()}\n\n")

            min_start = df.loc[s2_valid, 's2_start_min'].min()
            max_start = df.loc[s2_valid, 's2_start_max'].max()
            min_end = df.loc[s2_valid, 's2_end_min'].min()
            max_end = df.loc[s2_valid, 's2_end_max'].max()
            f.write(f"Value Ranges:\n")
            f.write(f"  2018 Q2: [{min_start:.0f}, {max_start:.0f}]\n")
            f.write(f"  2024 Q3: [{min_end:.0f}, {max_end:.0f}]\n")
            f.write(f"  Expected: [0, 10000] (reflectance × 10k)\n")
            f.write(f"  Flag threshold: >15000\n\n")

            range_issues = ((df.loc[s2_valid, 's2_start_min'] < 0) |
                           (df.loc[s2_valid, 's2_start_max'] > 15000) |
                           (df.loc[s2_valid, 's2_end_min'] < 0) |
                           (df.loc[s2_valid, 's2_end_max'] > 15000)).sum()
            f.write(f"Quality Status:\n")
            f.write(f"  Tiles with value range issues: {range_issues}/{s2_valid.sum()}\n")
            f.write(f"  Status: {'✓ PASS' if range_issues == 0 and high_nodata == 0 else '⚠ ISSUES FOUND'}\n\n")

        # PlanetScope summary
        ps_valid = df['ps_start_nodata_pct'].notna()
        if ps_valid.any():
            f.write("=" * 80 + "\n")
            f.write("PLANETSCOPE (3-5m resolution)\n")
            f.write("=" * 80 + "\n")
            f.write(f"Tiles processed: {ps_valid.sum()}/{len(df)}\n\n")

            avg_nodata_start = df.loc[ps_valid, 'ps_start_nodata_pct'].mean()
            avg_nodata_end = df.loc[ps_valid, 'ps_end_nodata_pct'].mean()
            f.write(f"NoData Analysis:\n")
            f.write(f"  Average NoData 2018 Q2: {avg_nodata_start:.2f}%\n")
            f.write(f"  Average NoData 2024 Q3: {avg_nodata_end:.2f}%\n")

            high_nodata = ((df.loc[ps_valid, 'ps_start_nodata_pct'] > 5) |
                          (df.loc[ps_valid, 'ps_end_nodata_pct'] > 5)).sum()
            f.write(f"  Tiles with >5% NoData: {high_nodata}/{ps_valid.sum()}\n\n")

            min_start = df.loc[ps_valid, 'ps_start_min'].min()
            max_start = df.loc[ps_valid, 'ps_start_max'].max()
            min_end = df.loc[ps_valid, 'ps_end_min'].min()
            max_end = df.loc[ps_valid, 'ps_end_max'].max()
            f.write(f"Value Ranges:\n")
            f.write(f"  2018 Q2 RGB: [{min_start:.0f}, {max_start:.0f}]\n")
            f.write(f"  2024 Q3 RGB: [{min_end:.0f}, {max_end:.0f}]\n\n")

            range_issues = ((df.loc[ps_valid, 'ps_start_max'] > 15000) |
                           (df.loc[ps_valid, 'ps_end_max'] > 15000)).sum()
            f.write(f"Quality Status:\n")
            f.write(f"  Tiles with value range issues: {range_issues}/{ps_valid.sum()}\n")
            f.write(f"  Status: {'✓ PASS' if range_issues == 0 and high_nodata == 0 else '⚠ ISSUES FOUND'}\n\n")

        # VHR summary
        vhr_valid = df['vhr_start_nodata_pct'].notna()
        if vhr_valid.any():
            f.write("=" * 80 + "\n")
            f.write("VHR GOOGLE (1m resolution)\n")
            f.write("=" * 80 + "\n")
            f.write(f"Tiles processed: {vhr_valid.sum()}/{len(df)}\n\n")

            avg_nodata_start = df.loc[vhr_valid, 'vhr_start_nodata_pct'].mean()
            avg_nodata_end = df.loc[vhr_valid, 'vhr_end_nodata_pct'].mean()
            f.write(f"NoData Analysis:\n")
            f.write(f"  Average NoData start year: {avg_nodata_start:.2f}%\n")
            f.write(f"  Average NoData end year: {avg_nodata_end:.2f}%\n")

            high_nodata = ((df.loc[vhr_valid, 'vhr_start_nodata_pct'] > 5) |
                          (df.loc[vhr_valid, 'vhr_end_nodata_pct'] > 5)).sum()
            f.write(f"  Tiles with >5% NoData: {high_nodata}/{vhr_valid.sum()}\n\n")

            avg_brightness_start = df.loc[vhr_valid, 'vhr_start_mean'].mean()
            avg_brightness_end = df.loc[vhr_valid, 'vhr_end_mean'].mean()
            f.write(f"Brightness Analysis:\n")
            f.write(f"  Average brightness start year: {avg_brightness_start:.1f}/255\n")
            f.write(f"  Average brightness end year: {avg_brightness_end:.1f}/255\n\n")

            min_start = df.loc[vhr_valid, 'vhr_start_min'].min()
            max_start = df.loc[vhr_valid, 'vhr_start_max'].max()
            min_end = df.loc[vhr_valid, 'vhr_end_min'].min()
            max_end = df.loc[vhr_valid, 'vhr_end_max'].max()
            f.write(f"Value Ranges:\n")
            f.write(f"  Start year RGB: [{min_start:.0f}, {max_start:.0f}]\n")
            f.write(f"  End year RGB: [{min_end:.0f}, {max_end:.0f}]\n\n")

            f.write(f"Quality Status:\n")
            f.write(f"  Status: {'✓ PASS' if high_nodata == 0 else '⚠ ISSUES FOUND'}\n\n")

        # AlphaEarth summary
        ae_valid = df['ae_nodata_pct'].notna()
        if ae_valid.any():
            f.write("=" * 80 + "\n")
            f.write("ALPHAEARTH (10m resolution, pre-trained embeddings)\n")
            f.write("=" * 80 + "\n")
            f.write(f"Tiles processed: {ae_valid.sum()}/{len(df)}\n\n")

            avg_nodata = df.loc[ae_valid, 'ae_nodata_pct'].mean()
            f.write(f"NoData Analysis:\n")
            f.write(f"  Average NoData: {avg_nodata:.2f}%\n")

            high_nodata = (df.loc[ae_valid, 'ae_nodata_pct'] > 5).sum()
            f.write(f"  Tiles with >5% NoData: {high_nodata}/{ae_valid.sum()}\n\n")

            min_val = df.loc[ae_valid, 'ae_min'].min()
            max_val = df.loc[ae_valid, 'ae_max'].max()
            mean_val = df.loc[ae_valid, 'ae_mean'].mean()
            std_val = df.loc[ae_valid, 'ae_std'].mean()
            f.write(f"Embedding Statistics:\n")
            f.write(f"  Value range: [{min_val:.3f}, {max_val:.3f}]\n")
            f.write(f"  Average value: {mean_val:.3f}\n")
            f.write(f"  Average std: {std_val:.3f}\n")
            f.write(f"  Note: Embeddings can be negative (normalized features)\n\n")

            f.write(f"Quality Status:\n")
            f.write(f"  Status: {'✓ PASS' if high_nodata == 0 else '⚠ ISSUES FOUND'}\n\n")

        # Mask summary
        mask_valid = df['mask_unique_values'].notna()
        if mask_valid.any():
            f.write("=" * 80 + "\n")
            f.write("MASKS (10m resolution, binary labels)\n")
            f.write("=" * 80 + "\n")
            f.write(f"Tiles processed: {mask_valid.sum()}/{len(df)}\n\n")

            invalid_masks = df.loc[mask_valid, 'mask_unique_values'].apply(
                lambda x: not set(str(x).split(',')).issubset({'0', '1'})
            ).sum()

            f.write(f"Binary Validation:\n")
            if invalid_masks == 0:
                f.write(f"  All masks contain only {{0, 1}}: ✓\n")
            else:
                f.write(f"  ⚠ Invalid masks (values other than 0,1): {invalid_masks}/{mask_valid.sum()}\n")

            avg_change = df.loc[mask_valid, 'mask_change_pct'].mean()
            min_change = df.loc[mask_valid, 'mask_change_pct'].min()
            max_change = df.loc[mask_valid, 'mask_change_pct'].max()
            median_change = df.loc[mask_valid, 'mask_change_pct'].median()

            f.write(f"\nChange Statistics:\n")
            f.write(f"  Average change: {avg_change:.2f}%\n")
            f.write(f"  Median change: {median_change:.2f}%\n")
            f.write(f"  Change range: [{min_change:.2f}%, {max_change:.2f}%]\n\n")

            # Categorize tiles by change amount
            zero_change = (df.loc[mask_valid, 'mask_change_pct'] == 0).sum()
            low_change = ((df.loc[mask_valid, 'mask_change_pct'] > 0) &
                         (df.loc[mask_valid, 'mask_change_pct'] < 5)).sum()
            moderate_change = ((df.loc[mask_valid, 'mask_change_pct'] >= 5) &
                              (df.loc[mask_valid, 'mask_change_pct'] < 30)).sum()
            high_change = (df.loc[mask_valid, 'mask_change_pct'] >= 30).sum()

            f.write(f"Change Distribution:\n")
            f.write(f"  Zero change (0%): {zero_change} tiles\n")
            f.write(f"  Low change (0-5%): {low_change} tiles\n")
            f.write(f"  Moderate change (5-30%): {moderate_change} tiles\n")
            f.write(f"  High change (≥30%): {high_change} tiles\n\n")

            f.write(f"Quality Status:\n")
            f.write(f"  Status: {'✓ PASS' if invalid_masks == 0 else '⚠ ISSUES FOUND'}\n\n")

        # Issues section
        if tiles_with_issues > 0:
            f.write("=" * 80 + "\n")
            f.write("TILES WITH QUALITY ISSUES\n")
            f.write("=" * 80 + "\n\n")

            for idx, row in df[df['quality_issues'] != 'None'].iterrows():
                f.write(f"REFID: {row['refid']}\n")
                f.write(f"  Issues: {row['quality_issues']}\n\n")

        # Footer
        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")


if __name__ == "__main__":
    try:
        print("\n" + "=" * 80)
        print("COMPREHENSIVE DATA QUALITY CHECK")
        print("Checking ALL 53 REFIDs across ALL 5 data sources")
        print("=" * 80 + "\n")

        # Load REFID list
        refid_list_file = Path(REPORTS_DIR) / "refid_list.txt"
        refids = load_refids(refid_list_file)

        print(f"✓ Loaded {len(refids)} REFIDs from {refid_list_file}")
        print(f"\nData sources to check:")
        print(f"  1. Sentinel-2 (2018 Q2 + 2024 Q3 bands)")
        print(f"  2. PlanetScope (2018 Q2 + 2024 Q3 RGB)")
        print(f"  3. VHR Google (start + end year RGB)")
        print(f"  4. AlphaEarth (2018 embeddings)")
        print(f"  5. Masks (binary labels)")
        print(f"\nProcessing...\n")

        data_root = Path(DATA_ROOT)
        results = []

        for refid in tqdm(refids, desc="Quality checks"):
            row = {'refid': refid}
            all_issues = []

            # Construct file paths
            sentinel_path = data_root / FOLDERS['sentinel'] / f"{refid}_RGBNIRRSWIRQ_Mosaic.tif"
            planetscope_path = data_root / FOLDERS['planetscope'] / f"{refid}_RGBQ_Mosaic.tif"
            vhr_path = data_root / FOLDERS['vhr'] / f"{refid}_RGBY_Mosaic.tif"
            alphaearth_path = data_root / FOLDERS['alphaearth'] / f"{refid}_VEY_Mosaic.tif"
            mask_path = data_root / FOLDERS['masks'] / f"{refid}_mask.tif"

            # Check Sentinel-2
            if sentinel_path.exists():
                s2_results = check_sentinel_temporal(sentinel_path)
                row.update({k: v for k, v in s2_results.items() if k != 'issues'})
                all_issues.extend(s2_results['issues'])
            else:
                row.update({
                    's2_start_nodata_pct': None, 's2_start_min': None, 's2_start_max': None,
                    's2_start_mean': None, 's2_start_std': None,
                    's2_end_nodata_pct': None, 's2_end_min': None, 's2_end_max': None,
                    's2_end_mean': None, 's2_end_std': None
                })
                all_issues.append("Sentinel-2 file not found")

            # Check PlanetScope
            if planetscope_path.exists():
                ps_results = check_planetscope_temporal(planetscope_path)
                row.update({k: v for k, v in ps_results.items() if k != 'issues'})
                all_issues.extend(ps_results['issues'])
            else:
                row.update({
                    'ps_start_nodata_pct': None, 'ps_start_min': None, 'ps_start_max': None,
                    'ps_start_mean': None, 'ps_start_std': None,
                    'ps_end_nodata_pct': None, 'ps_end_min': None, 'ps_end_max': None,
                    'ps_end_mean': None, 'ps_end_std': None
                })
                all_issues.append("PlanetScope file not found")

            # Check VHR Google
            if vhr_path.exists():
                vhr_results = check_vhr_temporal(vhr_path)
                row.update({k: v for k, v in vhr_results.items() if k != 'issues'})
                all_issues.extend(vhr_results['issues'])
            else:
                row.update({
                    'vhr_start_nodata_pct': None, 'vhr_start_min': None, 'vhr_start_max': None,
                    'vhr_start_mean': None, 'vhr_start_std': None,
                    'vhr_end_nodata_pct': None, 'vhr_end_min': None, 'vhr_end_max': None,
                    'vhr_end_mean': None, 'vhr_end_std': None
                })
                all_issues.append("VHR file not found")

            # Check AlphaEarth
            if alphaearth_path.exists():
                ae_results = check_alphaearth(alphaearth_path)
                row.update({k: v for k, v in ae_results.items() if k != 'issues'})
                all_issues.extend(ae_results['issues'])
            else:
                row.update({
                    'ae_nodata_pct': None, 'ae_min': None, 'ae_max': None,
                    'ae_mean': None, 'ae_std': None
                })
                all_issues.append("AlphaEarth file not found")

            # Check Mask
            if mask_path.exists():
                mask_results = check_mask(mask_path)
                row.update({k: v for k, v in mask_results.items() if k != 'issues'})
                all_issues.extend(mask_results['issues'])
            else:
                row.update({
                    'mask_unique_values': None,
                    'mask_change_pct': None
                })
                all_issues.append("Mask file not found")

            # Combine all issues
            row['quality_issues'] = '; '.join(all_issues) if all_issues else 'None'
            results.append(row)

        # Create DataFrame
        df = pd.DataFrame(results)

        # Save to CSV
        output_file = Path(REPORTS_DIR) / "data_quality.csv"
        df.to_csv(output_file, index=False)

        print(f"\n✓ Data quality report saved to: {output_file}")
        print(f"  Columns: {len(df.columns)}")
        print(f"  Rows: {len(df)}")

        # Print comprehensive summary
        print("\n" + "=" * 80)
        print("📊 DATA QUALITY SUMMARY")
        print("=" * 80 + "\n")

        # Sentinel-2 summary
        s2_valid = df['s2_start_nodata_pct'].notna()
        if s2_valid.any():
            print("SENTINEL-2 (10m resolution):")
            print(f"  Tiles processed: {s2_valid.sum()}/{len(df)}")

            avg_nodata_start = df.loc[s2_valid, 's2_start_nodata_pct'].mean()
            avg_nodata_end = df.loc[s2_valid, 's2_end_nodata_pct'].mean()
            print(f"  Average NoData 2018: {avg_nodata_start:.2f}%")
            print(f"  Average NoData 2024: {avg_nodata_end:.2f}%")

            high_nodata = ((df.loc[s2_valid, 's2_start_nodata_pct'] > 5) |
                          (df.loc[s2_valid, 's2_end_nodata_pct'] > 5)).sum()
            print(f"  Tiles with >5% NoData: {high_nodata}/{s2_valid.sum()}")

            min_start = df.loc[s2_valid, 's2_start_min'].min()
            max_start = df.loc[s2_valid, 's2_start_max'].max()
            min_end = df.loc[s2_valid, 's2_end_min'].min()
            max_end = df.loc[s2_valid, 's2_end_max'].max()
            print(f"  Value range 2018: [{min_start:.0f}, {max_start:.0f}]")
            print(f"  Value range 2024: [{min_end:.0f}, {max_end:.0f}]")
            print(f"  Expected range: [0, 10000] (flag if >15000)")

            range_issues = ((df.loc[s2_valid, 's2_start_min'] < 0) |
                           (df.loc[s2_valid, 's2_start_max'] > 15000) |
                           (df.loc[s2_valid, 's2_end_min'] < 0) |
                           (df.loc[s2_valid, 's2_end_max'] > 15000)).sum()
            print(f"  Tiles with value range issues: {range_issues}/{s2_valid.sum()}")

        # PlanetScope summary
        ps_valid = df['ps_start_nodata_pct'].notna()
        if ps_valid.any():
            print(f"\nPLANETSCOPE (3-5m resolution):")
            print(f"  Tiles processed: {ps_valid.sum()}/{len(df)}")

            avg_nodata_start = df.loc[ps_valid, 'ps_start_nodata_pct'].mean()
            avg_nodata_end = df.loc[ps_valid, 'ps_end_nodata_pct'].mean()
            print(f"  Average NoData 2018: {avg_nodata_start:.2f}%")
            print(f"  Average NoData 2024: {avg_nodata_end:.2f}%")

            high_nodata = ((df.loc[ps_valid, 'ps_start_nodata_pct'] > 5) |
                          (df.loc[ps_valid, 'ps_end_nodata_pct'] > 5)).sum()
            print(f"  Tiles with >5% NoData: {high_nodata}/{ps_valid.sum()}")

            min_start = df.loc[ps_valid, 'ps_start_min'].min()
            max_start = df.loc[ps_valid, 'ps_start_max'].max()
            min_end = df.loc[ps_valid, 'ps_end_min'].min()
            max_end = df.loc[ps_valid, 'ps_end_max'].max()
            print(f"  Value range 2018: [{min_start:.0f}, {max_start:.0f}]")
            print(f"  Value range 2024: [{min_end:.0f}, {max_end:.0f}]")

            range_issues = ((df.loc[ps_valid, 'ps_start_max'] > 15000) |
                           (df.loc[ps_valid, 'ps_end_max'] > 15000)).sum()
            print(f"  Tiles with value range issues: {range_issues}/{ps_valid.sum()}")

        # VHR summary
        vhr_valid = df['vhr_start_nodata_pct'].notna()
        if vhr_valid.any():
            print(f"\nVHR GOOGLE (1m resolution):")
            print(f"  Tiles processed: {vhr_valid.sum()}/{len(df)}")

            avg_nodata_start = df.loc[vhr_valid, 'vhr_start_nodata_pct'].mean()
            avg_nodata_end = df.loc[vhr_valid, 'vhr_end_nodata_pct'].mean()
            print(f"  Average NoData start: {avg_nodata_start:.2f}%")
            print(f"  Average NoData end: {avg_nodata_end:.2f}%")

            high_nodata = ((df.loc[vhr_valid, 'vhr_start_nodata_pct'] > 5) |
                          (df.loc[vhr_valid, 'vhr_end_nodata_pct'] > 5)).sum()
            print(f"  Tiles with >5% NoData: {high_nodata}/{vhr_valid.sum()}")

            avg_brightness_start = df.loc[vhr_valid, 'vhr_start_mean'].mean()
            avg_brightness_end = df.loc[vhr_valid, 'vhr_end_mean'].mean()
            print(f"  Average brightness start: {avg_brightness_start:.1f}/255")
            print(f"  Average brightness end: {avg_brightness_end:.1f}/255")

            min_start = df.loc[vhr_valid, 'vhr_start_min'].min()
            max_start = df.loc[vhr_valid, 'vhr_start_max'].max()
            min_end = df.loc[vhr_valid, 'vhr_end_min'].min()
            max_end = df.loc[vhr_valid, 'vhr_end_max'].max()
            print(f"  Value range start: [{min_start:.0f}, {max_start:.0f}]")
            print(f"  Value range end: [{min_end:.0f}, {max_end:.0f}]")

        # AlphaEarth summary
        ae_valid = df['ae_nodata_pct'].notna()
        if ae_valid.any():
            print(f"\nALPHAEARTH (10m resolution, embeddings):")
            print(f"  Tiles processed: {ae_valid.sum()}/{len(df)}")

            avg_nodata = df.loc[ae_valid, 'ae_nodata_pct'].mean()
            print(f"  Average NoData: {avg_nodata:.2f}%")

            high_nodata = (df.loc[ae_valid, 'ae_nodata_pct'] > 5).sum()
            print(f"  Tiles with >5% NoData: {high_nodata}/{ae_valid.sum()}")

            min_val = df.loc[ae_valid, 'ae_min'].min()
            max_val = df.loc[ae_valid, 'ae_max'].max()
            mean_val = df.loc[ae_valid, 'ae_mean'].mean()
            print(f"  Embedding range: [{min_val:.2f}, {max_val:.2f}]")
            print(f"  Average embedding value: {mean_val:.2f}")
            print(f"  (No expected range - embeddings can be negative)")

        # Mask summary
        mask_valid = df['mask_unique_values'].notna()
        if mask_valid.any():
            print(f"\nMASKS (10m resolution, binary):")
            print(f"  Tiles processed: {mask_valid.sum()}/{len(df)}")

            invalid_masks = df.loc[mask_valid, 'mask_unique_values'].apply(
                lambda x: not set(str(x).split(',')).issubset({'0', '1'})
            ).sum()

            if invalid_masks == 0:
                print(f"  All masks contain only {{0, 1}}: ✓")
            else:
                print(f"  ⚠️  Invalid masks (values other than 0,1): {invalid_masks}/{mask_valid.sum()}")

            avg_change = df.loc[mask_valid, 'mask_change_pct'].mean()
            min_change = df.loc[mask_valid, 'mask_change_pct'].min()
            max_change = df.loc[mask_valid, 'mask_change_pct'].max()
            print(f"  Average change: {avg_change:.2f}%")
            print(f"  Change range: [{min_change:.2f}%, {max_change:.2f}%]")

        # Overall summary
        print(f"\n" + "=" * 80)
        print("OVERALL:")
        tiles_with_issues = (df['quality_issues'] != 'None').sum()
        tiles_clean = len(df) - tiles_with_issues
        print(f"  Total tiles: {len(df)}")
        print(f"  Tiles passing all checks: {tiles_clean}/{len(df)}")
        print(f"  Tiles with quality issues: {tiles_with_issues}/{len(df)}")

        if tiles_with_issues == 0:
            print(f"\n✅ No quality issues detected across all sources!")
        else:
            print(f"\n⚠️  {tiles_with_issues} tile(s) have quality issues:")
            for idx, row in df[df['quality_issues'] != 'None'].iterrows():
                refid_short = row['refid'][:40] + "..." if len(row['refid']) > 40 else row['refid']
                print(f"\n  {refid_short}")
                print(f"    {row['quality_issues']}")

        print("\n" + "=" * 80)
        print("✓ Quality check complete!")
        print("=" * 80 + "\n")

        # Generate text summary report
        print("Generating summary report...")
        summary_file = Path(REPORTS_DIR) / "data_quality_summary.txt"
        generate_summary_report(df, summary_file)
        print(f"✓ Summary report saved to: {summary_file}\n")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
