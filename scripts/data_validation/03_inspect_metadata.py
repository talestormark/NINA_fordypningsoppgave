"""
Script to inspect raster metadata and validate band counts
"""

import sys
from pathlib import Path
import pandas as pd
import rasterio
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import DATA_ROOT, FOLDERS, FILE_PATTERNS, EXPECTED_BANDS, EXPECTED_CRS, REPORTS_DIR


def inspect_raster_metadata(filepath):
    """
    Open raster with rasterio and extract metadata

    Args:
        filepath: Path to raster file

    Returns:
        dict: Metadata dictionary with count, width, height, crs, bounds, dtype
    """
    try:
        with rasterio.open(filepath) as src:
            metadata = {
                'count': src.count,
                'width': src.width,
                'height': src.height,
                'crs': str(src.crs) if src.crs else None,
                'bounds': src.bounds,
                'dtype': str(src.dtypes[0]) if src.count > 0 else None,
                'transform': src.transform
            }
        return metadata
    except Exception as e:
        print(f"  ⚠️  Error reading {filepath.name}: {e}")
        return None


def construct_filepath(data_root, refid, data_type):
    """
    Construct filepath for a given REFID and data type

    Args:
        data_root: Root data directory
        refid: The REFID string
        data_type: One of 'sentinel', 'planetscope', 'vhr', 'alphaearth', 'mask'

    Returns:
        Path object
    """
    data_root = Path(data_root)
    folder_name = FOLDERS[data_type]

    # Construct filename based on pattern
    if data_type == 'sentinel':
        filename = f"{refid}_RGBNIRRSWIRQ_Mosaic.tif"
    elif data_type == 'planetscope':
        filename = f"{refid}_RGBQ_Mosaic.tif"
    elif data_type == 'vhr':
        filename = f"{refid}_RGBY_Mosaic.tif"
    elif data_type == 'alphaearth':
        filename = f"{refid}_VEY_Mosaic.tif"
    elif data_type == 'masks':
        filename = f"{refid}_mask.tif"
    else:
        raise ValueError(f"Unknown data type: {data_type}")

    return data_root / folder_name / filename


def validate_tile_metadata(data_root, refid):
    """
    Validate metadata for all sources for a given REFID

    Args:
        data_root: Root data directory
        refid: The REFID string

    Returns:
        dict: Metadata per source
    """
    metadata_dict = {}

    # Include all 5 sources: Sentinel, PlanetScope, VHR, AlphaEarth, Masks
    for data_type in ['sentinel', 'planetscope', 'vhr', 'alphaearth', 'masks']:
        filepath = construct_filepath(data_root, refid, data_type)

        if filepath.exists():
            metadata = inspect_raster_metadata(filepath)
            metadata_dict[data_type] = metadata
        else:
            metadata_dict[data_type] = None

    return metadata_dict


def check_band_count(actual, expected, data_type):
    """
    Check if band count matches expected value

    Returns:
        tuple: (is_valid, message)
    """
    if expected is None:
        return True, f"{actual} (not validated)"

    if actual == expected:
        return True, f"✓ {actual}"
    else:
        return False, f"⚠️ {actual} (expected {expected})"


if __name__ == "__main__":
    try:
        # Load REFID list
        refid_list_file = Path(REPORTS_DIR) / "refid_list.txt"
        with open(refid_list_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        # Filter out header lines and extract just the REFIDs
        # REFIDs start with 'a' or '-' and contain underscores
        refids = []
        for line in lines:
            if (line.startswith('a') or line.startswith('-')) and '_' in line and len(line) > 20:
                # Extract first token (REFID) from metadata table
                refid = line.split()[0]
                if refid not in refids:  # Avoid duplicates
                    refids.append(refid)

        print(f"\nLoaded {len(refids)} REFIDs from {refid_list_file}")
        print(f"Processing all {len(refids)} REFIDs...\n")
        print("=" * 80)

        # Process all REFIDs
        results = []
        refids_to_process = refids  # Process all instead of first 10

        for refid in tqdm(refids_to_process, desc="Validating metadata"):
            metadata_dict = validate_tile_metadata(DATA_ROOT, refid)

            # Extract key info
            row = {'refid': refid}

            # Sentinel
            if metadata_dict['sentinel']:
                s_meta = metadata_dict['sentinel']
                row['sentinel_bands'] = s_meta['count']
                row['sentinel_dims'] = f"{s_meta['width']}x{s_meta['height']}"
                row['sentinel_crs'] = s_meta['crs']
            else:
                row['sentinel_bands'] = None
                row['sentinel_dims'] = None
                row['sentinel_crs'] = None

            # PlanetScope
            if metadata_dict['planetscope']:
                p_meta = metadata_dict['planetscope']
                row['planetscope_bands'] = p_meta['count']
                row['planetscope_dims'] = f"{p_meta['width']}x{p_meta['height']}"
                row['planetscope_crs'] = p_meta['crs']
            else:
                row['planetscope_bands'] = None
                row['planetscope_dims'] = None
                row['planetscope_crs'] = None

            # VHR
            if metadata_dict['vhr']:
                v_meta = metadata_dict['vhr']
                row['vhr_bands'] = v_meta['count']
                row['vhr_dims'] = f"{v_meta['width']}x{v_meta['height']}"
                row['vhr_crs'] = v_meta['crs']
            else:
                row['vhr_bands'] = None
                row['vhr_dims'] = None
                row['vhr_crs'] = None

            # AlphaEarth
            if metadata_dict['alphaearth']:
                a_meta = metadata_dict['alphaearth']
                row['alphaearth_bands'] = a_meta['count']
                row['alphaearth_dims'] = f"{a_meta['width']}x{a_meta['height']}"
                row['alphaearth_crs'] = a_meta['crs']
            else:
                row['alphaearth_bands'] = None
                row['alphaearth_dims'] = None
                row['alphaearth_crs'] = None

            # Mask
            if metadata_dict['masks']:
                m_meta = metadata_dict['masks']
                row['mask_bands'] = m_meta['count']
                row['mask_dims'] = f"{m_meta['width']}x{m_meta['height']}"
                row['mask_crs'] = m_meta['crs']
            else:
                row['mask_bands'] = None
                row['mask_dims'] = None
                row['mask_crs'] = None

            # Check if all CRS match
            crs_values = [row['sentinel_crs'], row['planetscope_crs'], row['vhr_crs'],
                         row['alphaearth_crs'], row['mask_crs']]
            crs_values = [c for c in crs_values if c is not None]
            row['all_crs_match'] = len(set(crs_values)) <= 1 if crs_values else False

            # Check band counts
            checks_passed = []
            if row['sentinel_bands'] is not None:
                checks_passed.append(row['sentinel_bands'] == EXPECTED_BANDS['sentinel'])
            if row['planetscope_bands'] is not None:
                checks_passed.append(row['planetscope_bands'] == EXPECTED_BANDS['planetscope'])
            if row['vhr_bands'] is not None:
                checks_passed.append(row['vhr_bands'] == EXPECTED_BANDS['vhr'])
            if row['mask_bands'] is not None:
                checks_passed.append(row['mask_bands'] == EXPECTED_BANDS['mask'])

            row['all_checks_passed'] = all(checks_passed) if checks_passed else False

            results.append(row)

        # Create DataFrame
        df = pd.DataFrame(results)

        # Print summary table
        print("\n" + "=" * 80)
        print("\n📊 Metadata Validation Summary:\n")

        for idx, row in df.iterrows():
            print(f"REFID: {row['refid'][:30]}...")

            # Sentinel
            if row['sentinel_bands'] is not None:
                is_valid, msg = check_band_count(row['sentinel_bands'], EXPECTED_BANDS['sentinel'], 'sentinel')
                print(f"  Sentinel:    {msg:20s} | {row['sentinel_dims']:12s} | {row['sentinel_crs']}")
            else:
                print(f"  Sentinel:    ⚠️  FILE NOT FOUND")

            # PlanetScope
            if row['planetscope_bands'] is not None:
                is_valid, msg = check_band_count(row['planetscope_bands'], EXPECTED_BANDS['planetscope'], 'planetscope')
                print(f"  PlanetScope: {msg:20s} | {row['planetscope_dims']:12s} | {row['planetscope_crs']}")
            else:
                print(f"  PlanetScope: ⚠️  FILE NOT FOUND")

            # VHR
            if row['vhr_bands'] is not None:
                is_valid, msg = check_band_count(row['vhr_bands'], EXPECTED_BANDS['vhr'], 'vhr')
                print(f"  VHR:        {msg:20s} | {row['vhr_dims']:12s} | {row['vhr_crs']}")
            else:
                print(f"  VHR:        ⚠️  FILE NOT FOUND")

            # AlphaEarth
            if row['alphaearth_bands'] is not None:
                is_valid, msg = check_band_count(row['alphaearth_bands'], EXPECTED_BANDS['alphaearth'], 'alphaearth')
                print(f"  AlphaEarth: {msg:20s} | {row['alphaearth_dims']:12s} | {row['alphaearth_crs']}")
            else:
                print(f"  AlphaEarth: ⚠️  FILE NOT FOUND")

            # Mask
            if row['mask_bands'] is not None:
                is_valid, msg = check_band_count(row['mask_bands'], EXPECTED_BANDS['mask'], 'mask')
                print(f"  Mask:       {msg:20s} | {row['mask_dims']:12s} | {row['mask_crs']}")
            else:
                print(f"  Mask:       ⚠️  FILE NOT FOUND")

            # CRS match
            if row['all_crs_match']:
                print(f"  CRS Match:  ✓")
            else:
                print(f"  CRS Match:  ⚠️  CRS values don't match!")

            print()

        # Save to CSV
        output_file = Path(REPORTS_DIR) / "metadata_validation.csv"
        df.to_csv(output_file, index=False)
        print(f"✓ Metadata validation saved to: {output_file}")

        # Overall summary
        print("\n" + "=" * 80)
        print("\n📈 Overall Statistics:")
        print(f"   Tiles processed: {len(df)}")
        print(f"   All checks passed: {df['all_checks_passed'].sum()}/{len(df)}")
        print(f"   All CRS match: {df['all_crs_match'].sum()}/{len(df)}")

        if not df['all_checks_passed'].all():
            print("\n⚠️  Some tiles have band count mismatches!")
        else:
            print("\n✅ All band counts validated successfully!")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
