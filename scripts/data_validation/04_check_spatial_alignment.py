"""
Script to check spatial alignment between Sentinel and masks
"""

import sys
from pathlib import Path
import rasterio
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import DATA_ROOT, FOLDERS, REPORTS_DIR


def get_raster_spatial_info(filepath):
    """
    Extract spatial information from raster

    Returns:
        dict with bounds, transform, crs
    """
    with rasterio.open(filepath) as src:
        return {
            'bounds': src.bounds,
            'transform': src.transform,
            'crs': src.crs,
            'width': src.width,
            'height': src.height
        }


def compare_bounds(bounds1, bounds2, tolerance=0.0001):
    """
    Compare two bounding boxes within tolerance

    Args:
        bounds1, bounds2: rasterio.coords.BoundingBox objects
        tolerance: Tolerance in degrees

    Returns:
        tuple: (is_aligned, differences_dict)
    """
    diffs = {
        'left': abs(bounds1.left - bounds2.left),
        'bottom': abs(bounds1.bottom - bounds2.bottom),
        'right': abs(bounds1.right - bounds2.right),
        'top': abs(bounds1.top - bounds2.top)
    }

    max_diff = max(diffs.values())
    is_aligned = max_diff <= tolerance

    return is_aligned, diffs, max_diff


def compare_transforms(transform1, transform2, tolerance=1e-10):
    """
    Compare two affine transformations

    Returns:
        tuple: (is_same, differences)
    """
    # Convert to tuples for comparison
    t1 = transform1.to_gdal()
    t2 = transform2.to_gdal()

    diffs = [abs(a - b) for a, b in zip(t1, t2)]
    max_diff = max(diffs)
    is_same = max_diff <= tolerance

    return is_same, diffs, max_diff


def check_alignment(refid, data_root):
    """
    Check spatial alignment for a single REFID

    Args:
        refid: The REFID string
        data_root: Root data directory

    Returns:
        dict: Alignment check results
    """
    data_root = Path(data_root)

    # Construct file paths
    sentinel_path = data_root / FOLDERS['sentinel'] / f"{refid}_RGBNIRRSWIRQ_Mosaic.tif"
    mask_path = data_root / FOLDERS['masks'] / f"{refid}_mask.tif"

    # Check files exist
    if not sentinel_path.exists():
        return {'error': f"Sentinel file not found: {sentinel_path.name}"}
    if not mask_path.exists():
        return {'error': f"Mask file not found: {mask_path.name}"}

    # Get spatial info
    sentinel_info = get_raster_spatial_info(sentinel_path)
    mask_info = get_raster_spatial_info(mask_path)

    # Compare CRS
    crs_match = sentinel_info['crs'] == mask_info['crs']

    # Compare bounds
    bounds_aligned, bounds_diffs, bounds_max_diff = compare_bounds(
        sentinel_info['bounds'],
        mask_info['bounds'],
        tolerance=0.0001
    )

    # Compare transforms
    transform_match, transform_diffs, transform_max_diff = compare_transforms(
        sentinel_info['transform'],
        mask_info['transform']
    )

    # Compare dimensions
    dims_match = (sentinel_info['width'] == mask_info['width'] and
                  sentinel_info['height'] == mask_info['height'])

    # Overall alignment
    fully_aligned = crs_match and bounds_aligned and transform_match and dims_match

    return {
        'refid': refid,
        'crs_match': crs_match,
        'sentinel_crs': str(sentinel_info['crs']),
        'mask_crs': str(mask_info['crs']),
        'bounds_aligned': bounds_aligned,
        'bounds_max_diff': bounds_max_diff,
        'bounds_diffs': bounds_diffs,
        'transform_match': transform_match,
        'transform_max_diff': transform_max_diff,
        'dims_match': dims_match,
        'sentinel_dims': f"{sentinel_info['width']}x{sentinel_info['height']}",
        'mask_dims': f"{mask_info['width']}x{mask_info['height']}",
        'fully_aligned': fully_aligned
    }


if __name__ == "__main__":
    try:
        # Load REFID list
        refid_list_file = Path(REPORTS_DIR) / "refid_list.txt"
        with open(refid_list_file, 'r') as f:
            refids = [line.strip() for line in f if line.strip()]

        print(f"\nLoaded {len(refids)} REFIDs")
        print(f"Checking spatial alignment for first 5 REFIDs...\n")
        print("=" * 80)

        # Check first 5 REFIDs
        results = []
        refids_to_check = refids[:5]

        for refid in refids_to_check:
            print(f"\nREFID: {refid}")
            print("-" * 80)

            result = check_alignment(refid, DATA_ROOT)

            if 'error' in result:
                print(f"  ❌ ERROR: {result['error']}")
                results.append(result)
                continue

            results.append(result)

            # Print results
            print(f"  CRS Match:       {'✓' if result['crs_match'] else '⚠️'}")
            if not result['crs_match']:
                print(f"    Sentinel: {result['sentinel_crs']}")
                print(f"    Mask:     {result['mask_crs']}")

            print(f"  Bounds Aligned:  {'✓' if result['bounds_aligned'] else '⚠️'} (max diff: {result['bounds_max_diff']:.6f}°)")
            if not result['bounds_aligned']:
                print(f"    Differences: {result['bounds_diffs']}")

            print(f"  Transform Match: {'✓' if result['transform_match'] else '⚠️'} (max diff: {result['transform_max_diff']:.2e})")

            print(f"  Dimensions Match: {'✓' if result['dims_match'] else '⚠️'}")
            if not result['dims_match']:
                print(f"    Sentinel: {result['sentinel_dims']}")
                print(f"    Mask:     {result['mask_dims']}")

            if result['fully_aligned']:
                print(f"\n  ✅ FULLY ALIGNED")
            else:
                print(f"\n  ⚠️  ALIGNMENT ISSUES DETECTED")

        # Summary
        print("\n" + "=" * 80)
        print("\n📊 Summary:")

        successful_checks = [r for r in results if 'error' not in r]
        if successful_checks:
            aligned_count = sum(1 for r in successful_checks if r['fully_aligned'])
            print(f"   Tiles checked: {len(successful_checks)}")
            print(f"   Fully aligned: {aligned_count}/{len(successful_checks)}")

            if aligned_count == len(successful_checks):
                print(f"\n✅ All checked tiles are properly aligned!")
            else:
                print(f"\n⚠️  {len(successful_checks) - aligned_count} tile(s) have alignment issues")
        else:
            print("   No successful checks")

        # Save report
        output_file = Path(REPORTS_DIR) / "spatial_alignment.txt"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            f.write("Spatial Alignment Report\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Checked {len(successful_checks)} tiles\n")
            f.write(f"Fully aligned: {aligned_count}/{len(successful_checks)}\n\n")

            f.write("Individual Results:\n")
            f.write("-" * 80 + "\n\n")

            for result in results:
                if 'error' in result:
                    f.write(f"ERROR: {result['error']}\n\n")
                    continue

                f.write(f"REFID: {result['refid']}\n")
                f.write(f"  CRS Match: {result['crs_match']}\n")
                f.write(f"  Bounds Aligned: {result['bounds_aligned']} (max diff: {result['bounds_max_diff']:.6f}°)\n")
                f.write(f"  Transform Match: {result['transform_match']} (max diff: {result['transform_max_diff']:.2e})\n")
                f.write(f"  Dimensions Match: {result['dims_match']}\n")
                f.write(f"  Fully Aligned: {result['fully_aligned']}\n")
                f.write("\n")

            f.write("=" * 80 + "\n")
            f.write(f"Summary: {aligned_count}/{len(successful_checks)} tiles properly aligned\n")

        print(f"\n✓ Alignment report saved to: {output_file}")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
