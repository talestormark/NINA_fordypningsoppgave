"""
Script to check comprehensive spatial alignment across all data sources

Validates:
1. 10m resolution sources (Sentinel, AlphaEarth, Masks): exact alignment
2. Higher resolution sources (VHR, PlanetScope): geographic bounds coverage
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
    Check comprehensive spatial alignment for a single REFID across all sources

    Args:
        refid: The REFID string
        data_root: Root data directory

    Returns:
        dict: Alignment check results for all source pairs
    """
    data_root = Path(data_root)

    # Construct file paths for all sources
    sentinel_path = data_root / FOLDERS['sentinel'] / f"{refid}_RGBNIRRSWIRQ_Mosaic.tif"
    planetscope_path = data_root / FOLDERS['planetscope'] / f"{refid}_RGBQ_Mosaic.tif"
    vhr_path = data_root / FOLDERS['vhr'] / f"{refid}_RGBY_Mosaic.tif"
    alphaearth_path = data_root / FOLDERS['alphaearth'] / f"{refid}_VEY_Mosaic.tif"
    mask_path = data_root / FOLDERS['masks'] / f"{refid}_mask.tif"

    # Check files exist
    missing_files = []
    if not sentinel_path.exists():
        missing_files.append('Sentinel')
    if not planetscope_path.exists():
        missing_files.append('PlanetScope')
    if not vhr_path.exists():
        missing_files.append('VHR')
    if not alphaearth_path.exists():
        missing_files.append('AlphaEarth')
    if not mask_path.exists():
        missing_files.append('Mask')

    if missing_files:
        return {'error': f"Missing files: {', '.join(missing_files)}"}

    # Get spatial info for all sources
    sentinel_info = get_raster_spatial_info(sentinel_path)
    planetscope_info = get_raster_spatial_info(planetscope_path)
    vhr_info = get_raster_spatial_info(vhr_path)
    alphaearth_info = get_raster_spatial_info(alphaearth_path)
    mask_info = get_raster_spatial_info(mask_path)

    results = {'refid': refid}

    # === CHECK 1: 10m resolution sources (Sentinel, AlphaEarth, Masks) - Must match exactly ===

    # Sentinel ↔ Mask alignment
    sentinel_mask_crs = sentinel_info['crs'] == mask_info['crs']
    sentinel_mask_bounds_aligned, sentinel_mask_bounds_diffs, sentinel_mask_bounds_max = compare_bounds(
        sentinel_info['bounds'], mask_info['bounds'], tolerance=0.0001
    )
    sentinel_mask_transform, sentinel_mask_transform_diffs, sentinel_mask_transform_max = compare_transforms(
        sentinel_info['transform'], mask_info['transform']
    )
    sentinel_mask_dims = (sentinel_info['width'] == mask_info['width'] and
                          sentinel_info['height'] == mask_info['height'])
    sentinel_mask_aligned = (sentinel_mask_crs and sentinel_mask_bounds_aligned and
                            sentinel_mask_transform and sentinel_mask_dims)

    # Sentinel ↔ AlphaEarth alignment
    sentinel_alpha_crs = sentinel_info['crs'] == alphaearth_info['crs']
    sentinel_alpha_bounds_aligned, sentinel_alpha_bounds_diffs, sentinel_alpha_bounds_max = compare_bounds(
        sentinel_info['bounds'], alphaearth_info['bounds'], tolerance=0.0001
    )
    sentinel_alpha_transform, sentinel_alpha_transform_diffs, sentinel_alpha_transform_max = compare_transforms(
        sentinel_info['transform'], alphaearth_info['transform']
    )
    sentinel_alpha_dims = (sentinel_info['width'] == alphaearth_info['width'] and
                          sentinel_info['height'] == alphaearth_info['height'])
    sentinel_alpha_aligned = (sentinel_alpha_crs and sentinel_alpha_bounds_aligned and
                             sentinel_alpha_transform and sentinel_alpha_dims)

    # === CHECK 2: High-resolution sources (VHR, PlanetScope) - Bounds coverage only ===

    # VHR ↔ PlanetScope bounds coverage
    vhr_planet_crs = vhr_info['crs'] == planetscope_info['crs']
    vhr_planet_bounds_aligned, vhr_planet_bounds_diffs, vhr_planet_bounds_max = compare_bounds(
        vhr_info['bounds'], planetscope_info['bounds'], tolerance=0.001  # More lenient for different resolutions
    )

    # === CHECK 3: All sources CRS match ===
    all_crs = [sentinel_info['crs'], planetscope_info['crs'], vhr_info['crs'],
               alphaearth_info['crs'], mask_info['crs']]
    all_crs_match = len(set(str(crs) for crs in all_crs)) == 1

    # === Store results ===
    results.update({
        # 10m resolution group
        'sentinel_mask_aligned': sentinel_mask_aligned,
        'sentinel_mask_crs': sentinel_mask_crs,
        'sentinel_mask_bounds': sentinel_mask_bounds_aligned,
        'sentinel_mask_bounds_max_diff': sentinel_mask_bounds_max,
        'sentinel_mask_transform': sentinel_mask_transform,
        'sentinel_mask_dims': sentinel_mask_dims,

        'sentinel_alpha_aligned': sentinel_alpha_aligned,
        'sentinel_alpha_crs': sentinel_alpha_crs,
        'sentinel_alpha_bounds': sentinel_alpha_bounds_aligned,
        'sentinel_alpha_bounds_max_diff': sentinel_alpha_bounds_max,
        'sentinel_alpha_transform': sentinel_alpha_transform,
        'sentinel_alpha_dims': sentinel_alpha_dims,

        # High-resolution group
        'vhr_planet_bounds_aligned': vhr_planet_bounds_aligned,
        'vhr_planet_crs': vhr_planet_crs,
        'vhr_planet_bounds_max_diff': vhr_planet_bounds_max,

        # Overall
        'all_crs_match': all_crs_match,
        'crs_value': str(sentinel_info['crs']),

        # Dimensions for reference
        'sentinel_dims': f"{sentinel_info['width']}x{sentinel_info['height']}",
        'planetscope_dims': f"{planetscope_info['width']}x{planetscope_info['height']}",
        'vhr_dims': f"{vhr_info['width']}x{vhr_info['height']}",
        'alphaearth_dims': f"{alphaearth_info['width']}x{alphaearth_info['height']}",
        'mask_dims': f"{mask_info['width']}x{mask_info['height']}",

        # Overall pass/fail
        'all_checks_passed': (sentinel_mask_aligned and sentinel_alpha_aligned and
                             vhr_planet_bounds_aligned and all_crs_match)
    })

    return results


if __name__ == "__main__":
    try:
        # Load REFID list
        refid_list_file = Path(REPORTS_DIR) / "refid_list.txt"
        with open(refid_list_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        # Parse enhanced REFID list format
        refids = []
        for line in lines:
            if (line.startswith('a') or line.startswith('-')) and '_' in line and len(line) > 20:
                refid = line.split()[0]
                if refid not in refids:
                    refids.append(refid)

        print(f"\nLoaded {len(refids)} REFIDs from {refid_list_file}")
        print(f"Checking comprehensive spatial alignment for first 5 REFIDs...\n")
        print("=" * 80)

        # Check first 5 REFIDs
        results = []
        refids_to_check = refids[:5]

        for refid in refids_to_check:
            print(f"\nREFID: {refid[:30]}...")
            print("-" * 80)

            result = check_alignment(refid, DATA_ROOT)

            if 'error' in result:
                print(f"  ❌ ERROR: {result['error']}")
                results.append(result)
                continue

            results.append(result)

            # Print results organized by check type
            print(f"\n  📊 10m Resolution Sources (Sentinel, AlphaEarth, Masks):")
            print(f"     Sentinel ↔ Mask:       {'✓ ALIGNED' if result['sentinel_mask_aligned'] else '⚠️ MISALIGNED'}")
            if not result['sentinel_mask_aligned']:
                print(f"       CRS:       {'✓' if result['sentinel_mask_crs'] else '✗'}")
                print(f"       Bounds:    {'✓' if result['sentinel_mask_bounds'] else '✗'} (diff: {result['sentinel_mask_bounds_max_diff']:.6f}°)")
                print(f"       Transform: {'✓' if result['sentinel_mask_transform'] else '✗'}")
                print(f"       Dims:      {'✓' if result['sentinel_mask_dims'] else '✗'}")

            print(f"     Sentinel ↔ AlphaEarth: {'✓ ALIGNED' if result['sentinel_alpha_aligned'] else '⚠️ MISALIGNED'}")
            if not result['sentinel_alpha_aligned']:
                print(f"       CRS:       {'✓' if result['sentinel_alpha_crs'] else '✗'}")
                print(f"       Bounds:    {'✓' if result['sentinel_alpha_bounds'] else '✗'} (diff: {result['sentinel_alpha_bounds_max_diff']:.6f}°)")
                print(f"       Transform: {'✓' if result['sentinel_alpha_transform'] else '✗'}")
                print(f"       Dims:      {'✓' if result['sentinel_alpha_dims'] else '✗'}")

            print(f"\n  📊 High-Resolution Sources (VHR, PlanetScope):")
            print(f"     VHR ↔ PlanetScope:     {'✓ BOUNDS ALIGNED' if result['vhr_planet_bounds_aligned'] else '⚠️ BOUNDS MISALIGNED'}")
            if not result['vhr_planet_bounds_aligned']:
                print(f"       CRS:       {'✓' if result['vhr_planet_crs'] else '✗'}")
                print(f"       Bounds:    ✗ (diff: {result['vhr_planet_bounds_max_diff']:.6f}°)")

            print(f"\n  📊 Overall:")
            print(f"     All CRS Match:         {'✓' if result['all_crs_match'] else '⚠️'} ({result['crs_value']})")
            print(f"     Dimensions:")
            print(f"       Sentinel:    {result['sentinel_dims']}")
            print(f"       PlanetScope: {result['planetscope_dims']}")
            print(f"       VHR:         {result['vhr_dims']}")
            print(f"       AlphaEarth:  {result['alphaearth_dims']}")
            print(f"       Mask:        {result['mask_dims']}")

            if result['all_checks_passed']:
                print(f"\n  ✅ ALL ALIGNMENT CHECKS PASSED")
            else:
                print(f"\n  ⚠️  SOME ALIGNMENT CHECKS FAILED")

        # Summary
        print("\n" + "=" * 80)
        print("\n📊 Summary:")

        successful_checks = [r for r in results if 'error' not in r]
        if successful_checks:
            all_passed = sum(1 for r in successful_checks if r['all_checks_passed'])
            sentinel_mask_ok = sum(1 for r in successful_checks if r['sentinel_mask_aligned'])
            sentinel_alpha_ok = sum(1 for r in successful_checks if r['sentinel_alpha_aligned'])
            vhr_planet_ok = sum(1 for r in successful_checks if r['vhr_planet_bounds_aligned'])
            all_crs_ok = sum(1 for r in successful_checks if r['all_crs_match'])

            print(f"   Tiles checked: {len(successful_checks)}")
            print(f"   All checks passed: {all_passed}/{len(successful_checks)}")
            print(f"\n   Detailed results:")
            print(f"     Sentinel ↔ Mask aligned:       {sentinel_mask_ok}/{len(successful_checks)}")
            print(f"     Sentinel ↔ AlphaEarth aligned: {sentinel_alpha_ok}/{len(successful_checks)}")
            print(f"     VHR ↔ PlanetScope aligned:     {vhr_planet_ok}/{len(successful_checks)}")
            print(f"     All CRS match:                 {all_crs_ok}/{len(successful_checks)}")

            if all_passed == len(successful_checks):
                print(f"\n✅ All checked tiles passed all alignment checks!")
            else:
                print(f"\n⚠️  {len(successful_checks) - all_passed} tile(s) have alignment issues")
        else:
            print("   No successful checks")

        # Save report
        output_file = Path(REPORTS_DIR) / "spatial_alignment.txt"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("COMPREHENSIVE SPATIAL ALIGNMENT REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write("Validates alignment across all data sources:\n")
            f.write("  1. 10m resolution sources (Sentinel, AlphaEarth, Masks): exact alignment\n")
            f.write("  2. High-resolution sources (VHR, PlanetScope): geographic bounds coverage\n\n")
            f.write(f"Checked {len(successful_checks)} tiles\n")
            f.write(f"All checks passed: {all_passed}/{len(successful_checks)}\n\n")

            f.write("Detailed Results:\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Sentinel ↔ Mask aligned:       {sentinel_mask_ok}/{len(successful_checks)}\n")
            f.write(f"  Sentinel ↔ AlphaEarth aligned: {sentinel_alpha_ok}/{len(successful_checks)}\n")
            f.write(f"  VHR ↔ PlanetScope aligned:     {vhr_planet_ok}/{len(successful_checks)}\n")
            f.write(f"  All CRS match:                 {all_crs_ok}/{len(successful_checks)}\n\n")

            f.write("Individual Tile Results:\n")
            f.write("=" * 80 + "\n\n")

            for result in results:
                if 'error' in result:
                    f.write(f"REFID: {result.get('refid', 'unknown')}\n")
                    f.write(f"  ERROR: {result['error']}\n\n")
                    continue

                f.write(f"REFID: {result['refid']}\n")
                f.write(f"  10m Resolution Sources:\n")
                f.write(f"    Sentinel ↔ Mask:       {'ALIGNED' if result['sentinel_mask_aligned'] else 'MISALIGNED'}\n")
                f.write(f"      CRS:       {result['sentinel_mask_crs']}\n")
                f.write(f"      Bounds:    {result['sentinel_mask_bounds']} (max diff: {result['sentinel_mask_bounds_max_diff']:.6f}°)\n")
                f.write(f"      Transform: {result['sentinel_mask_transform']}\n")
                f.write(f"      Dims:      {result['sentinel_mask_dims']}\n")
                f.write(f"    Sentinel ↔ AlphaEarth: {'ALIGNED' if result['sentinel_alpha_aligned'] else 'MISALIGNED'}\n")
                f.write(f"      CRS:       {result['sentinel_alpha_crs']}\n")
                f.write(f"      Bounds:    {result['sentinel_alpha_bounds']} (max diff: {result['sentinel_alpha_bounds_max_diff']:.6f}°)\n")
                f.write(f"      Transform: {result['sentinel_alpha_transform']}\n")
                f.write(f"      Dims:      {result['sentinel_alpha_dims']}\n")
                f.write(f"  High-Resolution Sources:\n")
                f.write(f"    VHR ↔ PlanetScope:     {'ALIGNED' if result['vhr_planet_bounds_aligned'] else 'MISALIGNED'}\n")
                f.write(f"      CRS:       {result['vhr_planet_crs']}\n")
                f.write(f"      Bounds:    {result['vhr_planet_bounds_aligned']} (max diff: {result['vhr_planet_bounds_max_diff']:.6f}°)\n")
                f.write(f"  Overall:\n")
                f.write(f"    All CRS Match: {result['all_crs_match']} ({result['crs_value']})\n")
                f.write(f"    Dimensions:\n")
                f.write(f"      Sentinel:    {result['sentinel_dims']}\n")
                f.write(f"      PlanetScope: {result['planetscope_dims']}\n")
                f.write(f"      VHR:         {result['vhr_dims']}\n")
                f.write(f"      AlphaEarth:  {result['alphaearth_dims']}\n")
                f.write(f"      Mask:        {result['mask_dims']}\n")
                f.write(f"  Status: {'✓ ALL CHECKS PASSED' if result['all_checks_passed'] else '⚠ SOME CHECKS FAILED'}\n")
                f.write("\n")

            f.write("=" * 80 + "\n")
            f.write(f"SUMMARY: {all_passed}/{len(successful_checks)} tiles passed all alignment checks\n")

        print(f"\n✓ Comprehensive alignment report saved to: {output_file}")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
