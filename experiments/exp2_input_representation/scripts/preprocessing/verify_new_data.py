#!/usr/bin/env python3
"""
Verification script for new EPSG:3035 data exported from GEE.

Checks:
- CRS is EPSG:3035
- Pixel size is 10m x 10m (square)
- All sources present per tile (S2, PS, AE, mask)
- Band counts (S2: 126, PS: 42, AE: 448, mask: 1)
- Border-fill / nodata percentage
- PS resolution (3m or 10m)
- Grid alignment between sources
- Tile dimensions (>= 64px for crop)

Usage:
    python verify_new_data.py --data-dir data/processed/epsg3035_10m_v2/
    python verify_new_data.py --data-dir data/processed/epsg3035_10m_v2/ --verbose
"""

import argparse
import json
import sys
import numpy as np
import rasterio
from pathlib import Path
from collections import defaultdict


# Expected properties
EXPECTED_CRS = "EPSG:3035"
EXPECTED_PIXEL_SIZE = 10.0  # meters
PIXEL_SIZE_TOL = 0.5  # tolerance in meters

EXPECTED_BANDS = {
    "sentinel": 126,     # 9 bands x 14 timesteps
    "planetscope": 42,   # 3 bands x 14 timesteps
    "alphaearth": 448,   # 64 features x 7 years
    "masks": 1,
}

# Default directory layout (matches dataset.py conventions)
SUBDIR_NAMES = {
    "sentinel": "sentinel",
    "planetscope": "planetscope_10m",
    "alphaearth": "alphaearth",
    "masks": "masks",
}

FILE_PATTERNS = {
    "sentinel": "{refid}_RGBNIRRSWIRQ_Mosaic.tif",
    "planetscope": "{refid}_RGBQ_Mosaic.tif",
    "alphaearth": "{refid}_VEY_Mosaic.tif",
    "masks": "{refid}_mask.tif",
}

MIN_TILE_SIZE = 64  # pixels


def discover_refids(data_dir: Path) -> list:
    """Discover all unique refids from mask files."""
    mask_dir = data_dir / SUBDIR_NAMES["masks"]
    if not mask_dir.exists():
        print(f"ERROR: Mask directory not found: {mask_dir}")
        return []

    refids = []
    for f in sorted(mask_dir.glob("*_mask.tif")):
        refid = f.name.replace("_mask.tif", "")
        refids.append(refid)

    return refids


def check_raster(path: Path, expected_bands: int, source_name: str, verbose: bool = False) -> dict:
    """Check a single raster file and return issues."""
    issues = []
    info = {}

    if not path.exists():
        return {"exists": False, "issues": [f"File not found: {path.name}"]}

    try:
        with rasterio.open(path) as src:
            info["exists"] = True
            info["bands"] = src.count
            info["height"] = src.height
            info["width"] = src.width
            info["crs"] = str(src.crs) if src.crs else "None"
            info["dtype"] = str(src.dtypes[0])

            # CRS check
            if src.crs is None:
                issues.append(f"No CRS defined")
            elif str(src.crs).upper() != EXPECTED_CRS:
                issues.append(f"CRS is {src.crs}, expected {EXPECTED_CRS}")

            # Pixel size check
            res_x = abs(src.transform.a)
            res_y = abs(src.transform.e)
            info["res_x"] = res_x
            info["res_y"] = res_y

            if source_name != "planetscope":
                if abs(res_x - EXPECTED_PIXEL_SIZE) > PIXEL_SIZE_TOL:
                    issues.append(f"X pixel size {res_x:.2f}m, expected {EXPECTED_PIXEL_SIZE}m")
                if abs(res_y - EXPECTED_PIXEL_SIZE) > PIXEL_SIZE_TOL:
                    issues.append(f"Y pixel size {res_y:.2f}m, expected {EXPECTED_PIXEL_SIZE}m")
                if abs(res_x - res_y) > 0.01:
                    issues.append(f"Non-square pixels: {res_x:.2f}m x {res_y:.2f}m")

            # Band count check
            if src.count != expected_bands:
                issues.append(f"Band count {src.count}, expected {expected_bands}")

            # Tile size check
            if src.height < MIN_TILE_SIZE:
                issues.append(f"Height {src.height}px < {MIN_TILE_SIZE}px minimum")
            if src.width < MIN_TILE_SIZE:
                issues.append(f"Width {src.width}px < {MIN_TILE_SIZE}px minimum")

            # Nodata / border-fill check (sample first band)
            data = src.read(1)
            total_pixels = data.size
            nodata_val = src.nodata

            if nodata_val is not None:
                nodata_count = (data == nodata_val).sum()
            else:
                nodata_count = 0

            zero_count = (data == 0).sum()
            nan_count = np.isnan(data).sum() if np.issubdtype(data.dtype, np.floating) else 0

            info["nodata_pct"] = float(nodata_count / total_pixels * 100)
            info["zero_pct"] = float(zero_count / total_pixels * 100)
            info["nan_pct"] = float(nan_count / total_pixels * 100)

            # Flag high nodata
            fill_pct = max(info["nodata_pct"], info["nan_pct"])
            if source_name != "masks" and fill_pct > 5.0:
                issues.append(f"High nodata: {fill_pct:.1f}%")

            # Bounds
            info["bounds"] = list(src.bounds)

    except Exception as e:
        issues.append(f"Error reading file: {e}")
        info["exists"] = True

    info["issues"] = issues
    return info


def check_grid_alignment(tile_infos: dict) -> list:
    """Check that all sources for a tile share the same grid extent."""
    issues = []

    # Compare bounds across sources (excluding planetscope which may differ)
    reference = None
    reference_name = None

    for source_name in ["sentinel", "alphaearth", "masks"]:
        if source_name not in tile_infos:
            continue
        info = tile_infos[source_name]
        if not info.get("exists") or "bounds" not in info:
            continue

        bounds = info["bounds"]
        if reference is None:
            reference = bounds
            reference_name = source_name
        else:
            # Check bounds match within tolerance (1m)
            for i, (a, b) in enumerate(zip(reference, bounds)):
                if abs(a - b) > 1.0:
                    dim = ["left", "bottom", "right", "top"][i]
                    issues.append(
                        f"Grid misalignment: {reference_name} {dim}={a:.1f} vs "
                        f"{source_name} {dim}={b:.1f} (diff={abs(a-b):.1f}m)"
                    )

    # Check that S2/AE/mask have same height/width
    sizes = {}
    for source_name in ["sentinel", "alphaearth", "masks"]:
        if source_name in tile_infos and tile_infos[source_name].get("exists"):
            h = tile_infos[source_name].get("height")
            w = tile_infos[source_name].get("width")
            if h is not None:
                sizes[source_name] = (h, w)

    unique_sizes = set(sizes.values())
    if len(unique_sizes) > 1:
        parts = [f"{name}: {h}x{w}" for name, (h, w) in sizes.items()]
        issues.append(f"Dimension mismatch: {', '.join(parts)}")

    return issues


def main():
    parser = argparse.ArgumentParser(description="Verify new EPSG:3035 data from GEE")
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Root data directory to verify')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed per-tile info')
    parser.add_argument('--output', type=str, default=None,
                        help='Save verification report as JSON')
    parser.add_argument('--ps-expected-res', type=float, default=None,
                        help='Expected PlanetScope resolution (e.g., 3.0 or 10.0)')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    print(f"{'='*70}")
    print(f"Data Verification: {data_dir}")
    print(f"{'='*70}")

    if not data_dir.exists():
        print(f"ERROR: Data directory does not exist: {data_dir}")
        sys.exit(1)

    # Check subdirectories exist
    print("\nChecking directory structure...")
    for source, subdir in SUBDIR_NAMES.items():
        path = data_dir / subdir
        status = "OK" if path.exists() else "MISSING"
        n_files = len(list(path.glob("*.tif"))) if path.exists() else 0
        print(f"  {source:<15} {subdir:<20} {status:<10} ({n_files} files)")

    # Discover refids
    refids = discover_refids(data_dir)
    print(f"\nDiscovered {len(refids)} tiles from mask directory")

    if not refids:
        print("ERROR: No tiles found. Aborting.")
        sys.exit(1)

    # Verify each tile
    all_issues = defaultdict(list)
    tile_reports = {}
    summary = {
        "total_tiles": len(refids),
        "tiles_with_issues": 0,
        "missing_files": 0,
        "crs_issues": 0,
        "pixel_size_issues": 0,
        "band_count_issues": 0,
        "size_issues": 0,
        "nodata_issues": 0,
        "alignment_issues": 0,
    }

    print(f"\nVerifying {len(refids)} tiles...")
    for refid in refids:
        tile_info = {}

        for source_name, expected_bands in EXPECTED_BANDS.items():
            subdir = SUBDIR_NAMES[source_name]
            pattern = FILE_PATTERNS[source_name]
            filepath = data_dir / subdir / pattern.format(refid=refid)

            info = check_raster(filepath, expected_bands, source_name, verbose=args.verbose)
            tile_info[source_name] = info

            if not info.get("exists", True):
                summary["missing_files"] += 1

            for issue in info.get("issues", []):
                if "CRS" in issue:
                    summary["crs_issues"] += 1
                elif "pixel size" in issue or "Non-square" in issue:
                    summary["pixel_size_issues"] += 1
                elif "Band count" in issue:
                    summary["band_count_issues"] += 1
                elif "minimum" in issue:
                    summary["size_issues"] += 1
                elif "nodata" in issue.lower():
                    summary["nodata_issues"] += 1

        # Grid alignment
        alignment_issues = check_grid_alignment(tile_info)
        if alignment_issues:
            summary["alignment_issues"] += len(alignment_issues)
            tile_info["alignment_issues"] = alignment_issues

        # Check PlanetScope resolution if specified
        if args.ps_expected_res and "planetscope" in tile_info:
            ps_info = tile_info["planetscope"]
            if ps_info.get("exists") and "res_x" in ps_info:
                if abs(ps_info["res_x"] - args.ps_expected_res) > PIXEL_SIZE_TOL:
                    alignment_issues.append(
                        f"PS resolution {ps_info['res_x']:.2f}m, expected {args.ps_expected_res}m"
                    )

        # Collect all issues for this tile
        tile_issues = []
        for source_name, info in tile_info.items():
            if isinstance(info, dict) and "issues" in info:
                for issue in info["issues"]:
                    tile_issues.append(f"[{source_name}] {issue}")
        tile_issues.extend([f"[alignment] {i}" for i in alignment_issues])

        if tile_issues:
            all_issues[refid] = tile_issues
            summary["tiles_with_issues"] += 1

        tile_reports[refid] = tile_info

        if args.verbose:
            status = "ISSUES" if tile_issues else "OK"
            print(f"\n  {refid}: {status}")
            for source_name, info in tile_info.items():
                if isinstance(info, dict) and info.get("exists"):
                    print(f"    {source_name}: {info.get('bands', '?')} bands, "
                          f"{info.get('height', '?')}x{info.get('width', '?')}px, "
                          f"nodata={info.get('nodata_pct', 0):.1f}%")
            if tile_issues:
                for issue in tile_issues:
                    print(f"    ! {issue}")

    # Print summary
    print(f"\n{'='*70}")
    print("VERIFICATION SUMMARY")
    print(f"{'='*70}")
    print(f"  Total tiles:          {summary['total_tiles']}")
    print(f"  Tiles with issues:    {summary['tiles_with_issues']}")
    print(f"  Missing files:        {summary['missing_files']}")
    print(f"  CRS issues:           {summary['crs_issues']}")
    print(f"  Pixel size issues:    {summary['pixel_size_issues']}")
    print(f"  Band count issues:    {summary['band_count_issues']}")
    print(f"  Tile size issues:     {summary['size_issues']}")
    print(f"  Nodata issues:        {summary['nodata_issues']}")
    print(f"  Alignment issues:     {summary['alignment_issues']}")

    if all_issues:
        print(f"\n{'='*70}")
        print("TILES WITH ISSUES")
        print(f"{'='*70}")
        for refid in sorted(all_issues.keys()):
            print(f"\n  {refid}:")
            for issue in all_issues[refid]:
                print(f"    - {issue}")
    else:
        print(f"\n  All {summary['total_tiles']} tiles passed verification!")

    # Dimension summary
    print(f"\n{'='*70}")
    print("TILE DIMENSIONS")
    print(f"{'='*70}")
    heights = []
    widths = []
    for refid, info in tile_reports.items():
        if "sentinel" in info and info["sentinel"].get("exists"):
            heights.append(info["sentinel"]["height"])
            widths.append(info["sentinel"]["width"])

    if heights:
        print(f"  Height: min={min(heights)}, max={max(heights)}, "
              f"mean={np.mean(heights):.0f}, median={np.median(heights):.0f}")
        print(f"  Width:  min={min(widths)}, max={max(widths)}, "
              f"mean={np.mean(widths):.0f}, median={np.median(widths):.0f}")
        print(f"  Tiles < {MIN_TILE_SIZE}px in either dim: "
              f"{sum(1 for h, w in zip(heights, widths) if h < MIN_TILE_SIZE or w < MIN_TILE_SIZE)}")

    # Nodata summary
    print(f"\n{'='*70}")
    print("NODATA SUMMARY (Sentinel-2)")
    print(f"{'='*70}")
    nodata_pcts = []
    for refid, info in tile_reports.items():
        if "sentinel" in info and info["sentinel"].get("exists"):
            nodata_pcts.append(info["sentinel"].get("nodata_pct", 0))
    if nodata_pcts:
        print(f"  Mean nodata: {np.mean(nodata_pcts):.2f}%")
        print(f"  Max nodata:  {max(nodata_pcts):.2f}%")
        print(f"  Tiles > 5%:  {sum(1 for p in nodata_pcts if p > 5)}")

    # Save report
    if args.output:
        report = {
            "data_dir": str(data_dir),
            "summary": summary,
            "issues": dict(all_issues),
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: {output_path}")

    # Exit code
    if summary['tiles_with_issues'] > 0:
        print(f"\nWARNING: {summary['tiles_with_issues']} tile(s) have issues!")
        sys.exit(1)
    else:
        print("\nAll checks passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
