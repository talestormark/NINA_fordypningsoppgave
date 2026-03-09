#!/usr/bin/env python3
"""
Data verification for data_v2: check CRS, pixel size, band counts, nodata,
tile dimensions, and spatial alignment for all tiles and sources.

Adapted from PART2 verify_new_data.py for the data_v2 folder layout.

Usage:
    python preprocessing/scripts/02_verify_data.py
    python preprocessing/scripts/02_verify_data.py --verbose
"""

import argparse
import json
import sys
import numpy as np
import rasterio
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_V2 = PROJECT_ROOT / "data_v2"
OUTPUT_JSON = PROJECT_ROOT / "preprocessing" / "outputs" / "verification_report.json"

# Expected properties
EXPECTED_CRS = "EPSG:3035"

# Source config: (folder, suffix, expected_bands, expected_res_m, res_tolerance)
SOURCE_CONFIG = {
    "s2":            ("Sentinel",               "_RGBNIRRSWIRQ_Mosaic.tif", 126, 10.0, 0.5),
    "ps":            ("PlanetScope",            "_RGBQ_Mosaic.tif",          42, None, None),  # 3-5m, variable
    "vhr":           ("VHR_google",             "_RGBY_Mosaic.tif",           6,  1.0, 0.5),
    "ae":            ("AlphaEarth",             "_VEY_Mosaic.tif",          448, 10.0, 0.5),
    "mask_coarse":   ("Land_take_masks_coarse", "_mask.tif",                  1, 10.0, 0.5),
    "mask_detailed": ("Land_take_masks_detailed","_mask.tif",                 1,  1.0, 0.5),
}

# 10m sources that should be grid-aligned
ALIGNED_SOURCES = ["s2", "ae", "mask_coarse"]
MIN_TILE_SIZE = 64  # pixels at 10m


def discover_refids(data_dir: Path) -> set:
    """Discover all unique REFIDs across all source folders."""
    all_refids = set()
    for source_key, (folder, suffix, *_) in SOURCE_CONFIG.items():
        folder_path = data_dir / folder
        if not folder_path.exists():
            continue
        for f in folder_path.glob("*.tif"):
            if f.name.endswith(suffix):
                refid = f.name[: -len(suffix)]
                all_refids.add(refid)
    return all_refids


def check_raster(path: Path, expected_bands: int, expected_res: float,
                 res_tol: float, source_key: str) -> dict:
    """Check a single raster file. Returns info dict with issues list."""
    issues = []
    info = {"exists": False, "issues": issues}

    if not path.exists():
        issues.append(f"File not found: {path.name}")
        return info

    try:
        with rasterio.open(path) as src:
            info["exists"] = True
            info["bands"] = src.count
            info["height"] = src.height
            info["width"] = src.width
            info["crs"] = str(src.crs) if src.crs else "None"
            info["dtype"] = str(src.dtypes[0])

            # CRS
            if src.crs is None:
                issues.append("No CRS defined")
            elif str(src.crs).upper() != EXPECTED_CRS:
                issues.append(f"CRS is {src.crs}, expected {EXPECTED_CRS}")

            # Pixel size
            res_x = abs(src.transform.a)
            res_y = abs(src.transform.e)
            info["res_x"] = round(res_x, 4)
            info["res_y"] = round(res_y, 4)

            if expected_res is not None and res_tol is not None:
                if abs(res_x - expected_res) > res_tol:
                    issues.append(f"X pixel size {res_x:.2f}m, expected {expected_res}m")
                if abs(res_y - expected_res) > res_tol:
                    issues.append(f"Y pixel size {res_y:.2f}m, expected {expected_res}m")
                if abs(res_x - res_y) > 0.01:
                    issues.append(f"Non-square pixels: {res_x:.4f}m x {res_y:.4f}m")

            # Band count
            if src.count != expected_bands:
                issues.append(f"Band count {src.count}, expected {expected_bands}")

            # Tile dimensions (only check 10m sources)
            if expected_res is not None and expected_res >= 10.0:
                if src.height < MIN_TILE_SIZE:
                    issues.append(f"Height {src.height}px < {MIN_TILE_SIZE}px minimum")
                if src.width < MIN_TILE_SIZE:
                    issues.append(f"Width {src.width}px < {MIN_TILE_SIZE}px minimum")

            # Nodata check (sample band 1)
            data = src.read(1)
            total_pixels = data.size
            nodata_val = src.nodata

            nodata_count = int((data == nodata_val).sum()) if nodata_val is not None else 0
            zero_count = int((data == 0).sum())
            nan_count = int(np.isnan(data).sum()) if np.issubdtype(data.dtype, np.floating) else 0

            info["nodata_pct"] = round(float(nodata_count / total_pixels * 100), 2)
            info["zero_pct"] = round(float(zero_count / total_pixels * 100), 2)
            info["nan_pct"] = round(float(nan_count / total_pixels * 100), 2)

            fill_pct = max(info["nodata_pct"], info["nan_pct"])
            if source_key not in ("mask_coarse", "mask_detailed") and fill_pct > 5.0:
                issues.append(f"High nodata: {fill_pct:.1f}%")

            info["bounds"] = list(src.bounds)

    except Exception as e:
        info["exists"] = True
        issues.append(f"Error reading file: {e}")

    return info


def check_alignment(tile_infos: dict) -> list:
    """Check grid alignment among 10m sources (S2, AE, mask_coarse)."""
    issues = []
    reference = None
    ref_name = None

    for src_key in ALIGNED_SOURCES:
        if src_key not in tile_infos:
            continue
        info = tile_infos[src_key]
        if not info.get("exists") or "bounds" not in info:
            continue

        bounds = info["bounds"]
        if reference is None:
            reference = bounds
            ref_name = src_key
        else:
            for i, (a, b) in enumerate(zip(reference, bounds)):
                if abs(a - b) > 1.0:
                    dim = ["left", "bottom", "right", "top"][i]
                    issues.append(
                        f"Grid misalignment: {ref_name} {dim}={a:.1f} vs "
                        f"{src_key} {dim}={b:.1f} (diff={abs(a-b):.1f}m)"
                    )

    # Check matching dimensions
    sizes = {}
    for src_key in ALIGNED_SOURCES:
        if src_key in tile_infos and tile_infos[src_key].get("exists"):
            h = tile_infos[src_key].get("height")
            w = tile_infos[src_key].get("width")
            if h is not None:
                sizes[src_key] = (h, w)

    if len(set(sizes.values())) > 1:
        parts = [f"{k}: {h}x{w}" for k, (h, w) in sizes.items()]
        issues.append(f"Dimension mismatch: {', '.join(parts)}")

    return issues


def main():
    parser = argparse.ArgumentParser(description="Verify data_v2 raster properties")
    parser.add_argument("--verbose", action="store_true", help="Print per-tile details")
    parser.add_argument("--data-dir", type=str, default=str(DATA_V2),
                        help="Data directory (default: data_v2/)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    print(f"{'=' * 70}")
    print(f"DATA VERIFICATION — {data_dir}")
    print(f"{'=' * 70}")

    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        sys.exit(1)

    # Check directory structure
    print("\nDirectory structure:")
    for src_key, (folder, suffix, exp_bands, exp_res, _) in SOURCE_CONFIG.items():
        path = data_dir / folder
        exists = path.exists()
        n_files = len(list(path.glob("*.tif"))) if exists else 0
        status = "OK" if exists else "MISSING"
        print(f"  {src_key:<15} {folder:<25} {status:<8} ({n_files} files)")

    # Discover all REFIDs
    all_refids = sorted(discover_refids(data_dir))
    print(f"\nDiscovered {len(all_refids)} unique REFIDs across all sources")

    # Verify each tile
    summary = {
        "total_tiles": len(all_refids),
        "tiles_with_issues": 0,
        "missing_files": 0,
        "crs_issues": 0,
        "pixel_size_issues": 0,
        "band_count_issues": 0,
        "size_issues": 0,
        "nodata_issues": 0,
        "alignment_issues": 0,
    }
    all_issues = {}
    tile_reports = {}

    print(f"\nVerifying {len(all_refids)} tiles...")
    for idx, refid in enumerate(all_refids):
        if (idx + 1) % 50 == 0:
            print(f"  Progress: {idx + 1}/{len(all_refids)}")

        tile_info = {}
        for src_key, (folder, suffix, exp_bands, exp_res, res_tol) in SOURCE_CONFIG.items():
            filepath = data_dir / folder / f"{refid}{suffix}"
            info = check_raster(filepath, exp_bands, exp_res, res_tol, src_key)
            tile_info[src_key] = info

            if not info.get("exists"):
                # Only count as missing if the file SHOULD exist (folder exists)
                if (data_dir / folder).exists():
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

        # Alignment
        alignment_issues = check_alignment(tile_info)
        if alignment_issues:
            summary["alignment_issues"] += len(alignment_issues)
            tile_info["alignment_issues"] = alignment_issues

        # Collect tile issues
        tile_issues = []
        for src_key, info in tile_info.items():
            if isinstance(info, dict) and "issues" in info:
                for issue in info["issues"]:
                    tile_issues.append(f"[{src_key}] {issue}")
        if isinstance(tile_info.get("alignment_issues"), list):
            tile_issues.extend([f"[alignment] {i}" for i in tile_info["alignment_issues"]])

        if tile_issues:
            all_issues[refid] = tile_issues
            summary["tiles_with_issues"] += 1

        tile_reports[refid] = tile_info

        if args.verbose and tile_issues:
            print(f"\n  {refid}: ISSUES")
            for issue in tile_issues:
                print(f"    - {issue}")

    # Print summary
    print(f"\n{'=' * 70}")
    print("VERIFICATION SUMMARY")
    print(f"{'=' * 70}")
    for k, v in summary.items():
        print(f"  {k:<25} {v}")

    # Tiles with issues (abbreviated)
    if all_issues:
        # Only show tiles with issues OTHER than "File not found"
        real_issues = {
            refid: [i for i in issues if "File not found" not in i]
            for refid, issues in all_issues.items()
        }
        real_issues = {k: v for k, v in real_issues.items() if v}

        if real_issues:
            print(f"\n{'=' * 70}")
            print(f"TILES WITH NON-MISSING-FILE ISSUES ({len(real_issues)})")
            print(f"{'=' * 70}")
            for refid in sorted(real_issues.keys())[:20]:
                print(f"\n  {refid}:")
                for issue in real_issues[refid]:
                    print(f"    - {issue}")
            if len(real_issues) > 20:
                print(f"\n  ... and {len(real_issues) - 20} more tiles with issues")

    # Dimension stats from S2
    print(f"\n{'=' * 70}")
    print("S2 TILE DIMENSIONS")
    print(f"{'=' * 70}")
    heights, widths = [], []
    for info in tile_reports.values():
        if "s2" in info and info["s2"].get("exists"):
            heights.append(info["s2"]["height"])
            widths.append(info["s2"]["width"])
    if heights:
        print(f"  Height: min={min(heights)}, max={max(heights)}, "
              f"mean={np.mean(heights):.0f}, unique={len(set(heights))}")
        print(f"  Width:  min={min(widths)}, max={max(widths)}, "
              f"mean={np.mean(widths):.0f}, unique={len(set(widths))}")

    # Save JSON report
    report = {
        "data_dir": str(data_dir),
        "summary": summary,
        "issues": dict(all_issues),
        "tile_count_per_source": {
            src_key: sum(
                1 for r in tile_reports.values()
                if src_key in r and r[src_key].get("exists", False)
            )
            for src_key in SOURCE_CONFIG
        },
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to: {OUTPUT_JSON}")

    print("\nDone!")
    sys.exit(1 if summary["tiles_with_issues"] > 0 else 0)


if __name__ == "__main__":
    main()
