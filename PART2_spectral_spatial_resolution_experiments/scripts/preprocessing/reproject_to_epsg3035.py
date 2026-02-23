#!/usr/bin/env python3
"""
Reproject all modalities from EPSG:4326 to EPSG:3035 at 10m resolution.

Each tile's target grid is derived from its mask bounds, snapped outward to
10m-aligned coordinates. All modalities are warped to this identical grid so
that pixel indices correspond across sources.

Usage:
    python reproject_to_epsg3035.py                     # all tiles
    python reproject_to_epsg3035.py --refid <REFID>     # single tile
    python reproject_to_epsg3035.py --verify-only        # check outputs only
    python reproject_to_epsg3035.py --force              # overwrite existing
"""

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject, transform_bounds
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = REPO_ROOT / "data" / "raw"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "processed" / "epsg3035_10m_v1"

DST_CRS = "EPSG:3035"
DEFAULT_RESOLUTION = 10.0  # metres

# ---------------------------------------------------------------------------
# Modality configuration
# ---------------------------------------------------------------------------
MODALITIES = [
    {
        "name": "mask",
        "src_subdir": "Land_take_masks",
        "src_pattern": "{refid}_mask.tif",
        "dst_subdir": "masks",
        "resampling": Resampling.nearest,
        "required": True,
    },
    {
        "name": "sentinel",
        "src_subdir": "Sentinel",
        "src_pattern": "{refid}_RGBNIRRSWIRQ_Mosaic.tif",
        "dst_subdir": "sentinel",
        "resampling": Resampling.bilinear,
        "required": False,
    },
    {
        "name": "planetscope",
        "src_subdir": "PlanetScope",
        "src_pattern": "{refid}_RGBQ_Mosaic.tif",
        "dst_subdir": "planetscope_10m",
        "resampling": Resampling.average,
        "required": False,
    },
    {
        "name": "alphaearth",
        "src_subdir": "AlphaEarth",
        "src_pattern": "{refid}_VEY_Mosaic.tif",
        "dst_subdir": "alphaearth",
        "resampling": Resampling.bilinear,
        "required": False,
    },
]

EXPECTED_BANDS = {
    "mask": 1,
    "sentinel": 126,
    "planetscope": 42,
    "alphaearth": 448,
}

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def get_labeled_refids(raw_dir: Path) -> List[str]:
    """Discover tile refids from mask filenames."""
    mask_dir = raw_dir / "Land_take_masks"
    masks = sorted(mask_dir.glob("*_mask.tif"))
    refids = [p.stem.replace("_mask", "") for p in masks]
    log.info("Found %d labeled tiles", len(refids))
    return refids


def compute_target_grid(
    mask_path: Path, resolution: float
) -> Dict:
    """Derive EPSG:3035 target grid from mask bounds."""
    with rasterio.open(mask_path) as src:
        src_bounds = src.bounds  # in EPSG:4326

    b = transform_bounds(
        "EPSG:4326", DST_CRS,
        src_bounds.left, src_bounds.bottom, src_bounds.right, src_bounds.top,
    )

    left = math.floor(b[0] / resolution) * resolution
    bottom = math.floor(b[1] / resolution) * resolution
    right = math.ceil(b[2] / resolution) * resolution
    top = math.ceil(b[3] / resolution) * resolution

    width = int(round((right - left) / resolution))
    height = int(round((top - bottom) / resolution))

    transform = Affine(resolution, 0.0, left, 0.0, -resolution, top)

    return {
        "dst_crs": DST_CRS,
        "resolution_m": resolution,
        "bounds": {"left": left, "bottom": bottom, "right": right, "top": top},
        "size": {"width": width, "height": height},
        "width": width,
        "height": height,
        "transform": transform,
        "src_bounds_4326": {
            "left": src_bounds.left,
            "bottom": src_bounds.bottom,
            "right": src_bounds.right,
            "top": src_bounds.top,
        },
    }


def warp_raster(
    src_path: Path,
    grid: Dict,
    resampling: Resampling,
) -> np.ndarray:
    """Reproject a raster to the target grid, return destination array."""
    with rasterio.open(src_path) as src:
        src_data = src.read()
        dst_data = np.zeros(
            (src.count, grid["height"], grid["width"]),
            dtype=src.dtypes[0],
        )
        reproject(
            source=src_data,
            destination=dst_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=grid["transform"],
            dst_crs=grid["dst_crs"],
            resampling=resampling,
        )
    return dst_data


def write_output_tif(
    dst_path: Path,
    data: np.ndarray,
    grid: Dict,
    dtype: str,
) -> None:
    """Write a GeoTIFF with CRS, transform, and LZW compression."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    count = data.shape[0]
    profile = {
        "driver": "GTiff",
        "dtype": dtype,
        "width": grid["width"],
        "height": grid["height"],
        "count": count,
        "crs": grid["dst_crs"],
        "transform": grid["transform"],
        "compress": "lzw",
        "tiled": True,
    }
    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(data)


def save_grid_definition(
    grid: Dict, refid: str, output_dir: Path
) -> None:
    """Write per-tile JSON grid metadata."""
    grid_dir = output_dir / "grid_definitions"
    grid_dir.mkdir(parents=True, exist_ok=True)

    doc = {
        "refid": refid,
        "dst_crs": grid["dst_crs"],
        "resolution_m": grid["resolution_m"],
        "bounds": grid["bounds"],
        "size": grid["size"],
        "transform": list(grid["transform"])[:6],
        "src_bounds_4326": grid["src_bounds_4326"],
    }
    out_path = grid_dir / f"{refid}.json"
    with open(out_path, "w") as f:
        json.dump(doc, f, indent=2)


def process_tile(
    refid: str,
    raw_dir: Path,
    output_dir: Path,
    resolution: float,
    force: bool,
) -> Dict:
    """Reproject all modalities for one tile. Returns status dict."""
    status = {"refid": refid, "processed": [], "skipped": [], "missing": [], "errors": []}

    # --- Check mask exists (required) ---
    mask_path = raw_dir / "Land_take_masks" / f"{refid}_mask.tif"
    if not mask_path.exists():
        msg = f"Mask not found: {mask_path}"
        log.error(msg)
        status["errors"].append(msg)
        return status

    # --- Compute target grid from mask ---
    grid = compute_target_grid(mask_path, resolution)
    log.info(
        "  Grid: %d × %d px (%.0f × %.0f m)",
        grid["width"], grid["height"],
        grid["width"] * resolution, grid["height"] * resolution,
    )
    if grid["width"] < 64 or grid["height"] < 64:
        log.warning("  Tile %s is small: %d × %d px", refid, grid["width"], grid["height"])

    # --- Save grid definition ---
    save_grid_definition(grid, refid, output_dir)

    # --- Process each modality ---
    for mod in MODALITIES:
        src_filename = mod["src_pattern"].format(refid=refid)
        src_path = raw_dir / mod["src_subdir"] / src_filename
        dst_path = output_dir / mod["dst_subdir"] / src_filename

        if not src_path.exists():
            if mod["required"]:
                msg = f"Required file missing: {src_path}"
                log.error(msg)
                status["errors"].append(msg)
            else:
                log.warning("  %s: source not found, skipping", mod["name"])
                status["missing"].append(mod["name"])
            continue

        if dst_path.exists() and not force:
            log.info("  %s: already exists, skipping", mod["name"])
            status["skipped"].append(mod["name"])
            continue

        try:
            with rasterio.open(src_path) as src:
                src_dtype = src.dtypes[0]

            data = warp_raster(src_path, grid, mod["resampling"])
            write_output_tif(dst_path, data, grid, src_dtype)
            log.info("  %s: OK (%d bands, dtype=%s)", mod["name"], data.shape[0], src_dtype)
            status["processed"].append(mod["name"])
        except Exception as e:
            msg = f"{mod['name']}: {e}"
            log.error("  %s", msg)
            status["errors"].append(msg)

    return status


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_outputs(output_dir: Path, refids: List[str]) -> bool:
    """Post-hoc verification of all outputs. Returns True if all pass."""
    all_ok = True

    for refid in refids:
        tile_ok = True
        transforms = {}
        shapes = {}

        for mod in MODALITIES:
            src_filename = mod["src_pattern"].format(refid=refid)
            dst_path = output_dir / mod["dst_subdir"] / src_filename

            if not dst_path.exists():
                if mod["required"]:
                    log.error("VERIFY FAIL [%s] %s: file missing", refid, mod["name"])
                    tile_ok = False
                continue

            with rasterio.open(dst_path) as ds:
                # CRS check
                if str(ds.crs) != DST_CRS:
                    log.error(
                        "VERIFY FAIL [%s] %s: CRS=%s (expected %s)",
                        refid, mod["name"], ds.crs, DST_CRS,
                    )
                    tile_ok = False

                transforms[mod["name"]] = ds.transform
                shapes[mod["name"]] = (ds.width, ds.height)

                # Band count check
                expected = EXPECTED_BANDS.get(mod["name"])
                if expected is not None and ds.count != expected:
                    log.error(
                        "VERIFY FAIL [%s] %s: %d bands (expected %d)",
                        refid, mod["name"], ds.count, expected,
                    )
                    tile_ok = False

                # Minimum size check
                if ds.width < 64 or ds.height < 64:
                    log.warning(
                        "VERIFY WARN [%s] %s: small tile %d × %d",
                        refid, mod["name"], ds.width, ds.height,
                    )

                # Mask binary check
                if mod["name"] == "mask":
                    mask_data = ds.read(1)
                    unique = np.unique(mask_data)
                    if not np.all(np.isin(unique, [0, 1])):
                        log.error(
                            "VERIFY FAIL [%s] mask: non-binary values %s",
                            refid, unique,
                        )
                        tile_ok = False

        # Cross-modality consistency: all present modalities must share grid
        present = list(transforms.keys())
        if len(present) > 1:
            ref_name = present[0]
            ref_t = transforms[ref_name]
            ref_s = shapes[ref_name]
            for name in present[1:]:
                if transforms[name] != ref_t:
                    log.error(
                        "VERIFY FAIL [%s] transform mismatch: %s vs %s",
                        refid, ref_name, name,
                    )
                    tile_ok = False
                if shapes[name] != ref_s:
                    log.error(
                        "VERIFY FAIL [%s] shape mismatch: %s %s vs %s %s",
                        refid, ref_name, ref_s, name, shapes[name],
                    )
                    tile_ok = False

        if tile_ok:
            log.info("VERIFY OK [%s] (%d modalities)", refid, len(present))
        else:
            all_ok = False

    return all_ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproject tiles from EPSG:4326 to EPSG:3035 at 10m resolution.",
    )
    parser.add_argument(
        "--raw-dir", type=Path, default=DATA_DIR,
        help="Root of raw data directory (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Output directory (default: %(default)s)",
    )
    parser.add_argument(
        "--resolution", type=float, default=DEFAULT_RESOLUTION,
        help="Target resolution in metres (default: %(default)s)",
    )
    parser.add_argument(
        "--refid", type=str, default=None,
        help="Process a single tile by refid",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing outputs",
    )
    parser.add_argument(
        "--verify-only", action="store_true",
        help="Only run verification on existing outputs",
    )
    parser.add_argument(
        "--no-verify", action="store_true",
        help="Skip post-processing verification",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    log.info("Raw dir:    %s", args.raw_dir)
    log.info("Output dir: %s", args.output_dir)
    log.info("Resolution: %.1f m", args.resolution)
    log.info("DST CRS:    %s", DST_CRS)

    # --- Discover tiles ---
    all_refids = get_labeled_refids(args.raw_dir)
    if not all_refids:
        log.error("No tiles found in %s", args.raw_dir / "Land_take_masks")
        sys.exit(1)

    if args.refid:
        if args.refid not in all_refids:
            log.error("Refid %s not found in mask directory", args.refid)
            sys.exit(1)
        refids = [args.refid]
    else:
        refids = all_refids

    # --- Verify-only mode ---
    if args.verify_only:
        ok = verify_outputs(args.output_dir, refids)
        sys.exit(0 if ok else 1)

    # --- Process tiles ---
    log.info("Processing %d tile(s)...", len(refids))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for refid in tqdm(refids, desc="Reprojecting"):
        log.info("Tile: %s", refid)
        status = process_tile(refid, args.raw_dir, args.output_dir, args.resolution, args.force)
        results.append(status)

    # --- Summary ---
    n_processed = sum(1 for r in results if r["processed"])
    n_errors = sum(1 for r in results if r["errors"])
    log.info(
        "Done: %d tiles processed, %d with errors, %d total",
        n_processed, n_errors, len(results),
    )

    for r in results:
        if r["errors"]:
            log.error("  %s: %s", r["refid"], "; ".join(r["errors"]))
        if r["missing"]:
            log.warning("  %s: missing sources: %s", r["refid"], ", ".join(r["missing"]))

    # --- Verify ---
    if not args.no_verify:
        log.info("Running verification...")
        ok = verify_outputs(args.output_dir, refids)
        if not ok:
            log.error("Verification failed!")
            sys.exit(1)
        log.info("Verification passed.")


if __name__ == "__main__":
    main()
