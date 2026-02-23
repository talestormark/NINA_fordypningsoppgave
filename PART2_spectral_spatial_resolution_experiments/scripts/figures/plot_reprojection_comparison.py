#!/usr/bin/env python3
"""
Generate a side-by-side comparison of a tile before and after reprojection
from EPSG:4326 to EPSG:3035.

Produces a two-panel figure:
  Left:  EPSG:4326 RGB with non-square pixels (aspect='equal')
  Right: EPSG:3035 RGB with square 10m pixels, plus 64x64 crop box

Output: docs/REPORT/Images/Reprojection_before_after.pdf

Usage:
    python plot_reprojection_comparison.py
"""

import math
import warnings
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import rasterio

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed" / "epsg3035_10m_v1"
OUTPUT_DIR = (
    REPO_ROOT
    / "PART2_spectral_spatial_resolution_experiments"
    / "docs"
    / "REPORT"
    / "Images"
)

# Candidate tiles for the figure
TILES = {
    # lat ~52.1N, lon -0.5E: 38% non-square, significant rotation from LAEA center
    "rotated": "a-0-47134313698222_52-09242527089813",
    # lat ~48.6N, lon 10.3E: 34% non-square, near LAEA center → minimal rotation
    "aligned": "a10-27896225126639_48-57032226949789",
}

# S2 band layout: 9 bands/quarter, 2 quarters/year, 7 years = 126 bands
# Within each quarter: blue(0), green(1), red(2), RE1(3), ..., SWIR2(8)
BANDS_PER_QUARTER = 9
QUARTERS_PER_YEAR = 2
N_TIMESTEPS = 14  # 7 years × 2 quarters


def _temporal_mean_rgb(data):
    """Compute temporal mean RGB from all 14 timesteps. Returns (H, W, 3)."""
    H, W = data.shape[1], data.shape[2]
    rgb_stack = []
    for t in range(N_TIMESTEPS):
        base = t * BANDS_PER_QUARTER
        r, g, b = base + 2, base + 1, base + 0
        rgb_t = np.stack([data[r], data[g], data[b]], axis=-1).astype(np.float64)
        rgb_t[rgb_t == 0] = np.nan  # treat zero as missing
        rgb_stack.append(rgb_t)
    stacked = np.stack(rgb_stack, axis=0)  # (T, H, W, 3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(stacked, axis=0)  # (H, W, 3)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_s2_rgb_4326(refid):
    """Load raw EPSG:4326 tile, compute temporal mean RGB."""
    path = RAW_DIR / "Sentinel" / f"{refid}_RGBNIRRSWIRQ_Mosaic.tif"
    with rasterio.open(path) as ds:
        data = ds.read()  # (126, H, W)
        transform = ds.transform
        bounds = ds.bounds
    rgb = _temporal_mean_rgb(data)
    return rgb, transform, bounds


def load_s2_3035(refid):
    """Load reprojected EPSG:3035 tile. Returns all bands, mean RGB, transform, bounds."""
    path = PROCESSED_DIR / "sentinel" / f"{refid}_RGBNIRRSWIRQ_Mosaic.tif"
    with rasterio.open(path) as ds:
        data = ds.read()  # (126, H, W)
        transform = ds.transform
        bounds = ds.bounds
    rgb = _temporal_mean_rgb(data)
    return data, rgb, transform, bounds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def percentile_stretch(img, lo=2, hi=98):
    """Percentile stretch for display. Input (H, W, 3), output (H, W, 3) float32 [0,1]."""
    out = np.zeros_like(img, dtype=np.float32)
    for c in range(img.shape[2]):
        band = img[:, :, c].astype(np.float32)
        valid = band[band > 0]
        if len(valid) == 0:
            continue
        vmin = np.percentile(valid, lo)
        vmax = np.percentile(valid, hi)
        if vmax <= vmin:
            vmax = vmin + 1
        out[:, :, c] = np.clip((band - vmin) / (vmax - vmin), 0, 1)
    return out


def compute_pixel_size_meters_4326(transform, bounds):
    """Compute approximate pixel size in metres for an EPSG:4326 raster."""
    mid_lat = (bounds.bottom + bounds.top) / 2.0
    # 1 degree latitude ≈ 111320 m everywhere
    # 1 degree longitude ≈ 111320 * cos(lat) m
    dx_deg = abs(transform.a)  # pixel width in degrees (longitude)
    dy_deg = abs(transform.e)  # pixel height in degrees (latitude)
    dx_m = dx_deg * 111320 * math.cos(math.radians(mid_lat))
    dy_m = dy_deg * 111320
    return dx_m, dy_m


def compute_physical_footprint(bounds, crs_is_4326=True):
    """Compute approximate physical footprint in metres."""
    if crs_is_4326:
        mid_lat = (bounds.bottom + bounds.top) / 2.0
        width_m = (bounds.right - bounds.left) * 111320 * math.cos(math.radians(mid_lat))
        height_m = (bounds.top - bounds.bottom) * 111320
    else:
        width_m = bounds.right - bounds.left
        height_m = bounds.top - bounds.bottom
    return width_m, height_m


# ---------------------------------------------------------------------------
# Main plotting
# ---------------------------------------------------------------------------
def plot_comparison(refid, output_path):
    """Create the two-panel reprojection comparison figure."""
    # Load data
    rgb_4326, tf_4326, bounds_4326 = load_s2_rgb_4326(refid)
    allbands_3035, rgb_3035, tf_3035, bounds_3035 = load_s2_3035(refid)

    # Build nodata mask using ALL 126 bands.  Bilinear resampling blends
    # valid pixels with zero-fill near the footprint boundary, producing
    # small but non-zero fringe values.  Use a threshold on the sum of
    # absolute values across all bands to catch these.
    band_abs_sum = np.nansum(np.abs(allbands_3035), axis=0)  # (H, W)
    # Valid S2 pixels have TOA×10000 ≈ 200-5000 per band × 126 bands,
    # so the sum is typically > 10000.  Fringe pixels have sums < 100.
    nodata_3035 = band_abs_sum < 100

    # Replace NaN with 0 before stretch
    rgb_3035 = np.nan_to_num(rgb_3035, nan=0.0)

    # Stretch for display
    img_4326 = percentile_stretch(rgb_4326)
    img_3035 = percentile_stretch(rgb_3035)

    # Fill nodata with light gray to clearly distinguish it from imagery
    NODATA_COLOR = 0.88  # light gray
    img_3035[nodata_3035] = NODATA_COLOR

    h_4326, w_4326 = img_4326.shape[:2]
    h_3035, w_3035 = img_3035.shape[:2]

    # Pixel sizes
    dx_4326, dy_4326 = compute_pixel_size_meters_4326(tf_4326, bounds_4326)
    dx_3035 = abs(tf_3035.a)  # already in metres
    dy_3035 = abs(tf_3035.e)

    # Physical footprints
    fw_4326, fh_4326 = compute_physical_footprint(bounds_4326, crs_is_4326=True)
    fw_3035, fh_3035 = compute_physical_footprint(bounds_3035, crs_is_4326=False)

    # Width ratios so both panels have similar visual height
    ratio_left = w_4326 / h_4326
    ratio_right = w_3035 / h_3035
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2,
        figsize=(8, 3.8),
        gridspec_kw={"width_ratios": [ratio_left, ratio_right]},
    )

    # --- Left panel: EPSG:4326 ---
    ax_left.imshow(img_4326, aspect="equal", interpolation="nearest")
    ax_left.set_title("EPSG:4326 (WGS 84)", fontsize=11, fontweight="bold")
    ax_left.set_xticks([])
    ax_left.set_yticks([])

    info_4326 = (
        f"Pixel: {dx_4326:.1f} m $\\times$ {dy_4326:.1f} m\n"
        f"Raster: {w_4326} $\\times$ {h_4326} px\n"
        f"Footprint: ~{fw_4326:.0f} $\\times$ {fh_4326:.0f} m"
    )
    ax_left.text(
        0.03, 0.03, info_4326, transform=ax_left.transAxes,
        fontsize=7.5, verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="0.5", alpha=0.9),
    )

    # --- Right panel: EPSG:3035 (full grid, nodata as light gray) ---
    ax_right.imshow(img_3035, aspect="equal", interpolation="nearest")
    ax_right.set_title("EPSG:3035 (ETRS89-LAEA)", fontsize=11, fontweight="bold")
    ax_right.set_xticks([])
    ax_right.set_yticks([])

    info_3035 = (
        f"Pixel: {dx_3035:.0f} m $\\times$ {dy_3035:.0f} m\n"
        f"Raster: {w_3035} $\\times$ {h_3035} px\n"
        f"Footprint: {fw_3035:.0f} $\\times$ {fh_3035:.0f} m"
    )
    ax_right.text(
        0.03, 0.03, info_3035, transform=ax_right.transAxes,
        fontsize=7.5, verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="0.5", alpha=0.9),
    )

    # 64x64 crop box centred on the full 77×77 grid
    crop_size = 64
    cx = (w_3035 - crop_size) / 2.0
    cy = (h_3035 - crop_size) / 2.0
    rect = mpatches.Rectangle(
        (cx, cy), crop_size, crop_size,
        linewidth=1.5, edgecolor="cyan", facecolor="none",
        linestyle="--",
    )
    ax_right.add_patch(rect)
    ax_right.annotate(
        "Model input:\n64$\\times$64 px (640$\\times$640 m)",
        xy=(cx + 2, cy + 2), xycoords="data",
        fontsize=6.5, color="cyan", fontweight="bold",
        verticalalignment="top", horizontalalignment="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="black",
                  edgecolor="cyan", alpha=0.7, linewidth=0.8),
    )

    plt.tight_layout(pad=1.0)

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    # Also save PNG for easy visual check
    png_path = output_path.with_suffix(".png")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {output_path}")
    print(f"Saved: {png_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    for label, refid in TILES.items():
        suffix = "" if label == "aligned" else f"_{label}"
        output_path = OUTPUT_DIR / f"Reprojection_before_after{suffix}.pdf"
        print(f"\n[{label}] Generating comparison for: {refid}")
        print(f"  Output: {output_path}")
        plot_comparison(refid, output_path)

    # Copy the "aligned" version as the default (no suffix)
    print("\nDefault figure: aligned version (minimal rotation)")


if __name__ == "__main__":
    main()
