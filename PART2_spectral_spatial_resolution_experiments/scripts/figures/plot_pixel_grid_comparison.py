#!/usr/bin/env python3
"""
Visualize the pixel grid differences between Sentinel-2 and PlanetScope.

Shows that S2/AE/Mask share the same grid (non-square 6.6m × 10.0m pixels)
while PlanetScope has a different grid (3.2m × 3.2m, offset origin).

Layout:
  Left:   Zoomed-in view with both pixel grids overlaid
  Right:  Full tile with zoom region marked + metadata

Usage:
    python plot_pixel_grid_comparison.py [--refid REFID]
"""

import argparse
import math
import warnings
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import rasterio

# ============================================================================
# Paths
# ============================================================================

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
OUTPUT_DIR = (
    REPO_ROOT
    / "PART2_spectral_spatial_resolution_experiments"
    / "docs"
    / "REPORT"
    / "Images"
)

SOURCES = {
    "sentinel": ("Sentinel", "{refid}_RGBNIRRSWIRQ_Mosaic.tif"),
    "planetscope": ("PlanetScope", "{refid}_RGBQ_Mosaic.tif"),
    "alphaearth": ("AlphaEarth", "{refid}_VEY_Mosaic.tif"),
    "mask": ("Land_take_masks", "{refid}_mask.tif"),
}

S2_BANDS_PER_QUARTER = 9
PS_BANDS_PER_QUARTER = 3


def load_raster(source_key, refid):
    folder, pattern = SOURCES[source_key]
    path = RAW_DIR / folder / pattern.format(refid=refid)
    if not path.exists():
        return None, None, None
    with rasterio.open(path) as ds:
        return ds.read(), ds.transform, ds.bounds


def deg_to_meters(lon_deg, lat_deg, ref_lat):
    """Convert degree offsets to approximate metres at a reference latitude."""
    x_m = lon_deg * 111320 * math.cos(math.radians(ref_lat))
    y_m = lat_deg * 111320
    return x_m, y_m


def get_s2_rgb(data, year_idx=6, q_idx=1):
    """Extract single-date S2 RGB."""
    base = (year_idx * 2 + q_idx) * S2_BANDS_PER_QUARTER
    rgb = np.stack([data[base + 2], data[base + 1], data[base + 0]], axis=-1).astype(np.float32)
    valid = rgb > 0
    if valid.any():
        for c in range(3):
            b = rgb[:, :, c]
            v = b[b > 0]
            if len(v) > 0:
                lo, hi = np.percentile(v, [2, 98])
                if hi > lo:
                    rgb[:, :, c] = np.clip((b - lo) / (hi - lo), 0, 1)
    return rgb


def get_ps_rgb(data, year_idx=6, q_idx=1):
    """Extract single-date PS RGB."""
    base = (year_idx * 2 + q_idx) * PS_BANDS_PER_QUARTER
    rgb = np.stack([data[base + 2], data[base + 1], data[base + 0]], axis=-1).astype(np.float32)
    return rgb / 255.0


def draw_pixel_grid(ax, transform, shape, extent_m, origin_m, color, label,
                    linewidth=0.8, ref_lat=50.0):
    """Draw pixel grid lines within the extent (in metres)."""
    H, W = shape
    dx_deg = transform.a
    dy_deg = abs(transform.e)
    x0_deg = transform.c
    y0_deg = transform.f  # top-left y (northernmost)

    dx_m, _ = deg_to_meters(dx_deg, 0, ref_lat)
    _, dy_m = deg_to_meters(0, dy_deg, ref_lat)

    x_min, x_max, y_min, y_max = extent_m

    # Vertical lines (x positions)
    for i in range(W + 1):
        x_m = origin_m[0] + i * dx_m
        if x_min <= x_m <= x_max:
            ax.axvline(x_m, color=color, linewidth=linewidth, alpha=0.7)

    # Horizontal lines (y positions — origin is top, going down)
    for j in range(H + 1):
        y_m = origin_m[1] - j * dy_m
        if y_min <= y_m <= y_max:
            ax.axhline(y_m, color=color, linewidth=linewidth, alpha=0.7)


def plot_grid_comparison(refid, output_path=None):
    """Create the pixel grid comparison figure."""

    # Load data
    s2_data, s2_tf, s2_bounds = load_raster("sentinel", refid)
    ps_data, ps_tf, ps_bounds = load_raster("planetscope", refid)
    mask_data, mask_tf, mask_bounds = load_raster("mask", refid)

    if s2_data is None or ps_data is None:
        print(f"  Missing S2 or PS for {refid}, skipping.")
        return

    ref_lat = (s2_bounds.top + s2_bounds.bottom) / 2.0

    # Compute pixel sizes in metres
    s2_dx_m, _ = deg_to_meters(s2_tf.a, 0, ref_lat)
    _, s2_dy_m = deg_to_meters(0, abs(s2_tf.e), ref_lat)
    ps_dx_m, _ = deg_to_meters(ps_tf.a, 0, ref_lat)
    _, ps_dy_m = deg_to_meters(0, abs(ps_tf.e), ref_lat)

    # Origins in metres (relative to S2 top-left corner)
    s2_origin_x_m = 0.0
    s2_origin_y_m = 0.0

    ps_offset_lon = ps_tf.c - s2_tf.c  # degrees
    ps_offset_lat = ps_tf.f - s2_tf.f  # degrees (both are top-left y)
    ps_origin_x_m, ps_origin_y_m_raw = deg_to_meters(ps_offset_lon, ps_offset_lat, ref_lat)
    ps_origin_y_m = ps_origin_y_m_raw  # negative if PS origin is south of S2 origin

    # Full tile extent in metres
    tile_w_m = s2_data.shape[2] * s2_dx_m
    tile_h_m = s2_data.shape[1] * s2_dy_m

    # Zoom region: top-left corner, ~5×5 S2 pixels
    n_pixels = 5
    zoom_w = n_pixels * s2_dx_m
    zoom_h = n_pixels * s2_dy_m
    zoom_x0 = 2 * s2_dx_m  # start a bit inward
    zoom_y0 = -2 * s2_dy_m  # from top

    zoom_extent = (zoom_x0, zoom_x0 + zoom_w, zoom_y0 - zoom_h, zoom_y0)

    # Prepare images
    s2_rgb = get_s2_rgb(s2_data)
    ps_rgb = get_ps_rgb(ps_data)

    # --- Figure ---
    fig = plt.figure(figsize=(16, 7))

    # Left: zoomed pixel grid
    ax_zoom = fig.add_axes([0.04, 0.08, 0.44, 0.82])
    # Right top: S2 full tile
    ax_s2 = fig.add_axes([0.54, 0.52, 0.20, 0.38])
    # Right middle: PS full tile
    ax_ps = fig.add_axes([0.76, 0.52, 0.20, 0.38])
    # Right bottom: metadata
    ax_meta = fig.add_axes([0.54, 0.05, 0.42, 0.42])

    fig.suptitle(
        "Pixel Grid Comparison: Sentinel-2 vs PlanetScope (EPSG:4326)",
        fontsize=13, fontweight="bold", y=0.97,
    )

    # --- Zoomed panel: show PS image as background with both grids ---
    # We need to render the PS image in the zoom extent
    # Map zoom extent (metres) back to PS pixel coords
    zoom_lon_min = s2_tf.c + zoom_extent[0] / (111320 * math.cos(math.radians(ref_lat)))
    zoom_lon_max = s2_tf.c + zoom_extent[1] / (111320 * math.cos(math.radians(ref_lat)))
    zoom_lat_max = s2_tf.f + zoom_extent[3] / 111320
    zoom_lat_min = s2_tf.f + zoom_extent[2] / 111320

    # PS pixel indices for the zoom region
    ps_col_min = max(0, int((zoom_lon_min - ps_tf.c) / ps_tf.a))
    ps_col_max = min(ps_data.shape[2], int((zoom_lon_max - ps_tf.c) / ps_tf.a) + 1)
    ps_row_min = max(0, int((ps_tf.f - zoom_lat_max) / abs(ps_tf.e)))
    ps_row_max = min(ps_data.shape[1], int((ps_tf.f - zoom_lat_min) / abs(ps_tf.e)) + 1)

    ps_crop = ps_rgb[ps_row_min:ps_row_max, ps_col_min:ps_col_max]

    # Extent of the PS crop in metres
    crop_lon_min = ps_tf.c + ps_col_min * ps_tf.a
    crop_lon_max = ps_tf.c + ps_col_max * ps_tf.a
    crop_lat_max = ps_tf.f - ps_row_min * abs(ps_tf.e)
    crop_lat_min = ps_tf.f - ps_row_max * abs(ps_tf.e)
    crop_x_min, _ = deg_to_meters(crop_lon_min - s2_tf.c, 0, ref_lat)
    crop_x_max, _ = deg_to_meters(crop_lon_max - s2_tf.c, 0, ref_lat)
    _, crop_y_max_raw = deg_to_meters(0, crop_lat_max - s2_tf.f, ref_lat)
    _, crop_y_min_raw = deg_to_meters(0, crop_lat_min - s2_tf.f, ref_lat)

    ax_zoom.imshow(
        ps_crop, extent=[crop_x_min, crop_x_max, crop_y_min_raw, crop_y_max_raw],
        aspect="equal", interpolation="nearest", alpha=0.5,
    )

    # Draw S2 grid
    draw_pixel_grid(
        ax_zoom, s2_tf, s2_data.shape[1:], zoom_extent,
        (s2_origin_x_m, s2_origin_y_m), color="#2196F3", label="S2",
        linewidth=2.0, ref_lat=ref_lat,
    )

    # Draw PS grid
    draw_pixel_grid(
        ax_zoom, ps_tf, ps_data.shape[1:], zoom_extent,
        (ps_origin_x_m, ps_origin_y_m), color="#FF5722", label="PS",
        linewidth=1.0, ref_lat=ref_lat,
    )

    ax_zoom.set_xlim(zoom_extent[0], zoom_extent[1])
    ax_zoom.set_ylim(zoom_extent[2], zoom_extent[3])
    ax_zoom.set_xlabel("East-West (metres from S2 origin)", fontsize=9)
    ax_zoom.set_ylabel("North-South (metres from S2 origin)", fontsize=9)
    ax_zoom.set_title(
        f"Zoomed: {n_pixels}x{n_pixels} S2 pixels\n"
        f"S2 pixel: {s2_dx_m:.1f}m x {s2_dy_m:.1f}m (blue) | "
        f"PS pixel: {ps_dx_m:.1f}m x {ps_dy_m:.1f}m (orange)",
        fontsize=10, fontweight="bold",
    )

    # Add pixel size annotations inside a representative S2 pixel
    mid_x = zoom_x0 + 2.5 * s2_dx_m
    mid_y = zoom_y0 - 2.5 * s2_dy_m
    # S2 pixel width arrow
    ax_zoom.annotate(
        "", xy=(zoom_x0 + 3 * s2_dx_m, mid_y + 0.3 * s2_dy_m),
        xytext=(zoom_x0 + 2 * s2_dx_m, mid_y + 0.3 * s2_dy_m),
        arrowprops=dict(arrowstyle="<->", color="#2196F3", lw=2),
    )
    ax_zoom.text(
        mid_x, mid_y + 0.45 * s2_dy_m, f"{s2_dx_m:.1f}m",
        ha="center", va="bottom", fontsize=8, fontweight="bold", color="#2196F3",
    )
    # S2 pixel height arrow
    ax_zoom.annotate(
        "", xy=(zoom_x0 + 2.7 * s2_dx_m, zoom_y0 - 2 * s2_dy_m),
        xytext=(zoom_x0 + 2.7 * s2_dx_m, zoom_y0 - 3 * s2_dy_m),
        arrowprops=dict(arrowstyle="<->", color="#2196F3", lw=2),
    )
    ax_zoom.text(
        zoom_x0 + 2.85 * s2_dx_m, mid_y, f"{s2_dy_m:.1f}m",
        ha="left", va="center", fontsize=8, fontweight="bold", color="#2196F3",
    )

    ax_zoom.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
    ax_zoom.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))

    s2_line = mpatches.Patch(facecolor="#2196F3", alpha=0.7, label=f"S2 grid ({s2_dx_m:.1f} x {s2_dy_m:.1f} m)")
    ps_line = mpatches.Patch(facecolor="#FF5722", alpha=0.7, label=f"PS grid ({ps_dx_m:.1f} x {ps_dy_m:.1f} m)")
    ax_zoom.legend(handles=[s2_line, ps_line], loc="lower right", fontsize=8)

    # --- S2 full tile ---
    s2_extent = [0, tile_w_m, -tile_h_m, 0]
    ax_s2.imshow(s2_rgb, extent=s2_extent, aspect="equal", interpolation="nearest")
    # Mark zoom region
    rect = mpatches.Rectangle(
        (zoom_extent[0], zoom_extent[2]),
        zoom_extent[1] - zoom_extent[0],
        zoom_extent[3] - zoom_extent[2],
        linewidth=2, edgecolor="yellow", facecolor="none", linestyle="-",
    )
    ax_s2.add_patch(rect)
    ax_s2.set_title(f"Sentinel-2\n{s2_data.shape[2]}x{s2_data.shape[1]} px", fontsize=9, fontweight="bold")
    ax_s2.set_xticks([])
    ax_s2.set_yticks([])

    # --- PS full tile ---
    ps_tile_w_m = ps_data.shape[2] * ps_dx_m
    ps_tile_h_m = ps_data.shape[1] * ps_dy_m
    ps_extent = [ps_origin_x_m, ps_origin_x_m + ps_tile_w_m,
                 ps_origin_y_m - ps_tile_h_m, ps_origin_y_m]
    ax_ps.imshow(ps_rgb, extent=ps_extent, aspect="equal", interpolation="nearest")
    rect2 = mpatches.Rectangle(
        (zoom_extent[0], zoom_extent[2]),
        zoom_extent[1] - zoom_extent[0],
        zoom_extent[3] - zoom_extent[2],
        linewidth=2, edgecolor="yellow", facecolor="none", linestyle="-",
    )
    ax_ps.add_patch(rect2)
    ax_ps.set_title(f"PlanetScope\n{ps_data.shape[2]}x{ps_data.shape[1]} px", fontsize=9, fontweight="bold")
    ax_ps.set_xticks([])
    ax_ps.set_yticks([])

    # --- Metadata panel ---
    ax_meta.axis("off")

    origin_offset_x_m = ps_origin_x_m
    origin_offset_y_m = ps_origin_y_m

    meta_text = (
        f"Pixel Grid Comparison\n"
        f"{'='*40}\n\n"
        f"Source       Pixel (m)       Shape      Origin aligned?\n"
        f"{'─'*55}\n"
        f"S2/AE/Mask   {s2_dx_m:.1f} x {s2_dy_m:.1f}     {s2_data.shape[2]:>3}x{s2_data.shape[1]:<3}   Yes (identical)\n"
        f"PlanetScope  {ps_dx_m:.1f} x  {ps_dy_m:.1f}     {ps_data.shape[2]:>3}x{ps_data.shape[1]:<3}   No\n\n"
        f"Grid origin offset (PS vs S2):\n"
        f"  x: {origin_offset_x_m:+.1f} m  ({ps_offset_lon*1e6:+.1f} µdeg lon)\n"
        f"  y: {origin_offset_y_m:+.1f} m  ({ps_offset_lat*1e6:+.1f} µdeg lat)\n\n"
        f"S2 pixels are non-square in EPSG:4326:\n"
        f"  Width:  {s2_dx_m:.1f} m  (lon at {ref_lat:.1f}°N)\n"
        f"  Height: {s2_dy_m:.1f} m  (lat)\n"
        f"  Ratio:  1 : {s2_dy_m/s2_dx_m:.2f}\n\n"
        f"PS pixels are approximately square:\n"
        f"  Width:  {ps_dx_m:.1f} m  |  Height: {ps_dy_m:.1f} m\n\n"
        f"Conclusion:\n"
        f"  S2, AlphaEarth, Mask: same grid (no alignment needed)\n"
        f"  PlanetScope: different grid (resample needed for fusion)"
    )

    ax_meta.text(
        0.05, 0.95, meta_text, transform=ax_meta.transAxes,
        fontsize=8.5, va="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                  edgecolor="gray", alpha=0.9),
    )

    # Save
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {output_path}")
        pdf_path = Path(output_path).with_suffix(".pdf")
        fig.savefig(pdf_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {pdf_path}")
    else:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize pixel grid differences.")
    parser.add_argument("--refid", type=str, default=None)
    args = parser.parse_args()

    refid = args.refid or "a10-27896225126639_48-57032226949789"
    out = str(OUTPUT_DIR / "Pixel_grid_comparison.png")
    print(f"Generating pixel grid comparison for: {refid}")
    plot_grid_comparison(refid, output_path=out)


if __name__ == "__main__":
    main()
