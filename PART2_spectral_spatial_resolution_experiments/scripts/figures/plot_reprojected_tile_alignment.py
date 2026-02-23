#!/usr/bin/env python3
"""
Visualize a reprojected tile with before/after imagery and mask overlay.

Shows 2018 vs 2024 imagery for each source (S2, PS, AE) so the viewer can
verify that the change detection mask aligns with visible land-cover change.

Layout (3×3):
  Row 1: S2 2018 + contour | S2 2024 + contour  | Mask
  Row 2: PS 2018 + contour | PS 2024 + contour   | Mask (crop box)
  Row 3: AE 2018 + contour | AE 2024 + contour   | Metadata

All panels share the same EPSG:3035 @10m grid.

Usage:
    python plot_reprojected_tile_alignment.py
    python plot_reprojected_tile_alignment.py --refid <REFID>
    python plot_reprojected_tile_alignment.py --all
"""

import argparse
import csv
import json
import warnings
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap
from skimage.morphology import dilation, disk
from sklearn.decomposition import PCA

# ============================================================================
# Paths
# ============================================================================

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed" / "epsg3035_10m_v1"
SPLIT_CSV = REPO_ROOT / "outputs" / "splits" / "split_info.csv"
GEOJSON = REPO_ROOT / "land_take_bboxes_650m_v1.geojson"
OUTPUT_DIR = (
    REPO_ROOT
    / "PART2_spectral_spatial_resolution_experiments"
    / "docs"
    / "REPORT"
    / "Images"
)

SOURCES = {
    "sentinel": ("sentinel", "{refid}_RGBNIRRSWIRQ_Mosaic.tif"),
    "planetscope": ("planetscope_10m", "{refid}_RGBQ_Mosaic.tif"),
    "alphaearth": ("alphaearth", "{refid}_VEY_Mosaic.tif"),
    "mask": ("masks", "{refid}_mask.tif"),
}

# S2: 126 bands = 7 years × 2 quarters × 9 bands
# Band order per quarter: blue, green, red, RE1, RE2, RE3, NIR, SWIR1, SWIR2
S2_BANDS_PER_QUARTER = 9
S2_QUARTERS_PER_YEAR = 2

# PS: 42 bands = 7 years × 2 quarters × 3 bands
# Band order per quarter: blue, green, red
PS_BANDS_PER_QUARTER = 3
PS_QUARTERS_PER_YEAR = 2


# ============================================================================
# Data loading
# ============================================================================

def load_raster(source_key, refid):
    """Load a reprojected raster, returning (data, transform, bounds) or Nones."""
    folder, pattern = SOURCES[source_key]
    path = PROCESSED_DIR / folder / pattern.format(refid=refid)
    if not path.exists():
        return None, None, None
    with rasterio.open(path) as ds:
        data = ds.read()
        return data, ds.transform, ds.bounds


def percentile_stretch(img, lo=2, hi=98):
    """Percentile stretch for display. Input (H, W, 3), output [0,1]."""
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


def get_s2_rgb_for_year(data, year):
    """Extract S2 RGB for a given year by averaging Q2+Q3. Returns (H, W, 3)."""
    year_idx = year - 2018
    rgb_quarters = []
    for q in range(S2_QUARTERS_PER_YEAR):
        base = (year_idx * S2_QUARTERS_PER_YEAR + q) * S2_BANDS_PER_QUARTER
        rgb_q = np.stack([data[base + 2], data[base + 1], data[base + 0]], axis=-1).astype(np.float64)
        rgb_q[rgb_q == 0] = np.nan
        rgb_quarters.append(rgb_q)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.stack(rgb_quarters, axis=0), axis=0)


def get_ps_rgb_for_year(data, year):
    """Extract PS RGB for a given year by averaging Q2+Q3. Returns (H, W, 3)."""
    year_idx = year - 2018
    rgb_quarters = []
    for q in range(PS_QUARTERS_PER_YEAR):
        base = (year_idx * PS_QUARTERS_PER_YEAR + q) * PS_BANDS_PER_QUARTER
        rgb_q = np.stack([data[base + 2], data[base + 1], data[base + 0]], axis=-1).astype(np.float64)
        rgb_q[rgb_q == 0] = np.nan
        rgb_quarters.append(rgb_q)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.stack(rgb_quarters, axis=0), axis=0)


def prepare_alphaearth_pca(data, year=2024):
    """PCA false-color from AlphaEarth 64-feature embedding for a given year."""
    year_idx = year - 2018
    start = year_idx * 64
    features = data[start:start + 64]  # (64, H, W)
    H, W = features.shape[1], features.shape[2]

    flat = features.reshape(64, -1).T  # (H*W, 64)
    valid = np.isfinite(flat).all(axis=1) & (np.abs(flat).sum(axis=1) > 0)
    if valid.sum() < 10:
        return np.zeros((H, W, 3), dtype=np.float32)

    pca = PCA(n_components=3)
    pca.fit(flat[valid])
    transformed = np.zeros((H * W, 3), dtype=np.float32)
    transformed[valid] = pca.transform(flat[valid])

    rgb = transformed.reshape(H, W, 3)
    for c in range(3):
        ch = rgb[:, :, c]
        vals = ch[valid.reshape(H, W)]
        if len(vals) == 0:
            continue
        vmin, vmax = np.percentile(vals, [2, 98])
        if vmax <= vmin:
            vmax = vmin + 1
        rgb[:, :, c] = np.clip((ch - vmin) / (vmax - vmin), 0, 1)
    return rgb


def make_nodata_mask(data):
    """Border-fill mask: True where all bands are zero (reprojection fill).

    Uses a relative threshold: pixels whose abs-sum across all bands is less
    than 1% of the median valid-pixel abs-sum are treated as nodata.  This
    works for both S2 (large TOA values) and AlphaEarth (small floats).
    """
    abssum = np.nansum(np.abs(data.astype(np.float64)), axis=0)
    # Median of non-zero pixels as reference scale
    nonzero = abssum[abssum > 0]
    if len(nonzero) == 0:
        return np.ones(abssum.shape, dtype=bool)
    threshold = np.median(nonzero) * 0.01
    return abssum < threshold


def make_mask_contour(mask, thickness=2):
    """Mask boundary via repeated dilation for visibility."""
    dilated = dilation(mask.astype(bool), disk(thickness))
    return dilated & ~mask.astype(bool)


def apply_nodata_gray(img, nodata, gray=0.88):
    """Set nodata pixels to light gray."""
    img = img.copy()
    img[nodata] = gray
    return img


def draw_contour_overlay(ax, mask, color="red", thickness=2):
    """Draw mask boundary contour and semi-transparent fill on an axes."""
    H, W = mask.shape
    # Semi-transparent fill
    fill_rgba = np.zeros((H, W, 4), dtype=np.float32)
    fill_rgba[mask == 1, 0] = 1.0  # red
    fill_rgba[mask == 1, 3] = 0.25
    ax.imshow(fill_rgba, aspect="equal", interpolation="nearest")
    # Thick contour
    boundary = make_mask_contour(mask, thickness=thickness)
    contour_rgba = np.zeros((H, W, 4), dtype=np.float32)
    if color == "red":
        contour_rgba[boundary, 0] = 1.0
    elif color == "yellow":
        contour_rgba[boundary, 0] = 1.0
        contour_rgba[boundary, 1] = 1.0
    elif color == "cyan":
        contour_rgba[boundary, 1] = 1.0
        contour_rgba[boundary, 2] = 1.0
    contour_rgba[boundary, 3] = 1.0
    ax.imshow(contour_rgba, aspect="equal", interpolation="nearest")


# ============================================================================
# Metadata
# ============================================================================

def load_tile_metadata(refid):
    """Load metadata from split CSV and geojson."""
    meta = {"refid": refid}
    if SPLIT_CSV.exists():
        with open(SPLIT_CSV) as f:
            for row in csv.DictReader(f):
                if row["refid"] == refid:
                    meta["change_ratio"] = f"{float(row['change_ratio']):.1f}%"
                    meta["change_level"] = row["change_level"]
                    meta["split"] = row["split"]
                    break
    if GEOJSON.exists():
        with open(GEOJSON) as f:
            gj = json.load(f)
        for feat in gj["features"]:
            if feat["properties"]["PLOTID"] == refid:
                meta["country"] = feat["properties"].get("country", "?")
                meta["land_use"] = feat["properties"].get("r", "?")
                meta["change_type"] = feat["properties"].get("change_type", "?")
                break
    return meta


def get_labeled_refids():
    """Get all labeled refids from the split CSV."""
    refids = []
    if SPLIT_CSV.exists():
        with open(SPLIT_CSV) as f:
            for row in csv.DictReader(f):
                refids.append(row["refid"])
    return refids


# ============================================================================
# Plotting
# ============================================================================

def plot_tile(refid, output_path=None):
    """Create the 3×3 before/after alignment figure for one reprojected tile."""

    # Load all sources
    s2_data, s2_tf, s2_bounds = load_raster("sentinel", refid)
    ps_data, ps_tf, ps_bounds = load_raster("planetscope", refid)
    ae_data, ae_tf, ae_bounds = load_raster("alphaearth", refid)
    mask_data, mask_tf, mask_bounds = load_raster("mask", refid)

    if mask_data is None:
        print(f"  Mask not found for {refid}, skipping.")
        return

    mask = mask_data[0]  # (H, W)
    H, W = mask.shape
    meta = load_tile_metadata(refid)

    # --- Prepare before/after images ---
    panels = {}  # key -> (before_img, after_img, nodata_mask)

    if s2_data is not None:
        nodata = make_nodata_mask(s2_data)
        before = np.nan_to_num(get_s2_rgb_for_year(s2_data, 2018), nan=0.0)
        after = np.nan_to_num(get_s2_rgb_for_year(s2_data, 2024), nan=0.0)
        panels["s2"] = (
            apply_nodata_gray(percentile_stretch(before), nodata),
            apply_nodata_gray(percentile_stretch(after), nodata),
            nodata,
        )

    if ps_data is not None:
        nodata = make_nodata_mask(ps_data)
        before = np.nan_to_num(get_ps_rgb_for_year(ps_data, 2018), nan=0.0)
        after = np.nan_to_num(get_ps_rgb_for_year(ps_data, 2024), nan=0.0)
        panels["ps"] = (
            apply_nodata_gray(percentile_stretch(before), nodata),
            apply_nodata_gray(percentile_stretch(after), nodata),
            nodata,
        )

    if ae_data is not None:
        nodata = make_nodata_mask(ae_data)
        before = apply_nodata_gray(prepare_alphaearth_pca(ae_data, year=2018), nodata)
        after = apply_nodata_gray(prepare_alphaearth_pca(ae_data, year=2024), nodata)
        panels["ae"] = (before, after, nodata)

    # --- Figure: 3 rows × 3 cols ---
    fig, axes = plt.subplots(3, 3, figsize=(14, 13))
    fig.suptitle(
        f"Before / After with Change Mask  —  EPSG:3035 @ 10 m\n"
        f"{meta.get('country', '?')} | {meta.get('change_type', '?')} | "
        f"{meta.get('change_ratio', '?')} change | {meta.get('change_level', '?')} | "
        f"split: {meta.get('split', '?')}",
        fontsize=12, fontweight="bold", y=0.99,
    )

    row_sources = [
        ("s2", "Sentinel-2"),
        ("ps", "PlanetScope"),
        ("ae", "AlphaEarth"),
    ]
    contour_color = "yellow"

    for row, (key, label) in enumerate(row_sources):
        ax_before = axes[row, 0]
        ax_after = axes[row, 1]
        ax_right = axes[row, 2]

        if key in panels:
            before_img, after_img, nodata = panels[key]

            # Before panel
            ax_before.imshow(before_img, aspect="equal", interpolation="nearest")
            draw_contour_overlay(ax_before, mask, color=contour_color)
            ax_before.set_title(f"{label} — 2018", fontsize=10, fontweight="bold")

            # After panel
            ax_after.imshow(after_img, aspect="equal", interpolation="nearest")
            draw_contour_overlay(ax_after, mask, color=contour_color)
            ax_after.set_title(f"{label} — 2024", fontsize=10, fontweight="bold")
        else:
            ax_before.text(0.5, 0.5, f"{label}\nnot available",
                           ha="center", va="center", transform=ax_before.transAxes,
                           fontsize=11, color="gray")
            ax_after.text(0.5, 0.5, f"{label}\nnot available",
                          ha="center", va="center", transform=ax_after.transAxes,
                          fontsize=11, color="gray")
            ax_before.set_title(f"{label} — 2018", fontsize=10)
            ax_after.set_title(f"{label} — 2024", fontsize=10)

        for ax in [ax_before, ax_after]:
            ax.set_xticks([])
            ax.set_yticks([])

        # Right column
        ax_right.set_xticks([])
        ax_right.set_yticks([])

        if row == 0:
            # Mask panel
            mask_cmap = ListedColormap(["#1a1a2e", "#e63946"])
            ax_right.imshow(mask, aspect="equal", cmap=mask_cmap,
                            interpolation="nearest", vmin=0, vmax=1)
            ax_right.set_title(f"Change Mask\n{W}$\\times${H} px @ 10 m", fontsize=10,
                               fontweight="bold")

        elif row == 1:
            # Mask with 64×64 crop box
            mask_cmap = ListedColormap(["#1a1a2e", "#e63946"])
            ax_right.imshow(mask, aspect="equal", cmap=mask_cmap,
                            interpolation="nearest", vmin=0, vmax=1)
            crop_size = 64
            cx = (W - crop_size) / 2.0
            cy = (H - crop_size) / 2.0
            rect = mpatches.Rectangle(
                (cx, cy), crop_size, crop_size,
                linewidth=2, edgecolor="cyan", facecolor="none", linestyle="--",
            )
            ax_right.add_patch(rect)
            ax_right.annotate(
                f"64$\\times$64 crop\n(640$\\times$640 m)",
                xy=(cx + 2, cy + 2), fontsize=7.5, color="cyan", fontweight="bold",
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="black",
                          edgecolor="cyan", alpha=0.7, linewidth=0.8),
            )
            ax_right.set_title("Mask + Model Crop", fontsize=10, fontweight="bold")

        elif row == 2:
            # Metadata panel
            ax_right.axis("off")
            dx = abs(s2_tf.a) if s2_tf else "?"
            dy = abs(s2_tf.e) if s2_tf else "?"

            nodata_pct = ""
            if "s2" in panels:
                pct = 100.0 * panels["s2"][2].sum() / panels["s2"][2].size
                nodata_pct = f"Border-fill: {pct:.0f}% of pixels\n"

            meta_text = (
                f"Tile: {refid[:38]}...\n\n"
                f"Country:     {meta.get('country', '?')}\n"
                f"Land use:    {meta.get('land_use', '?')}\n"
                f"Change type: {meta.get('change_type', '?')}\n"
                f"Change:      {meta.get('change_ratio', '?')}\n"
                f"Split:       {meta.get('split', '?')}\n\n"
                f"Grid: EPSG:3035 @ {dx}x{dy} m\n"
                f"Raster: {W} x {H} px\n"
                f"Footprint: {W*10} x {H*10} m\n"
                f"{nodata_pct}"
            )

            ax_right.text(
                0.05, 0.95, meta_text, transform=ax_right.transAxes,
                fontsize=8.5, va="top", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                          edgecolor="gray", alpha=0.9),
            )

            # Legend
            fill_patch = mpatches.Patch(facecolor="red", alpha=0.25, label="Mask fill")
            contour_patch = mpatches.Patch(facecolor="yellow", alpha=1.0,
                                           label="Mask contour")
            gray_patch = mpatches.Patch(facecolor="0.88", edgecolor="0.5",
                                        label="No-data (border)")
            change_patch = mpatches.Patch(facecolor="#e63946", label="Change (mask=1)")
            nochange_patch = mpatches.Patch(facecolor="#1a1a2e", label="No change (mask=0)")

            ax_right.legend(
                handles=[fill_patch, contour_patch, gray_patch, change_patch, nochange_patch],
                loc="lower left", fontsize=7.5, frameon=True,
                bbox_to_anchor=(0.02, 0.02),
            )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

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


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Visualize reprojected tile before/after alignment (EPSG:3035 @10m).")
    parser.add_argument("--refid", type=str, default=None,
                        help="Tile REFID to visualize")
    parser.add_argument("--output", type=str, default=None,
                        help="Output PNG path")
    parser.add_argument("--all", action="store_true",
                        help="Generate for all labeled tiles")
    args = parser.parse_args()

    output_dir = OUTPUT_DIR

    if args.all:
        refids = get_labeled_refids()
        print(f"Generating before/after figures for {len(refids)} tiles...")
        for i, refid in enumerate(refids):
            print(f"  [{i+1}/{len(refids)}] {refid}")
            out = output_dir / f"Reprojected_alignment_{refid[:30]}.png"
            plot_tile(refid, output_path=str(out))
        print(f"\nDone. Figures saved to: {output_dir}")

    elif args.refid:
        out = args.output or str(output_dir / f"Reprojected_alignment_{args.refid[:30]}.png")
        print(f"Generating before/after figure for: {args.refid}")
        plot_tile(args.refid, output_path=out)

    else:
        refids = get_labeled_refids()
        demo_refid = "a10-27896225126639_48-57032226949789"
        if demo_refid not in refids and refids:
            demo_refid = refids[0]
        out = str(output_dir / "Reprojected_tile_alignment.png")
        print(f"Using demo tile: {demo_refid}")
        plot_tile(demo_refid, output_path=out)


if __name__ == "__main__":
    main()
