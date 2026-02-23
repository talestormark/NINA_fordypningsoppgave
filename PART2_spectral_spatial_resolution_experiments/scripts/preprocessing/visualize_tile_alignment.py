#!/usr/bin/env python3
"""
Visualize a single tile across all data sources with mask overlay.

Produces a 2×5 panel figure:
  Top row:    S2 RGB | PS RGB | VHR RGB | AlphaEarth PCA | Mask
  Bottom row: S2+mask | PS+mask | VHR+mask | AE+mask    | Metadata

Usage:
    python visualize_tile_alignment.py --refid <REFID> [--output <path.png>]
    python visualize_tile_alignment.py --all  # generate for all labeled tiles
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap
from skimage.morphology import dilation, disk
from sklearn.decomposition import PCA

# ============================================================================
# Paths
# ============================================================================

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "data" / "raw"
SPLIT_CSV = REPO_ROOT / "outputs" / "splits" / "split_info.csv"
GEOJSON = REPO_ROOT / "land_take_bboxes_650m_v1.geojson"

SOURCES = {
    "sentinel": ("Sentinel", "{refid}_RGBNIRRSWIRQ_Mosaic.tif"),
    "planetscope": ("PlanetScope", "{refid}_RGBQ_Mosaic.tif"),
    "vhr": ("VHR_google", "{refid}_RGBY_Mosaic.tif"),
    "alphaearth": ("AlphaEarth", "{refid}_VEY_Mosaic.tif"),
    "mask": ("Land_take_masks", "{refid}_mask.tif"),
}

# ============================================================================
# Band indices (0-indexed)
# ============================================================================

# Sentinel-2: 126 bands = 7 years × 2 quarters × 9 bands
# Order per quarter: blue, green, red, R1, R2, R3, nir, swir1, swir2
S2_BANDS_PER_QUARTER = 9
S2_QUARTERS_PER_YEAR = 2
S2_BANDS_PER_YEAR = S2_BANDS_PER_QUARTER * S2_QUARTERS_PER_YEAR  # 18

# PlanetScope: 42 bands = 7 years × 2 quarters × 3 bands
# Order per quarter: blue, green, red
PS_BANDS_PER_QUARTER = 3
PS_QUARTERS_PER_YEAR = 2
PS_BANDS_PER_YEAR = PS_BANDS_PER_QUARTER * PS_QUARTERS_PER_YEAR  # 6


def get_s2_rgb_bands(year_idx, quarter_idx):
    """Get 0-indexed band indices for S2 red, green, blue."""
    start = year_idx * S2_BANDS_PER_YEAR + quarter_idx * S2_BANDS_PER_QUARTER
    # blue=0, green=1, red=2 within quarter
    return start + 2, start + 1, start + 0  # R, G, B


def get_ps_rgb_bands(year_idx, quarter_idx):
    """Get 0-indexed band indices for PS red, green, blue."""
    start = year_idx * PS_BANDS_PER_YEAR + quarter_idx * PS_BANDS_PER_QUARTER
    # blue=0, green=1, red=2 within quarter
    return start + 2, start + 1, start + 0  # R, G, B


# ============================================================================
# Data loading
# ============================================================================

def load_raster(source_key, refid):
    """Load a raster file, returning (data, bounds)."""
    folder, pattern = SOURCES[source_key]
    path = DATA_DIR / folder / pattern.format(refid=refid)
    if not path.exists():
        return None, None
    with rasterio.open(path) as ds:
        data = ds.read()  # (bands, H, W)
        bounds = ds.bounds  # BoundingBox(left, bottom, right, top)
    return data, bounds


def percentile_stretch(img, lo=2, hi=98):
    """Percentile stretch for display. Input (H, W, 3), output (H, W, 3) in [0,1]."""
    out = np.zeros_like(img, dtype=np.float32)
    for c in range(img.shape[2]):
        band = img[:, :, c].astype(np.float32)
        vmin = np.percentile(band[band > 0], lo) if np.any(band > 0) else 0
        vmax = np.percentile(band[band > 0], hi) if np.any(band > 0) else 1
        if vmax <= vmin:
            vmax = vmin + 1
        out[:, :, c] = np.clip((band - vmin) / (vmax - vmin), 0, 1)
    return out


def prepare_s2_rgb(data, year=2024, quarter="Q3"):
    """Extract and stretch S2 RGB for display."""
    year_idx = year - 2018
    quarter_idx = 0 if quarter == "Q2" else 1
    r, g, b = get_s2_rgb_bands(year_idx, quarter_idx)
    rgb = np.stack([data[r], data[g], data[b]], axis=-1)
    return percentile_stretch(rgb)


def prepare_ps_rgb(data, year=2024, quarter="Q3"):
    """Extract and stretch PS RGB for display."""
    year_idx = year - 2018
    quarter_idx = 0 if quarter == "Q2" else 1
    r, g, b = get_ps_rgb_bands(year_idx, quarter_idx)
    rgb = np.stack([data[r], data[g], data[b]], axis=-1)
    return percentile_stretch(rgb)


def prepare_vhr_rgb(data, timepoint="end"):
    """Extract VHR RGB. Bands: 0-2 = start (2018) R,G,B; 3-5 = end (2025) R,G,B."""
    if timepoint == "end":
        rgb = np.stack([data[3], data[4], data[5]], axis=-1)
    else:
        rgb = np.stack([data[0], data[1], data[2]], axis=-1)
    return rgb.astype(np.float32) / 255.0


def prepare_alphaearth_pca(data, year=2024):
    """PCA false-color from AlphaEarth 64-feature embedding for a given year."""
    year_idx = year - 2018
    # 448 bands = 64 features × 7 years, ordered by year
    start = year_idx * 64
    features = data[start:start + 64]  # (64, H, W)
    H, W = features.shape[1], features.shape[2]

    # Reshape to (pixels, features) for PCA
    flat = features.reshape(64, -1).T  # (H*W, 64)

    # Handle NaN/inf
    valid = np.isfinite(flat).all(axis=1)
    if valid.sum() < 10:
        return np.zeros((H, W, 3), dtype=np.float32)

    pca = PCA(n_components=3)
    pca.fit(flat[valid])
    transformed = np.zeros((H * W, 3), dtype=np.float32)
    transformed[valid] = pca.transform(flat[valid])

    rgb = transformed.reshape(H, W, 3)
    # Normalize each component to [0, 1]
    for c in range(3):
        ch = rgb[:, :, c]
        vmin, vmax = np.percentile(ch[valid.reshape(H, W)], [2, 98])
        if vmax <= vmin:
            vmax = vmin + 1
        rgb[:, :, c] = np.clip((ch - vmin) / (vmax - vmin), 0, 1)

    return rgb


def make_mask_overlay(mask, alpha=0.3):
    """Create RGBA overlay: red fill where mask=1."""
    H, W = mask.shape
    overlay = np.zeros((H, W, 4), dtype=np.float32)
    overlay[mask == 1, 0] = 1.0  # Red channel
    overlay[mask == 1, 3] = alpha
    return overlay


def make_mask_contour(mask):
    """Create mask boundary by dilating and XOR-ing with original."""
    dilated = dilation(mask.astype(bool), disk(1))
    boundary = dilated & ~mask.astype(bool)
    return boundary


# ============================================================================
# Metadata loading
# ============================================================================

def load_tile_metadata(refid):
    """Load metadata for a tile from split CSV and geojson."""
    meta = {"refid": refid}

    # Split info
    if SPLIT_CSV.exists():
        import csv
        with open(SPLIT_CSV) as f:
            for row in csv.DictReader(f):
                if row["refid"] == refid:
                    meta["change_ratio"] = f"{float(row['change_ratio']):.1f}%"
                    meta["change_level"] = row["change_level"]
                    meta["split"] = row["split"]
                    break

    # Geojson metadata
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
        import csv
        with open(SPLIT_CSV) as f:
            for row in csv.DictReader(f):
                refids.append(row["refid"])
    return refids


# ============================================================================
# Plotting
# ============================================================================

def plot_tile(refid, output_path=None):
    """Create the 2×5 alignment figure for one tile."""

    # Load all data
    s2_data, s2_bounds = load_raster("sentinel", refid)
    ps_data, ps_bounds = load_raster("planetscope", refid)
    vhr_data, vhr_bounds = load_raster("vhr", refid)
    ae_data, ae_bounds = load_raster("alphaearth", refid)
    mask_data, mask_bounds = load_raster("mask", refid)

    if mask_data is None:
        print(f"  Mask not found for {refid}, skipping.")
        return

    mask = mask_data[0]  # (H, W)
    mask_extent = [mask_bounds.left, mask_bounds.right,
                   mask_bounds.bottom, mask_bounds.top]

    # Prepare RGB images
    images = {}
    extents = {}
    labels_top = []
    labels_bot = []

    if s2_data is not None:
        images["s2"] = prepare_s2_rgb(s2_data)
        extents["s2"] = [s2_bounds.left, s2_bounds.right,
                         s2_bounds.bottom, s2_bounds.top]
        labels_top.append(f"Sentinel-2 RGB\n{s2_data.shape[2]}×{s2_data.shape[1]} px (~10 m)")
    else:
        labels_top.append("Sentinel-2\n(not available)")

    if ps_data is not None:
        images["ps"] = prepare_ps_rgb(ps_data)
        extents["ps"] = [ps_bounds.left, ps_bounds.right,
                         ps_bounds.bottom, ps_bounds.top]
        labels_top.append(f"PlanetScope RGB\n{ps_data.shape[2]}×{ps_data.shape[1]} px (~3 m)")
    else:
        labels_top.append("PlanetScope\n(not available)")

    if vhr_data is not None:
        images["vhr"] = prepare_vhr_rgb(vhr_data, "end")
        extents["vhr"] = [vhr_bounds.left, vhr_bounds.right,
                          vhr_bounds.bottom, vhr_bounds.top]
        labels_top.append(f"VHR Google RGB\n{vhr_data.shape[2]}×{vhr_data.shape[1]} px (~1 m)")
    else:
        labels_top.append("VHR Google\n(not available)")

    if ae_data is not None:
        images["ae"] = prepare_alphaearth_pca(ae_data)
        extents["ae"] = [ae_bounds.left, ae_bounds.right,
                         ae_bounds.bottom, ae_bounds.top]
        labels_top.append(f"AlphaEarth PCA\n{ae_data.shape[2]}×{ae_data.shape[1]} px (~10 m)")
    else:
        labels_top.append("AlphaEarth\n(not available)")

    # Metadata
    meta = load_tile_metadata(refid)

    # Create figure
    fig, axes = plt.subplots(2, 5, figsize=(24, 10))
    fig.suptitle(
        f"Tile: {refid}\n"
        f"{meta.get('country', '?')} | {meta.get('change_type', '?')} | "
        f"{meta.get('land_use', '?')} | {meta.get('change_ratio', '?')} change | "
        f"split: {meta.get('split', '?')}",
        fontsize=13, fontweight="bold", y=0.98
    )

    source_keys = ["s2", "ps", "vhr", "ae"]

    # --- Top row: raw imagery ---
    for col, key in enumerate(source_keys):
        ax = axes[0, col]
        if key in images:
            ax.imshow(images[key], extent=extents[key], aspect="auto",
                      interpolation="nearest")
        else:
            ax.text(0.5, 0.5, "Not available", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="gray")
        ax.set_title(labels_top[col], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    # Top-right: mask only
    ax_mask = axes[0, 4]
    mask_cmap = ListedColormap(["#1a1a2e", "#e63946"])
    ax_mask.imshow(mask, extent=mask_extent, aspect="auto",
                   cmap=mask_cmap, interpolation="nearest", vmin=0, vmax=1)
    ax_mask.set_title(f"Ground Truth Mask\n{mask.shape[1]}×{mask.shape[0]} px (~10 m)",
                      fontsize=10)
    ax_mask.set_xticks([])
    ax_mask.set_yticks([])

    # --- Bottom row: imagery + mask overlay ---
    mask_overlay = make_mask_overlay(mask, alpha=0.35)
    mask_boundary = make_mask_contour(mask)

    # Create boundary overlay (red edges)
    boundary_rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
    boundary_rgba[mask_boundary, 0] = 1.0  # Red
    boundary_rgba[mask_boundary, 3] = 1.0  # Fully opaque edges

    for col, key in enumerate(source_keys):
        ax = axes[1, col]
        if key in images:
            ax.imshow(images[key], extent=extents[key], aspect="auto",
                      interpolation="nearest")
            # Semi-transparent red fill
            ax.imshow(mask_overlay, extent=mask_extent, aspect="auto",
                      interpolation="nearest")
            # Red boundary contour
            ax.imshow(boundary_rgba, extent=mask_extent, aspect="auto",
                      interpolation="nearest")
        else:
            ax.text(0.5, 0.5, "Not available", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="gray")
        ax.set_title(f"{key.upper()} + mask overlay", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    # Bottom-right: metadata text box
    ax_meta = axes[1, 4]
    ax_meta.axis("off")

    meta_text = (
        f"Tile ID:\n  {refid[:30]}...\n\n"
        f"Country: {meta.get('country', '?')}\n"
        f"Land use: {meta.get('land_use', '?')}\n"
        f"Change type:\n  {meta.get('change_type', '?')}\n"
        f"Change ratio: {meta.get('change_ratio', '?')}\n"
        f"Change level: {meta.get('change_level', '?')}\n"
        f"Split: {meta.get('split', '?')}\n\n"
        f"Pixel grids:\n"
    )
    for key, label in [("s2", "S2"), ("ps", "PS"), ("vhr", "VHR"), ("ae", "AE")]:
        if key in images:
            h, w = images[key].shape[:2]
            meta_text += f"  {label}: {w}×{h}\n"
        else:
            meta_text += f"  {label}: N/A\n"
    meta_text += f"  Mask: {mask.shape[1]}×{mask.shape[0]}\n"

    # Add geographic bounds
    meta_text += f"\nBounds (EPSG:4326):\n"
    meta_text += f"  W: {mask_bounds.left:.4f}\n"
    meta_text += f"  E: {mask_bounds.right:.4f}\n"
    meta_text += f"  S: {mask_bounds.bottom:.4f}\n"
    meta_text += f"  N: {mask_bounds.top:.4f}\n"

    # Legend
    fill_patch = mpatches.Patch(facecolor="red", alpha=0.35, label="Mask fill")
    edge_patch = mpatches.Patch(facecolor="red", alpha=1.0, label="Mask edge")

    ax_meta.text(0.05, 0.95, meta_text, transform=ax_meta.transAxes,
                 fontsize=8, verticalalignment="top", fontfamily="monospace",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                           edgecolor="gray", alpha=0.9))
    ax_meta.legend(handles=[fill_patch, edge_patch], loc="lower center",
                   fontsize=9, frameon=True)

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight",
                    facecolor="white")
        print(f"  Saved: {output_path}")
    else:
        plt.show()

    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Visualize tile alignment across all data sources.")
    parser.add_argument("--refid", type=str, default=None,
                        help="Tile REFID to visualize")
    parser.add_argument("--output", type=str, default=None,
                        help="Output PNG path (default: auto-named in outputs/)")
    parser.add_argument("--all", action="store_true",
                        help="Generate for all labeled tiles")
    args = parser.parse_args()

    output_dir = (Path(__file__).resolve().parent.parent
                  / "outputs" / "tile_alignment")

    if args.all:
        refids = get_labeled_refids()
        print(f"Generating alignment figures for {len(refids)} tiles...")
        for i, refid in enumerate(refids):
            print(f"  [{i+1}/{len(refids)}] {refid}")
            out = output_dir / f"{refid}_alignment.png"
            plot_tile(refid, output_path=str(out))
        print(f"\nDone. Figures saved to: {output_dir}")

    elif args.refid:
        out = args.output or str(output_dir / f"{args.refid}_alignment.png")
        print(f"Generating alignment figure for: {args.refid}")
        plot_tile(args.refid, output_path=out)

    else:
        # Default: pick one test tile for demonstration
        refids = get_labeled_refids()
        # Pick a moderate-change test tile
        demo_refid = "a0-07602270798631_51-64536656448906"
        if demo_refid not in refids and refids:
            demo_refid = refids[0]
        out = str(output_dir / f"{demo_refid}_alignment.png")
        print(f"No --refid specified, using demo tile: {demo_refid}")
        plot_tile(demo_refid, output_path=out)


if __name__ == "__main__":
    main()
