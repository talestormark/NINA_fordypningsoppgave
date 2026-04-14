#!/usr/bin/env python3
"""
Visualize sparse point labels overlaid on tiles for thesis figures.

Produces a multi-panel figure showing:
- Column 1: S2 RGB composite (end year)
- Column 2: Dense ground truth mask
- Column 3: Sparse point labels overlaid on mask
- Column 4: RF prediction vs U-Net prediction (if available)

Each row is a different tile (low/moderate/high change level).
"""

import json
import numpy as np
import pandas as pd
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
PROCESSED_DIR = REPO_ROOT / "data" / "processed" / "epsg3035_10m_v2"
S2_DIR = PROCESSED_DIR / "sentinel"
MASK_DIR = PROCESSED_DIR / "masks"
AE_DIR = PROCESSED_DIR / "alphaearth"
SPLITS_CSV = REPO_ROOT / "preprocessing" / "outputs" / "splits" / "unified" / "split_info.csv"
SPARSE_LABELS_JSON = Path(__file__).resolve().parents[2] / "outputs" / "sparse_labels" / "sparse_labels_seed42.json"
OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "figures"

S2_N_TIMESTEPS = 14
S2_N_BANDS = 9
N_YEARS = 7


def load_s2_rgb_composite(refid, year_idx=6):
    """Load S2 RGB for a specific year (default: 2024, index 6)."""
    path = S2_DIR / f"{refid}_RGBNIRRSWIRQ_Mosaic.tif"
    with rasterio.open(path) as src:
        raw = src.read().astype(np.float64)
    all_ts = raw.reshape(S2_N_TIMESTEPS, S2_N_BANDS, raw.shape[1], raw.shape[2])
    # Annual composite (Q2+Q3 average)
    q2 = all_ts[year_idx * 2]
    q3 = all_ts[year_idx * 2 + 1]
    composite = (q2 + q3) / 2.0
    # RGB bands (indices 0, 1, 2) — scale for display
    rgb = composite[:3]  # (3, H, W)
    rgb = rgb.transpose(1, 2, 0)  # (H, W, 3)
    # Percentile stretch for visualization
    for c in range(3):
        p2, p98 = np.percentile(rgb[:, :, c], [2, 98])
        rgb[:, :, c] = np.clip((rgb[:, :, c] - p2) / (p98 - p2 + 1e-8), 0, 1)
    return rgb


def load_mask(refid):
    path = MASK_DIR / f"{refid}_mask.tif"
    with rasterio.open(path) as src:
        return (src.read(1) > 0).astype(np.float32)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(SPARSE_LABELS_JSON) as f:
        sparse = json.load(f)

    splits = pd.read_csv(SPLITS_CSV)

    # Select example tiles: one per change level
    examples = []
    for level in ['low', 'moderate', 'high']:
        sub = splits[(splits['split'] == 'train') & (splits['change_level'] == level)]
        sub = sub.sort_values('change_ratio')
        tile = sub.iloc[len(sub) // 2]
        examples.append(tile)

    # =========================================================================
    # Figure 1: Three-column visualization (RGB, Dense Mask, Sparse Points)
    # =========================================================================
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))

    for row, tile in enumerate(examples):
        refid = tile['refid']
        change_pct = tile['change_ratio']
        level = tile['change_level']

        rgb = load_s2_rgb_composite(refid)
        mask = load_mask(refid)
        points = sparse['tiles'].get(refid, [])

        H, W = mask.shape

        # Column 1: S2 RGB
        ax = axes[row, 0]
        ax.imshow(rgb)
        ax.set_title(f"Sentinel-2 RGB (2024)" if row == 0 else "")
        ax.set_ylabel(f"{level.capitalize()}\n({change_pct:.1f}% change)", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

        # Column 2: Dense mask
        ax = axes[row, 1]
        mask_vis = np.zeros((H, W, 3))
        mask_vis[mask > 0] = [0.85, 0.2, 0.2]  # red for change
        mask_vis[mask == 0] = [0.9, 0.9, 0.9]   # light grey for no change
        ax.imshow(mask_vis)
        n_change = int(mask.sum())
        ax.set_title(f"Dense mask ({n_change} px)" if row == 0 else f"({n_change} px)")
        ax.set_xticks([])
        ax.set_yticks([])

        # Column 3: Sparse points on grey background
        ax = axes[row, 2]
        bg = np.ones((H, W, 3)) * 0.85  # neutral grey background
        ax.imshow(bg)

        pos_y = [p[0] for p in points if p[2] == 1]
        pos_x = [p[1] for p in points if p[2] == 1]
        neg_y = [p[0] for p in points if p[2] == 0]
        neg_x = [p[1] for p in points if p[2] == 0]

        ax.scatter(neg_x, neg_y, c='steelblue', s=25, marker='o',
                   edgecolors='white', linewidths=0.5, zorder=3, label='No change')
        ax.scatter(pos_x, pos_y, c='firebrick', s=25, marker='o',
                   edgecolors='white', linewidths=0.5, zorder=3, label='Change')

        n_pts = len(points)
        n_pos = len(pos_y)
        ax.set_title(f"Sparse labels ({n_pts} pts)" if row == 0 else f"({n_pts} pts)")
        ax.set_xticks([])
        ax.set_yticks([])

        if row == 0:
            ax.legend(loc='lower right', fontsize=7, framealpha=0.9,
                      markerscale=1.2, handletextpad=0.3)

    plt.tight_layout(pad=1.5)
    fig.savefig(OUT_DIR / "sparse_labels_overview.pdf", dpi=300, bbox_inches='tight')
    fig.savefig(OUT_DIR / "sparse_labels_overview.png", dpi=200, bbox_inches='tight')
    print(f"Saved: {OUT_DIR / 'sparse_labels_overview.pdf'}")

    # =========================================================================
    # Figure 2: Dense vs Sparse comparison for one tile (detailed)
    # =========================================================================
    tile = examples[1]  # moderate change
    refid = tile['refid']
    rgb = load_s2_rgb_composite(refid)
    mask = load_mask(refid)
    points = sparse['tiles'].get(refid, [])
    H, W = mask.shape

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))

    # Panel a: S2 RGB (start year 2018)
    rgb_start = load_s2_rgb_composite(refid, year_idx=0)
    axes[0].imshow(rgb_start)
    axes[0].set_title("(a) S2 RGB 2018", fontsize=10)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # Panel b: S2 RGB (end year 2024)
    axes[1].imshow(rgb)
    axes[1].set_title("(b) S2 RGB 2024", fontsize=10)
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    # Panel c: Dense mask
    mask_vis = np.zeros((H, W, 3))
    mask_vis[mask > 0] = [0.85, 0.2, 0.2]
    mask_vis[mask == 0] = [0.9, 0.9, 0.9]
    axes[2].imshow(mask_vis)
    axes[2].set_title(f"(c) Dense mask\n({int(mask.sum())} change pixels)", fontsize=10)
    axes[2].set_xticks([])
    axes[2].set_yticks([])

    # Panel d: Sparse points overlaid on RGB
    axes[3].imshow(rgb, alpha=0.7)
    pos_y = [p[0] for p in points if p[2] == 1]
    pos_x = [p[1] for p in points if p[2] == 1]
    neg_y = [p[0] for p in points if p[2] == 0]
    neg_x = [p[1] for p in points if p[2] == 0]
    axes[3].scatter(neg_x, neg_y, c='steelblue', s=40, marker='o',
                    edgecolors='white', linewidths=0.8, zorder=3)
    axes[3].scatter(pos_x, pos_y, c='firebrick', s=40, marker='o',
                    edgecolors='white', linewidths=0.8, zorder=3)

    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor='firebrick', edgecolor='white', label=f'Change ({len(pos_y)} pts)'),
        mpatches.Patch(facecolor='steelblue', edgecolor='white', label=f'No change ({len(neg_y)} pts)'),
    ]
    axes[3].legend(handles=legend_elements, loc='lower right', fontsize=8, framealpha=0.9)
    axes[3].set_title(f"(d) Sparse labels\n({len(points)} points)", fontsize=10)
    axes[3].set_xticks([])
    axes[3].set_yticks([])

    plt.tight_layout(pad=1.0)
    fig.savefig(OUT_DIR / "dense_vs_sparse_detail.pdf", dpi=300, bbox_inches='tight')
    fig.savefig(OUT_DIR / "dense_vs_sparse_detail.png", dpi=200, bbox_inches='tight')
    print(f"Saved: {OUT_DIR / 'dense_vs_sparse_detail.pdf'}")

    # =========================================================================
    # Figure 3: Annotation effort comparison (conceptual)
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    # Dense annotation
    ax = axes[0]
    mask_vis = np.zeros((H, W, 4))
    mask_vis[mask > 0] = [0.85, 0.2, 0.2, 0.6]
    mask_vis[mask == 0] = [0.0, 0.0, 0.0, 0.0]
    ax.imshow(rgb)
    ax.imshow(mask_vis)
    # Draw polygon outline around change regions
    from scipy.ndimage import binary_dilation, binary_erosion
    boundary = binary_dilation(mask > 0, iterations=1) & ~(mask > 0)
    boundary_vis = np.zeros((H, W, 4))
    boundary_vis[boundary] = [1, 0, 0, 1]
    ax.imshow(boundary_vis)
    ax.set_title("Dense annotation\n(polygon delineation)", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])

    # Sparse annotation
    ax = axes[1]
    ax.imshow(rgb)
    ax.scatter(pos_x, pos_y, c='firebrick', s=50, marker='o',
               edgecolors='white', linewidths=1.0, zorder=3)
    ax.scatter(neg_x, neg_y, c='steelblue', s=50, marker='o',
               edgecolors='white', linewidths=1.0, zorder=3)
    ax.set_title(f"Sparse annotation\n({len(points)} point clicks)", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])

    legend_elements = [
        mpatches.Patch(facecolor='firebrick', edgecolor='white', label='Change'),
        mpatches.Patch(facecolor='steelblue', edgecolor='white', label='No change'),
    ]
    axes[1].legend(handles=legend_elements, loc='lower right', fontsize=9, framealpha=0.9)

    plt.tight_layout(pad=1.5)
    fig.savefig(OUT_DIR / "annotation_effort_comparison.pdf", dpi=300, bbox_inches='tight')
    fig.savefig(OUT_DIR / "annotation_effort_comparison.png", dpi=200, bbox_inches='tight')
    print(f"Saved: {OUT_DIR / 'annotation_effort_comparison.pdf'}")

    print(f"\nAll figures saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
