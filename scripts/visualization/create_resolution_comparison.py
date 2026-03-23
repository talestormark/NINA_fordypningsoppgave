#!/usr/bin/env python3
"""
Create a figure comparing the four data sources at their native resolutions
for the same tile, highlighting the resolution differences.

Layout: single row with 4 columns (one per source), labels a)-d) below.

Output:
    - REPORT/figures/resolution_comparison.pdf
    - REPORT/figures/resolution_comparison.png
"""

import rasterio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_V2 = PROJECT_ROOT / "data_v2"
OUTPUT_DIR = PROJECT_ROOT / "REPORT" / "figures"

TILE = "a12-77039004735471_41-58543580125914"


def read_vhr_rgb(refid, year="end"):
    """Read VHR RGB for start or end year."""
    path = DATA_V2 / "VHR_google" / f"{refid}_RGBY_Mosaic.tif"
    with rasterio.open(path) as src:
        if year == "end":
            rgb = src.read([4, 5, 6])
        else:
            rgb = src.read([1, 2, 3])
    return np.moveaxis(rgb, 0, -1)


def read_ps_rgb(refid):
    """Read PlanetScope RGB (latest Q3 composite)."""
    path = DATA_V2 / "PlanetScope" / f"{refid}_RGBQ_Mosaic.tif"
    with rasterio.open(path) as src:
        rgb = src.read([40, 41, 42])
    return np.moveaxis(rgb, 0, -1)


def read_s2_rgb(refid):
    """Read Sentinel-2 true-colour RGB (latest Q3 composite)."""
    path = DATA_V2 / "Sentinel" / f"{refid}_RGBNIRRSWIRQ_Mosaic.tif"
    with rasterio.open(path) as src:
        bands = src.read([120, 119, 118])  # R, G, B
    rgb = np.moveaxis(bands, 0, -1).astype(np.float64)
    return rgb


def normalize_s2(rgb, low_pct=2, high_pct=98):
    """Percentile-based contrast stretch for S2."""
    out = np.zeros_like(rgb, dtype=np.float64)
    for i in range(3):
        band = rgb[:, :, i]
        valid = band[band > 0]
        if len(valid) == 0:
            continue
        lo = np.percentile(valid, low_pct)
        hi = np.percentile(valid, high_pct)
        if hi > lo:
            out[:, :, i] = np.clip((band - lo) / (hi - lo), 0, 1)
    return out


def read_ae_pca(refid):
    """Read AlphaEarth and compute PCA-3 for visualization."""
    path = DATA_V2 / "AlphaEarth" / f"{refid}_VEY_Mosaic.tif"
    with rasterio.open(path) as src:
        data = src.read()
    h, w = data.shape[1], data.shape[2]
    flat = data.reshape(data.shape[0], -1).T
    flat = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)

    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(flat)
    pca_img = pca_result.reshape(h, w, 3)

    for i in range(3):
        lo, hi = pca_img[:, :, i].min(), pca_img[:, :, i].max()
        if hi > lo:
            pca_img[:, :, i] = (pca_img[:, :, i] - lo) / (hi - lo)

    return pca_img


def create_figure():
    """Create resolution comparison: 1 row x 4 columns with a)-d) labels."""
    refid = TILE
    print(f"  Processing tile: {refid[:30]}...")

    labels = [
        "a) VHR Google imagery (1 m)",
        "b) PlanetScope (3–5 m)",
        "c) Sentinel-2 (10 m)",
        "d) AlphaEarth embeddings (10 m)",
    ]

    # Read all sources
    vhr = read_vhr_rgb(refid, year="end")
    ps = read_ps_rgb(refid)
    s2 = normalize_s2(read_s2_rgb(refid))
    ae = read_ae_pca(refid)

    images = [vhr, ps, s2, ae]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), dpi=300)

    for col, (img, label) in enumerate(zip(images, labels)):
        axes[col].imshow(img)
        axes[col].set_xlabel(label, fontsize=10, labelpad=8)
        axes[col].set_xticks([])
        axes[col].set_yticks([])
        for spine in axes[col].spines.values():
            spine.set_edgecolor("0.3")
            spine.set_linewidth(0.5)

    plt.tight_layout(w_pad=1.0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "png"]:
        path = OUTPUT_DIR / f"resolution_comparison.{ext}"
        plt.savefig(path, bbox_inches="tight", dpi=300)
        print(f"  Saved: {path}")

    plt.close(fig)


def main():
    print("=" * 60)
    print("CREATING RESOLUTION COMPARISON FIGURE")
    print("=" * 60)
    create_figure()
    print("\nDone!")


if __name__ == "__main__":
    main()
