#!/usr/bin/env python3
"""Three-panel data-sources figure for the Data chapter.

Shows one example tile (Switzerland, Agriculture, 10.1% change) rendered
in each of the three remote-sensing sources for the year 2018:
    (a) Sentinel-2 RGB, Q2 2018 (true-colour, percentile-stretched)
    (b) AlphaEarth 2018 embedding, first three principal components as RGB
    (c) VHR Google, start-year (2018) RGB

Output:
    REPORT/Figures/3_Data_Study_Area/data_sources_a.{pdf,png}
    REPORT/Figures/3_Data_Study_Area/data_sources_b.{pdf,png}
    REPORT/Figures/3_Data_Study_Area/data_sources_c.{pdf,png}
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from pathlib import Path
from sklearn.decomposition import PCA

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
})

REPO = Path(__file__).resolve().parents[2]
S2_DIR  = REPO / "data_v2" / "Sentinel"
VHR_DIR = REPO / "data_v2" / "VHR_google"
AE_DIR  = REPO / "data_v2" / "AlphaEarth"
OUT_DIR = REPO / "REPORT" / "Figures" / "3_Data_Study_Area"

TILE = "a6-60060174716002_46-5717071263202"  # Switzerland, Agriculture, 10.1% change

# Sentinel-2 Q2 2018 RGB bands (1-indexed: blue=1, green=2, red=3)
S2_RGB_BANDS = [3, 2, 1]

# VHR 2018 (start-year) bands (1-indexed)
VHR_START_BANDS = [1, 2, 3]

# AlphaEarth 2018 features: 64 features, bands 1..64 (1-indexed)
AE_2018_BANDS = list(range(1, 1 + 64))


def stretch(arr, p_lo=2, p_hi=98, mask_zero=False):
    """Per-array percentile clip and normalise to [0, 1]."""
    finite = arr[np.isfinite(arr)]
    if mask_zero:
        finite = finite[finite > 0]
    if finite.size == 0:
        return np.zeros_like(arr)
    lo, hi = np.percentile(finite, [p_lo, p_hi])
    return np.clip((arr - lo) / max(hi - lo, 1e-9), 0, 1)


def save_panel(img, letter):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("0.5")
        spine.set_linewidth(0.5)
    plt.tight_layout(pad=0.1)
    for ext in ("pdf", "png"):
        out = OUT_DIR / f"data_sources_{letter}.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.02)
        print(f"Saved: {out}")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- (a) Sentinel-2 RGB Q2 2024 ---
    with rasterio.open(S2_DIR / f"{TILE}_RGBNIRRSWIRQ_Mosaic.tif") as ds:
        s2 = ds.read(S2_RGB_BANDS).astype(np.float32)
    s2_rgb = np.stack([stretch(s2[i]) for i in range(3)], axis=-1)
    save_panel(s2_rgb, "a")

    # --- (b) AlphaEarth 2018 PCA-to-RGB ---
    with rasterio.open(AE_DIR / f"{TILE}_VEY_Mosaic.tif") as ds:
        ae = ds.read(AE_2018_BANDS).astype(np.float32)
    C, H, W = ae.shape
    flat = ae.reshape(C, H * W).T  # (H*W, 64)
    pca = PCA(n_components=3)
    pcs = pca.fit_transform(flat)  # (H*W, 3)
    pcs = pcs.reshape(H, W, 3)
    ae_rgb = np.stack([stretch(pcs[..., i]) for i in range(3)], axis=-1)
    save_panel(ae_rgb, "b")

    # --- (c) VHR start-year (2018) RGB ---
    with rasterio.open(VHR_DIR / f"{TILE}_RGBY_Mosaic.tif") as ds:
        v = ds.read(VHR_START_BANDS).astype(np.float32)
    vhr_rgb = np.stack([stretch(v[i], mask_zero=True) for i in range(3)], axis=-1)
    save_panel(vhr_rgb, "c")

    print(f"\nTile: {TILE}")
    print(f"S2  shape: {s2.shape}, VHR shape: {v.shape}, AE shape: {ae.shape}")
    print(f"AE explained variance (first 3 PCs): {pca.explained_variance_ratio_}")


if __name__ == "__main__":
    main()
