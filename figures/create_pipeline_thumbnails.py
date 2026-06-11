#!/usr/bin/env python3
"""Generate small thumbnails used in the Methodology figures.

One example test tile is used for every panel so the figures stay visually
coherent across the chapter.

Outputs (each as PDF + PNG):
    pipeline_thumb_s2.{pdf,png}          Sentinel-2 RGB at t_1 (2024 Q3)
    pipeline_thumb_s2_t0.{pdf,png}       Sentinel-2 RGB at t_0 (2018 Q3)
    pipeline_thumb_ae.{pdf,png}          AlphaEarth PCA-RGB at t_1 (2024)
    pipeline_thumb_mask.{pdf,png}        Ground-truth mask (full resolution)
    pipeline_thumb_prediction.{pdf,png}  Binary prediction (64x64)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from pathlib import Path
from sklearn.decomposition import PCA

from landtake.paths import REPO_ROOT as REPO
S2_DIR = REPO / "data_v2" / "Sentinel"
AE_DIR = REPO / "data_v2" / "AlphaEarth"
MASK_DIR = REPO / "data_v2" / "Land_take_masks_coarse"
PRED_DIR = (
    REPO
    / "experiments/exp2_input_representation"
    / "outputs"
    / "experiments"
    / "A3_s2_9band"
    / "predictions"
)
OUT_DIR = REPO / "REPORT" / "Figures" / "3_Methodology"

# Test tile with substantial change (~34% positive pixels). Same tile used in all
# four thumbnails so the pipeline is visually coherent across the chapter.
TILE = "a36-53163574265847_40-35433561270496"

# Sentinel-2: 126 bands = 14 timesteps x 9 spectral bands.
# Use end-year (2024) Q3 composite. Layout per dataset.py: T0..T13.
# t=12 is 2024 Q2, t=13 is 2024 Q3. Take Q3.
S2_N_BANDS = 9
S2_N_T = 14
# Timestep layout: t=0 is 2018 Q2, t=1 is 2018 Q3, ..., t=13 is 2024 Q3.
# We pick Q3 for both endpoints so the seasonal context is identical.
S2_T0_T = 1   # 2018 Q3
S2_T1_T = 13  # 2024 Q3
# RGB indices within a single timestep (1-indexed in rasterio):
# bands are [B,G,R,RE1,RE2,RE3,NIR,SWIR1,SWIR2].
# 1-indexed rasterio band index = t * 9 + spectral_idx + 1.
def _rgb_bands(t):
    return (t * S2_N_BANDS + 3, t * S2_N_BANDS + 2, t * S2_N_BANDS + 1)
S2_T0_R, S2_T0_G, S2_T0_B = _rgb_bands(S2_T0_T)
S2_T1_R, S2_T1_G, S2_T1_B = _rgb_bands(S2_T1_T)

# AlphaEarth: 448 bands = 7 years x 64 features. Use 2024 (last year).
AE_N_F = 64
AE_LAST_YEAR_START = 6 * AE_N_F + 1  # 1-indexed
AE_LAST_YEAR_BANDS = list(range(AE_LAST_YEAR_START, AE_LAST_YEAR_START + AE_N_F))


def stretch(arr, p_lo=2, p_hi=98):
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr)
    lo, hi = np.percentile(finite, [p_lo, p_hi])
    return np.clip((arr - lo) / max(hi - lo, 1e-9), 0, 1)


def save_thumb(rgb_array, name, vmin=None, vmax=None, cmap=None):
    """Save a square thumbnail. Removes all spines and ticks."""
    fig, ax = plt.subplots(figsize=(2.4, 2.4))
    if cmap is None:
        ax.imshow(rgb_array)
    else:
        ax.imshow(rgb_array, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("0.4")
        spine.set_linewidth(0.8)
    plt.tight_layout(pad=0.05)
    for ext in ("pdf", "png"):
        out = OUT_DIR / f"pipeline_thumb_{name}.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.02)
        print(f"Saved: {out}")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- (a) Sentinel-2 end-year RGB (t_1, 2024 Q3) ---
    with rasterio.open(S2_DIR / f"{TILE}_RGBNIRRSWIRQ_Mosaic.tif") as ds:
        s2_t1 = ds.read([S2_T1_R, S2_T1_G, S2_T1_B]).astype(np.float32)
        s2_t0 = ds.read([S2_T0_R, S2_T0_G, S2_T0_B]).astype(np.float32)
    s2_rgb_t1 = np.stack([stretch(s2_t1[i]) for i in range(3)], axis=-1)
    save_thumb(s2_rgb_t1, "s2")

    # --- (a') Sentinel-2 start-year RGB (t_0, 2018 Q3) ---
    s2_rgb_t0 = np.stack([stretch(s2_t0[i]) for i in range(3)], axis=-1)
    save_thumb(s2_rgb_t0, "s2_t0")

    # --- (b) AlphaEarth end-year PCA-RGB ---
    with rasterio.open(AE_DIR / f"{TILE}_VEY_Mosaic.tif") as ds:
        ae = ds.read(AE_LAST_YEAR_BANDS).astype(np.float32)
    C, H, W = ae.shape
    flat = ae.reshape(C, H * W).T
    pca = PCA(n_components=3)
    pcs = pca.fit_transform(flat).reshape(H, W, 3)
    ae_rgb = np.stack([stretch(pcs[..., i]) for i in range(3)], axis=-1)
    save_thumb(ae_rgb, "ae")

    # --- (c) Ground-truth mask (full tile resolution) ---
    with rasterio.open(MASK_DIR / f"{TILE}_mask.tif") as ds:
        mask = ds.read(1).astype(np.float32)
    save_thumb(mask, "mask", vmin=0, vmax=1, cmap="Reds")

    # --- (d) Binary prediction at the 64x64 cropped resolution ---
    npz = np.load(PRED_DIR / f"{TILE}.npz")
    pred = (npz["prob"] > 0.5).astype(np.float32)
    save_thumb(pred, "prediction", vmin=0, vmax=1, cmap="Reds")


if __name__ == "__main__":
    main()
