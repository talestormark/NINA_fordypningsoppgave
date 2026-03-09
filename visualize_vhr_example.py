#!/usr/bin/env python3
"""
Visualize a VHR land-take example: pre (2018), post (2025), and binary mask.

Produces a 3-panel figure suitable for report/presentation use.
"""

import rasterio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import zoom

# ── Configuration ────────────────────────────────────────────────────────────
# Tile with highest change (64% positive pixels)
REFID = "a-2-52025858362194_53-72050616794933"

DATA_DIR = Path(__file__).parent / "data" / "raw"
VHR_DIR = DATA_DIR / "VHR_google"
MASK_DIR = DATA_DIR / "Land_take_masks"
OUTPUT_PATH = Path(__file__).parent / "vhr_example.png"

# ── Load data ────────────────────────────────────────────────────────────────
vhr_path = VHR_DIR / f"{REFID}_RGBY_Mosaic.tif"
mask_path = MASK_DIR / f"{REFID}_mask.tif"

with rasterio.open(vhr_path) as src:
    vhr = src.read()  # (6, H, W): bands 0-2 = 2018 RGB, bands 3-5 = 2025 RGB

img_2018 = vhr[0:3].transpose(1, 2, 0)  # (H, W, 3) uint8
img_2025 = vhr[3:6].transpose(1, 2, 0)  # (H, W, 3) uint8

with rasterio.open(mask_path) as src:
    mask = src.read(1)  # (H_mask, W_mask)

mask = (mask > 0).astype(np.uint8)

# Resample mask to VHR resolution if needed
if mask.shape != img_2018.shape[:2]:
    zoom_factors = (img_2018.shape[0] / mask.shape[0],
                    img_2018.shape[1] / mask.shape[1])
    mask = zoom(mask, zoom_factors, order=0)

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(img_2018)
axes[0].set_title("2018 (pre)", fontsize=14)
axes[0].axis("off")

axes[1].imshow(img_2025)
axes[1].set_title("2025 (post)", fontsize=14)
axes[1].axis("off")

axes[2].imshow(mask, cmap="gray", vmin=0, vmax=1)
axes[2].set_title("Land take", fontsize=14)
axes[2].axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()

print(f"Saved to {OUTPUT_PATH}")
print(f"Tile: {REFID}")
print(f"VHR shape: {img_2018.shape}, Mask shape: {mask.shape}")
print(f"Change pixels: {mask.sum()} / {mask.size} ({mask.mean()*100:.1f}%)")
