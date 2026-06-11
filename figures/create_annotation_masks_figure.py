#!/usr/bin/env python3
"""
Generate the annotation_masks.pdf figure for the data chapter.

Three panels: (a) VHR start year, (b) VHR end year, (c) Coarse mask.
Picks a tile with clear, visible change for the illustration.
"""

import numpy as np
import pandas as pd
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
})

from landtake.paths import REPO_ROOT as REPO
VHR_DIR = REPO / "data_v2" / "VHR_google"
MASK_DIR = REPO / "data_v2" / "Land_take_masks_coarse"
SPLITS_CSV = REPO / "preprocessing" / "outputs" / "splits" / "unified" / "split_info.csv"
ANNO_CSV = REPO / "data_v2" / "annotations_metadata_final.csv"
OUT_DIR = REPO / "REPORT" / "Figures" / "3_Data_Study_Area"

# Select a tile with clear, moderate change
# The previous figure showed a forest clearing — pick something similar
PREFERRED_REFID = "a8-80249467517939_47-54398344853953"  # fallback to a moderate-change tile


def load_vhr(refid, year_idx, crop_pct=0.03):
    """Load VHR RGB for start (idx=0) or end (idx=1) year, with small inward crop
    to remove the red annotation boundary line."""
    path = VHR_DIR / f"{refid}_RGBY_Mosaic.tif"
    with rasterio.open(path) as src:
        # VHR has 6 bands: 3 RGB × 2 dates
        # First 3 bands = start year, next 3 = end year
        if year_idx == 0:
            rgb = src.read([1, 2, 3])
        else:
            rgb = src.read([4, 5, 6])
    rgb = rgb.astype(np.float32)
    rgb = rgb.transpose(1, 2, 0)
    # Crop a small margin to remove the red annotation boundary
    H, W, _ = rgb.shape
    dh = int(H * crop_pct)
    dw = int(W * crop_pct)
    rgb = rgb[dh:H - dh, dw:W - dw, :]
    # Percentile stretch
    for c in range(3):
        p2, p98 = np.percentile(rgb[:, :, c][rgb[:, :, c] > 0], [2, 98]) if (rgb[:, :, c] > 0).any() else (0, 255)
        rgb[:, :, c] = np.clip((rgb[:, :, c] - p2) / (p98 - p2 + 1e-8), 0, 1)
    return rgb


def load_mask(refid, crop_pct=0.03):
    path = MASK_DIR / f"{refid}_mask.tif"
    with rasterio.open(path) as src:
        mask = (src.read(1) > 0).astype(np.float32)
    H, W = mask.shape
    dh = int(H * crop_pct)
    dw = int(W * crop_pct)
    return mask[dh:H - dh, dw:W - dw]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    splits = pd.read_csv(SPLITS_CSV)
    anno = pd.read_csv(ANNO_CSV)

    # Pick a tile: moderate change ratio, train split, with a clear change pattern
    # Filter to tiles with perfect annotation alignment (start=2018, end=2024)
    # to avoid the worst reprojection artifacts in VHR
    anno_idx = anno.set_index("REFID")
    splits["startYear"] = splits["refid"].map(lambda r: anno_idx.loc[r, "startYear"] if r in anno_idx.index else None)
    splits["endYear"] = splits["refid"].map(lambda r: anno_idx.loc[r, "endYear"] if r in anno_idx.index else None)

    # Extract latitude from refid (format: a{lon}_{lat} where lon/lat use - as decimal point)
    def parse_lat(refid):
        try:
            lat_part = refid.split("_")[-1]  # e.g. "45-7859705066069"
            # First "-" is the decimal point, but a leading "-" means negative
            if lat_part.startswith("-"):
                rest = lat_part[1:]
                deg, dec = rest.split("-", 1)
                return -float(f"{deg}.{dec.replace('-', '')}")
            else:
                deg, dec = lat_part.split("-", 1)
                return float(f"{deg}.{dec.replace('-', '')}")
        except Exception:
            return None

    splits["lat"] = splits["refid"].map(parse_lat)

    # Lower latitudes = less EPSG:3035 rotation distortion
    candidates = splits[
        (splits["split"] == "train")
        & (splits["change_level"] == "moderate")
        & (splits["change_ratio"].between(8, 20))
        & (splits["startYear"] == 2018)
        & (splits["endYear"] == 2024)
        & (splits["lat"] < 47)  # southern / central Europe
    ].sort_values("lat").copy()

    # Skip tiles already shown
    SKIP_REFIDS = {
        "a17-81046022198732_48-02360903747409",
        "a-0-7288032620039_48-63928444006693",
        "a16-15317118163363_45-7859705066069",
        "a9-27236257749211_45-08294767924716",
    }

    refid = None
    for _, row in candidates.iterrows():
        r = row["refid"]
        if r in SKIP_REFIDS:
            continue
        vhr_path = VHR_DIR / f"{r}_RGBY_Mosaic.tif"
        if not (vhr_path.exists() and (MASK_DIR / f"{r}_mask.tif").exists()):
            continue
        # Check for excessive zero-padding in VHR (skip if >2% all-zero pixels)
        with rasterio.open(vhr_path) as src:
            sample = src.read([1, 2, 3])
        zero_frac = (sample == 0).all(axis=0).sum() / sample[0].size
        if zero_frac > 0.02:
            continue
        refid = r
        break

    if refid is None:
        raise RuntimeError("No suitable tile found")

    change_pct = splits[splits["refid"] == refid]["change_ratio"].iloc[0]
    anno_row = anno[anno["REFID"] == refid].iloc[0] if (anno["REFID"] == refid).any() else None
    start_year = int(anno_row["startYear"]) if anno_row is not None else "?"
    end_year = int(anno_row["endYear"]) if anno_row is not None else "?"

    print(f"Selected tile: {refid}")
    print(f"  Change ratio: {change_pct:.1f}%")
    print(f"  Annotation years: {start_year} → {end_year}")

    rgb_start = load_vhr(refid, year_idx=0)
    rgb_end = load_vhr(refid, year_idx=1)
    mask = load_mask(refid)

    H, W = mask.shape
    print(f"  VHR shape: {rgb_start.shape}, Mask shape: {mask.shape}")

    # Coarse mask: red on light grey background
    mask_vis = np.ones((H, W, 3)) * 0.95
    mask_vis[mask > 0] = [0.85, 0.2, 0.2]

    panels = [
        ("a", rgb_start),
        ("b", rgb_end),
        ("c", mask_vis),
    ]

    for letter, img in panels:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("0.5")
            spine.set_linewidth(0.5)
        plt.tight_layout(pad=0.1)
        for ext in ("pdf", "png"):
            out = OUT_DIR / f"annotation_masks_{letter}.{ext}"
            fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.02)
            print(f"Saved: {out}")
        plt.close(fig)


if __name__ == "__main__":
    main()
