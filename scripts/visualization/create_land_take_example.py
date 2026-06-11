#!/usr/bin/env python3
"""
Generate the land_take_example figure for the background chapter.

Produces three standalone files (one per panel) so the LaTeX figure can place
them as subfigures with subfigure-level (a)/(b)/(c) captions:

    land_take_example_a.{pdf,png}  VHR imagery at the start year
    land_take_example_b.{pdf,png}  VHR imagery at the end year
    land_take_example_c.{pdf,png}  Binary land-take mask with legend
"""

import numpy as np
import pandas as pd
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
VHR_DIR = REPO / "data_v2" / "VHR_google"
MASK_DIR = REPO / "data_v2" / "Land_take_masks_coarse"
SPLITS_CSV = REPO / "preprocessing" / "outputs" / "splits" / "unified" / "split_info.csv"
ANNO_CSV = REPO / "data_v2" / "annotations_metadata_final.csv"
OUT_DIR = REPO / "REPORT" / "figures"

NO_LANDTAKE_COLOR = (0.18, 0.45, 0.25)
LANDTAKE_COLOR = (0.85, 0.2, 0.2)


def load_vhr(refid, year_idx, crop_pct=0.03):
    path = VHR_DIR / f"{refid}_RGBY_Mosaic.tif"
    with rasterio.open(path) as src:
        if year_idx == 0:
            rgb = src.read([1, 2, 3])
        else:
            rgb = src.read([4, 5, 6])
    rgb = rgb.astype(np.float32).transpose(1, 2, 0)
    H, W, _ = rgb.shape
    dh = int(H * crop_pct)
    dw = int(W * crop_pct)
    rgb = rgb[dh:H - dh, dw:W - dw, :]
    for c in range(3):
        positive = rgb[:, :, c][rgb[:, :, c] > 0]
        if positive.size:
            p2, p98 = np.percentile(positive, [2, 98])
        else:
            p2, p98 = 0, 255
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


def parse_lat(refid):
    try:
        lat_part = refid.split("_")[-1]
        if lat_part.startswith("-"):
            rest = lat_part[1:]
            deg, dec = rest.split("-", 1)
            return -float(f"{deg}.{dec.replace('-', '')}")
        deg, dec = lat_part.split("-", 1)
        return float(f"{deg}.{dec.replace('-', '')}")
    except Exception:
        return None


def select_tile():
    splits = pd.read_csv(SPLITS_CSV)
    splits["lat"] = splits["refid"].map(parse_lat)

    # Background-chapter example: substantial, visually obvious land take.
    candidates = splits[
        (splits["split"] == "train")
        & (splits["change_ratio"].between(20, 60))
        & (splits["startYear"] == 2018)
        & (splits["endYear"] == 2024)
    ].sort_values("change_ratio", ascending=False).copy()

    # Avoid tiles already used in other figures
    SKIP_REFIDS = {
        "a17-81046022198732_48-02360903747409",
        "a-0-7288032620039_48-63928444006693",
        "a16-15317118163363_45-7859705066069",
        "a9-27236257749211_45-08294767924716",
        "a8-80249467517939_47-54398344853953",
    }

    for _, row in candidates.iterrows():
        r = row["refid"]
        if r in SKIP_REFIDS:
            continue
        vhr_path = VHR_DIR / f"{r}_RGBY_Mosaic.tif"
        mask_path = MASK_DIR / f"{r}_mask.tif"
        if not (vhr_path.exists() and mask_path.exists()):
            continue
        with rasterio.open(vhr_path) as src:
            sample = src.read([1, 2, 3])
        zero_frac = (sample == 0).all(axis=0).sum() / sample[0].size
        if zero_frac > 0.02:
            continue
        return r, row

    raise RuntimeError("No suitable tile found")


def save_panel(image_array, out_stem, legend_handles=None):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(image_array)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if legend_handles is not None:
        ax.legend(
            handles=legend_handles,
            loc="lower right",
            frameon=True,
            framealpha=0.9,
            fontsize=9,
        )
    plt.tight_layout(pad=0.2)
    fig.savefig(OUT_DIR / f"{out_stem}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{out_stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    refid, row = select_tile()
    print(f"Selected tile: {refid}")
    print(f"  Change ratio: {row['change_ratio']:.1f}%")
    print(f"  Annotation years: {int(row['startYear'])} -> {int(row['endYear'])}")

    rgb_start = load_vhr(refid, year_idx=0)
    rgb_end = load_vhr(refid, year_idx=1)
    mask = load_mask(refid)
    H, W = mask.shape

    mask_vis = np.empty((H, W, 3), dtype=np.float32)
    mask_vis[mask <= 0] = NO_LANDTAKE_COLOR
    mask_vis[mask > 0] = LANDTAKE_COLOR

    legend_handles = [
        Patch(facecolor=NO_LANDTAKE_COLOR, edgecolor="none", label="No land take"),
        Patch(facecolor=LANDTAKE_COLOR, edgecolor="none", label="Land take"),
    ]

    save_panel(rgb_start, "land_take_example_a")
    save_panel(rgb_end, "land_take_example_b")
    save_panel(mask_vis, "land_take_example_c", legend_handles=legend_handles)

    for stem in ("land_take_example_a", "land_take_example_b", "land_take_example_c"):
        print(f"Saved: {OUT_DIR / f'{stem}.pdf'}")
        print(f"Saved: {OUT_DIR / f'{stem}.png'}")


if __name__ == "__main__":
    main()
