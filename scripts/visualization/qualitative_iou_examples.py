#!/usr/bin/env python3
"""
Qualitative IoU example figure for Discussion §7.

Renders three test tiles picked at the 10th, 50th, and 90th percentiles of
per-tile macro test IoU under the masked-loss U-Net trained on AlphaEarth at
50 balanced sparse points per tile (E4_ae_unet_sparse).

Each row: pre-change Sentinel-2 RGB | post-change S2 RGB | GT overlay | prediction overlay.

The pre/post Sentinel-2 composites are picked at the timesteps matching each
tile's annotation start and end years (per annotations_metadata_final.csv).

Output: REPORT/Figures/7_Discussion/qualitative_iou_examples.{pdf,png}
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import ListedColormap

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
})

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
REPO = Path("/cluster/home/tmstorma/NINA_fordypningsoppgave")
EXPERIMENT = "E4_ae_unet_sparse"
PRED_DIR = REPO / "PART2_spectral_spatial_resolution_experiments" / "outputs" / "experiments" / EXPERIMENT / "predictions"
TEST_RESULTS_JSON = REPO / "PART2_spectral_spatial_resolution_experiments" / "outputs" / "experiments" / EXPERIMENT / "test_results.json"

S2_DIR = REPO / "data_v2" / "Sentinel"
MASK_DIR = REPO / "data_v2" / "Land_take_masks_coarse"
ANNOT_META_CSV = REPO / "data_v2" / "annotations_metadata_final.csv"
SPLIT_CSV = REPO / "preprocessing" / "outputs" / "splits" / "unified" / "split_info.csv"

OUTPUT_DIR = REPO / "REPORT" / "Figures" / "7_Discussion"
OUTPUT_BASENAME = "qualitative_iou_examples"

PERCENTILES = [10, 50, 90]
CROP_SIZE = 64

S2_N_TIMESTEPS = 14
S2_N_BANDS = 9
N_YEARS = 7  # 2018-2024


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compose_annual(data: np.ndarray) -> np.ndarray:
    """Reduce 14 quarterly timesteps to 7 annual composites (Q2+Q3 average).

    Mirrors dataset.py:_compose_annual. NaN-aware fallback included.
    """
    composites = []
    for year_idx in range(N_YEARS):
        q2 = data[year_idx * 2]
        q3 = data[year_idx * 2 + 1]
        q2_nan = np.isnan(q2).sum() / q2.size * 100
        q3_nan = np.isnan(q3).sum() / q3.size * 100
        if q2_nan > 50 and q3_nan < 20:
            composites.append(q3)
        elif q3_nan > 50 and q2_nan < 20:
            composites.append(q2)
        else:
            composites.append((q2 + q3) / 2.0)
    return np.stack(composites, axis=0)


def center_crop(arr: np.ndarray, size: int = CROP_SIZE) -> np.ndarray:
    """Center crop the last two axes of arr to (size, size). Pad if smaller."""
    h, w = arr.shape[-2], arr.shape[-1]
    pad_h = max(0, size - h)
    pad_w = max(0, size - w)
    if pad_h > 0 or pad_w > 0:
        pad_widths = [(0, 0)] * (arr.ndim - 2) + [(0, pad_h), (0, pad_w)]
        arr = np.pad(arr, pad_widths, mode="constant", constant_values=0)
        h, w = arr.shape[-2], arr.shape[-1]
    top = (h - size) // 2
    left = (w - size) // 2
    return arr[..., top:top + size, left:left + size]


def stretch_p2_p98(rgb: np.ndarray) -> np.ndarray:
    """Per-channel p2-p98 stretch to [0, 1]. rgb shape: (3, H, W)."""
    out = np.zeros_like(rgb, dtype=np.float32)
    for c in range(3):
        ch = rgb[c]
        ch_valid = ch[~np.isnan(ch)] if np.any(np.isnan(ch)) else ch
        if ch_valid.size == 0:
            out[c] = 0
            continue
        p2, p98 = np.percentile(ch_valid, [2, 98])
        if p98 > p2:
            out[c] = np.clip((ch - p2) / (p98 - p2), 0, 1)
        else:
            out[c] = 0
    return out


def load_s2_pre_post_rgb(refid: str, start_year: int, end_year: int):
    """Load pre- and post-change Sentinel-2 RGB composites for one tile.

    Returns (pre_rgb (H,W,3), post_rgb (H,W,3)), each in [0, 1] after per-channel
    p2-p98 stretch, center-cropped to (CROP_SIZE, CROP_SIZE).
    """
    s2_path = S2_DIR / f"{refid}_RGBNIRRSWIRQ_Mosaic.tif"
    with rasterio.open(s2_path) as src:
        raw = src.read().astype(np.float32)  # (126, H, W)

    # Reshape to (14, 9, H, W) then composite to (7, 9, H, W)
    all_ts = raw.reshape(S2_N_TIMESTEPS, S2_N_BANDS, raw.shape[1], raw.shape[2])
    annual = compose_annual(all_ts)  # (7, 9, H, W)

    # Indices into the 7-year stack
    start_idx = max(start_year, 2018) - 2018
    end_idx = min(end_year, 2024) - 2018
    start_idx = max(0, min(start_idx, N_YEARS - 1))
    end_idx = max(0, min(end_idx, N_YEARS - 1))
    if end_idx == start_idx:
        end_idx = min(N_YEARS - 1, start_idx + 1)

    # RGB = bands 0, 1, 2 of the 9-band S2 stack
    pre = annual[start_idx, [0, 1, 2], :, :]   # (3, H, W)
    post = annual[end_idx, [0, 1, 2], :, :]    # (3, H, W)

    pre = center_crop(pre)
    post = center_crop(post)

    pre = stretch_p2_p98(pre).transpose(1, 2, 0)    # (H, W, 3)
    post = stretch_p2_p98(post).transpose(1, 2, 0)  # (H, W, 3)
    return pre, post, start_idx + 2018, end_idx + 2018


def load_prob_and_gt(refid: str):
    """Load prediction probability and GT mask, both (CROP_SIZE, CROP_SIZE)."""
    npz = np.load(PRED_DIR / f"{refid}.npz")
    return npz["prob"].astype(np.float32), npz["mask"].astype(np.uint8)


def compute_iou(pred_bin: np.ndarray, gt: np.ndarray) -> float:
    p = pred_bin.astype(bool)
    g = gt.astype(bool)
    union = (p | g).sum()
    return float((p & g).sum() / union) if union > 0 else 0.0


def compute_pr(pred_bin: np.ndarray, gt: np.ndarray):
    """Return (precision, recall) at threshold 0.5."""
    p = pred_bin.astype(bool)
    g = gt.astype(bool)
    tp = float((p & g).sum())
    fp = float((p & ~g).sum())
    fn = float((~p & g).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load test refids
    split_df = pd.read_csv(SPLIT_CSV)
    test_refids = split_df[split_df["split"] == "test"]["refid"].tolist()
    assert len(test_refids) == 40, f"Expected 40 test tiles, got {len(test_refids)}"

    # 2. Load annotation metadata
    annot_df = pd.read_csv(ANNOT_META_CSV).set_index("REFID")

    # 3. Compute per-tile IoU from NPZs and cross-check against test_results.json
    with open(TEST_RESULTS_JSON) as f:
        test_results = json.load(f)
    reported = test_results["ensemble"]["per_sample"]

    per_tile = []
    for refid in test_refids:
        prob, gt = load_prob_and_gt(refid)
        pred_bin = (prob > 0.5).astype(np.uint8)
        iou = compute_iou(pred_bin, gt)
        per_tile.append((refid, iou))
        ref_iou = reported.get(refid, {}).get("iou")
        if ref_iou is not None and abs(iou - ref_iou) > 0.01:
            print(f"  warn: IoU mismatch refid={refid} computed={iou:.4f} reported={ref_iou:.4f}")

    iou_values = np.array([x[1] for x in per_tile])
    print(f"Test IoU stats over 40 tiles: mean={iou_values.mean()*100:.2f}%, "
          f"p10={np.percentile(iou_values, 10)*100:.2f}%, "
          f"p50={np.percentile(iou_values, 50)*100:.2f}%, "
          f"p90={np.percentile(iou_values, 90)*100:.2f}%")

    # 4. Pick the three refids at chosen percentiles (nearest-rank)
    sorted_tiles = sorted(per_tile, key=lambda x: x[1])
    picks = []
    for p in PERCENTILES:
        idx = int(np.ceil(p / 100 * len(sorted_tiles))) - 1
        idx = max(0, min(idx, len(sorted_tiles) - 1))
        picks.append((p, sorted_tiles[idx]))
    print("\nPicked tiles:")
    for p, (refid, iou) in picks:
        print(f"  P{p:2d}: refid={refid}, IoU={iou*100:.2f}%")

    # 5. Build figure
    fig, axes = plt.subplots(3, 4, figsize=(11, 8.5))
    col_titles = ["Pre (Sentinel-2 RGB)", "Post (Sentinel-2 RGB)",
                  "Ground truth", "Prediction vs truth"]
    for j, t in enumerate(col_titles):
        axes[0, j].set_title(t, fontsize=11, pad=6)

    # Colors for the agreement map. Okabe-Ito colourblind-safe palette, all
    # distinct from the ground-truth red overlay in the previous column. The
    # true-negative (no-change) class is the neutral light-grey background.
    tp_color = (0.0, 0.620, 0.451)    # bluish green: true positive (correct detection)
    fp_color = (0.800, 0.475, 0.655)  # reddish purple: false positive (overprediction)
    fn_color = (0.902, 0.624, 0.0)    # orange: false negative (missed change)
    tn_color = (0.87, 0.87, 0.87)     # light grey: true negative (correct no-change)

    for i, (pct, (refid, iou)) in enumerate(picks):
        # Annotation years
        row = annot_df.loc[refid]
        start_year = int(row["startYear"])
        end_year = int(row["endYear"])

        # Imagery, mask, prediction
        pre, post, pre_year, post_year = load_s2_pre_post_rgb(refid, start_year, end_year)
        prob, gt = load_prob_and_gt(refid)
        pred_bin = (prob > 0.5).astype(np.uint8)
        precision, recall = compute_pr(pred_bin, gt)

        # GT overlay: pure bright red, high alpha
        red_cmap = ListedColormap([(1.0, 0.0, 0.0)])
        gt_over = np.ma.masked_where(gt == 0, gt)

        # Agreement categories on a true-negative (no-change) background.
        # Mutually exclusive, so each pixel is exactly one flat colour.
        p = pred_bin.astype(bool)
        g = gt.astype(bool)
        agree_rgb = np.empty((p.shape[0], p.shape[1], 3), dtype=np.float32)
        agree_rgb[...] = tn_color
        agree_rgb[p & g] = tp_color
        agree_rgb[p & ~g] = fp_color
        agree_rgb[~p & g] = fn_color

        # Plot
        axes[i, 0].imshow(pre, interpolation="nearest")
        axes[i, 1].imshow(post, interpolation="nearest")
        # Darken the post background slightly so overlays stand out
        post_dim = post * 0.7
        axes[i, 2].imshow(post_dim, interpolation="nearest")
        axes[i, 2].imshow(gt_over, cmap=red_cmap, alpha=0.85,
                          interpolation="nearest")
        axes[i, 3].imshow(agree_rgb, interpolation="nearest")

        # Year tags below the pre/post panels
        axes[i, 0].set_xlabel(str(pre_year), fontsize=9, labelpad=2)
        axes[i, 1].set_xlabel(str(post_year), fontsize=9, labelpad=2)

        # Row label: percentile + IoU, precision, recall
        row_label = (
            f"$p_{{{pct}}}$\n"
            f"IoU = {iou * 100:.1f}%\n"
            f"P = {precision * 100:.1f}%\n"
            f"R = {recall * 100:.1f}%"
        )
        axes[i, 0].set_ylabel(row_label, fontsize=10, rotation=0, labelpad=42,
                              va="center", ha="right")

        for j in range(4):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            for spine in axes[i, j].spines.values():
                spine.set_linewidth(0.5)
                spine.set_color("0.5")

    # Legend for the error column
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=tp_color, label="True positive"),
        Patch(facecolor=fp_color, label="False positive (overprediction)"),
        Patch(facecolor=fn_color, label="False negative (missed land take)"),
        Patch(facecolor=tn_color, edgecolor="0.6", linewidth=0.5,
              label="True negative (no land take)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=4, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(pad=0.4)
    pdf_path = OUTPUT_DIR / f"{OUTPUT_BASENAME}.pdf"
    png_path = OUTPUT_DIR / f"{OUTPUT_BASENAME}.png"
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    print(f"\nSaved {pdf_path}")
    print(f"Saved {png_path}")


if __name__ == "__main__":
    main()
