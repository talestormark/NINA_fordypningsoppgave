#!/usr/bin/env python3
"""
Qualitative point-budget saturation figure for Discussion (RQ3c).

For one representative (median-IoU) test tile, shows the masked-loss U-Net's
prediction across point budgets (10, 20, 30, 50, 200 balanced points) and the
dense reference, as agreement maps. The predicted maps stop changing beyond
~50 points: the spatial counterpart of the budget curve (fig:rq3_budget_curve).

Helpers (compose_annual, center_crop, stretch_p2_p98, load_s2_pre_post_rgb,
compute_iou) are copied from qualitative_iou_examples.py to keep this script
self-contained, matching the convention of scripts/visualization/.

Prediction sources (per-tile ensemble probability maps):
  10/20/30/200 pts -> experiments/annotation_efficiency/outputs/
                      E4_D2_alphaearth_sparse_n{5,10,15,100}/ensemble_predictions/{refid}.npz
  50 pts           -> PART2/.../E4_ae_unet_sparse/predictions/{refid}.npz
  dense            -> PART2/.../D2_alphaearth/predictions/{refid}.npz
All NPZs carry arrays "prob" and "mask".

Output: REPORT/Figures/7_Discussion/qualitative_budget_saturation.{pdf,png}
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
})

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
REPO = Path("/cluster/home/tmstorma/NINA_fordypningsoppgave")
PART2_EXP = REPO / "PART2_spectral_spatial_resolution_experiments" / "outputs" / "experiments"
AE_OUT = REPO / "experiments" / "annotation_efficiency" / "outputs"

# Tile selection: median of the 50-point model's per-tile IoU distribution.
SELECT_DIR = PART2_EXP / "E4_ae_unet_sparse" / "predictions"

# Budget -> directory holding per-tile ensemble prediction NPZs (prob, mask).
BUDGETS = [
    ("10 points", AE_OUT / "E4_D2_alphaearth_sparse_n5" / "ensemble_predictions"),
    ("20 points", AE_OUT / "E4_D2_alphaearth_sparse_n10" / "ensemble_predictions"),
    ("30 points", AE_OUT / "E4_D2_alphaearth_sparse_n15" / "ensemble_predictions"),
    ("50 points", PART2_EXP / "E4_ae_unet_sparse" / "predictions"),
    ("200 points", AE_OUT / "E4_D2_alphaearth_sparse_n100" / "ensemble_predictions"),
    ("Dense", PART2_EXP / "D2_alphaearth" / "predictions"),
]

# Tile shown in the committed figure: a representative tile with a clean
# point-budget saturation (52 -> 56 -> 59% as the budget grows, then a plateau).
# Override with --refid. The median-IoU tile used by
# fig:disc_qualitative_iou_examples is a3-28755789400137_50-64539824344691.
SELECTED_REFID = "a27-9364000673535_41-24133449388484"

S2_DIR = REPO / "data_v2" / "Sentinel"
ANNOT_META_CSV = REPO / "data_v2" / "annotations_metadata_final.csv"
SPLIT_CSV = REPO / "preprocessing" / "outputs" / "splits" / "unified" / "split_info.csv"

OUTPUT_DIR = REPO / "REPORT" / "Figures" / "7_Discussion"
OUTPUT_BASENAME = "qualitative_budget_saturation"

CROP_SIZE = 64
S2_N_TIMESTEPS = 14
S2_N_BANDS = 9
N_YEARS = 7  # 2018-2024

# Okabe-Ito agreement-map palette (matches qualitative_iou_examples.py).
tp_color = (0.0, 0.620, 0.451)    # bluish green: true positive
fp_color = (0.800, 0.475, 0.655)  # reddish purple: false positive (overprediction)
fn_color = (0.902, 0.624, 0.0)    # orange: false negative (missed change)
tn_color = (0.87, 0.87, 0.87)     # light grey: true negative (no land take)


# ---------------------------------------------------------------------------
# Helpers (copied from qualitative_iou_examples.py)
# ---------------------------------------------------------------------------

def compose_annual(data: np.ndarray) -> np.ndarray:
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
    s2_path = S2_DIR / f"{refid}_RGBNIRRSWIRQ_Mosaic.tif"
    with rasterio.open(s2_path) as src:
        raw = src.read().astype(np.float32)
    all_ts = raw.reshape(S2_N_TIMESTEPS, S2_N_BANDS, raw.shape[1], raw.shape[2])
    annual = compose_annual(all_ts)
    start_idx = max(start_year, 2018) - 2018
    end_idx = min(end_year, 2024) - 2018
    start_idx = max(0, min(start_idx, N_YEARS - 1))
    end_idx = max(0, min(end_idx, N_YEARS - 1))
    if end_idx == start_idx:
        end_idx = min(N_YEARS - 1, start_idx + 1)
    pre = annual[start_idx, [0, 1, 2], :, :]
    post = annual[end_idx, [0, 1, 2], :, :]
    pre = stretch_p2_p98(center_crop(pre)).transpose(1, 2, 0)
    post = stretch_p2_p98(center_crop(post)).transpose(1, 2, 0)
    return pre, post, start_idx + 2018, end_idx + 2018


def compute_iou(pred_bin: np.ndarray, gt: np.ndarray) -> float:
    p = pred_bin.astype(bool)
    g = gt.astype(bool)
    union = (p | g).sum()
    return float((p & g).sum() / union) if union > 0 else 0.0


def load_prob_and_gt(pred_dir: Path, refid: str):
    npz = np.load(pred_dir / f"{refid}.npz")
    prob = npz["prob"].astype(np.float32)
    mask = npz["mask"].astype(np.uint8)
    if prob.ndim == 3:      # (1, H, W) -> (H, W)
        prob = prob.squeeze(0)
    if mask.ndim == 3:
        mask = mask.squeeze(0)
    return prob, mask


def agreement_rgb(pred_bin: np.ndarray, gt: np.ndarray) -> np.ndarray:
    p = pred_bin.astype(bool)
    g = gt.astype(bool)
    rgb = np.empty((p.shape[0], p.shape[1], 3), dtype=np.float32)
    rgb[...] = tn_color
    rgb[p & g] = tp_color
    rgb[p & ~g] = fp_color
    rgb[~p & g] = fn_color
    return rgb


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def pick_median_tile():
    """Refid nearest the median of the 50-point model's per-tile IoU."""
    split_df = pd.read_csv(SPLIT_CSV)
    test_refids = split_df[split_df["split"] == "test"]["refid"].tolist()
    per_tile = []
    for refid in test_refids:
        f = SELECT_DIR / f"{refid}.npz"
        if not f.exists():
            continue
        prob, gt = load_prob_and_gt(SELECT_DIR, refid)
        per_tile.append((refid, compute_iou((prob > 0.5), gt)))
    per_tile.sort(key=lambda x: x[1])
    idx = int(np.ceil(0.50 * len(per_tile))) - 1
    idx = max(0, min(idx, len(per_tile) - 1))
    return per_tile[idx]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--refid", default=None, help="override tile (default: median 50-pt IoU tile)")
    ap.add_argument("--suffix", default="", help="append to output basename")
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.refid:
        refid = args.refid
        _prob, _gt = load_prob_and_gt(SELECT_DIR, refid)
        sel_iou = compute_iou((_prob > 0.5), _gt)
        print(f"Tile (override): {refid} (50-pt IoU = {sel_iou*100:.1f}%)")
    else:
        refid = SELECTED_REFID
        _prob, _gt = load_prob_and_gt(SELECT_DIR, refid)
        sel_iou = compute_iou((_prob > 0.5), _gt)
        print(f"Selected tile: {refid} (50-pt IoU = {sel_iou*100:.1f}%)")

    annot = pd.read_csv(ANNOT_META_CSV).set_index("REFID")
    start_year = int(annot.loc[refid, "startYear"])
    end_year = int(annot.loc[refid, "endYear"])
    pre, post, pre_year, post_year = load_s2_pre_post_rgb(refid, start_year, end_year)
    post_dim = post * 0.7

    # Load per-budget predictions for this tile.
    panels = []
    gt_ref = None
    for label, pred_dir in BUDGETS:
        prob, gt = load_prob_and_gt(pred_dir, refid)
        gt_ref = gt if gt_ref is None else gt_ref
        pred_bin = (prob > 0.5)
        panels.append((label, agreement_rgb(pred_bin, gt), compute_iou(pred_bin, gt)))

    # ---- Layout: [Ground truth] | wider gap | [budgets, tightly spaced] ----
    # A spacer column separates the GT reference from the budget series; the
    # budget panels themselves sit close together so they read as one series.
    n_b = len(panels)  # 6
    GT_GAP = 0.30      # width of the spacer column (GT-to-first-budget gap)
    BUDGET_WSPACE = 0.10  # gap between adjacent budget panels (small)
    fig = plt.figure(figsize=(13, 2.6))
    gs = GridSpec(1, n_b + 2, width_ratios=[1.0, GT_GAP] + [1.0] * n_b,
                  wspace=BUDGET_WSPACE)

    red_cmap = ListedColormap([(1.0, 0.0, 0.0)])
    gt_over = np.ma.masked_where(gt_ref == 0, gt_ref)
    ax_gt = fig.add_subplot(gs[0, 0])
    ax_gt.imshow(post_dim, interpolation="nearest")
    ax_gt.imshow(gt_over, cmap=red_cmap, alpha=0.85, interpolation="nearest")
    ax_gt.set_title("Ground truth", fontsize=10, pad=4)

    budget_axes = []
    for k, (label, rgb, iou) in enumerate(panels):
        ax = fig.add_subplot(gs[0, k + 2])  # +2 skips GT (col 0) and spacer (col 1)
        ax.imshow(rgb, interpolation="nearest")
        ax.set_title(label, fontsize=10, pad=4)
        budget_axes.append(ax)

    for ax in [ax_gt] + budget_axes:
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.5); sp.set_color("0.5")

    legend_handles = [
        Patch(facecolor=tp_color, label="True positive"),
        Patch(facecolor=fp_color, label="False positive (overprediction)"),
        Patch(facecolor=fn_color, label="False negative (missed land take)"),
        Patch(facecolor=tn_color, edgecolor="0.6", linewidth=0.5,
              label="True negative (no land take)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))

    base = OUTPUT_BASENAME + args.suffix
    pdf_path = OUTPUT_DIR / f"{base}.pdf"
    png_path = OUTPUT_DIR / f"{base}.png"
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")
    print("Per-budget IoU: " + ", ".join(f"{l}={i*100:.1f}%" for l, _, i in panels))


if __name__ == "__main__":
    main()
