#!/usr/bin/env python3
"""
Qualitative analysis of CV (out-of-fold) predictions.

Generates systematic visualizations for model sanity checks before test evaluation.

KEY DESIGN CHOICES:
1. Uses the SAME tiles across all temporal modes for apples-to-apples comparison
2. Tiles selected based on MEAN IoU across all conditions
3. ALSO includes "divergence" tiles where modes differ significantly
4. Shows the ACTUAL annual composite the model sees (not raw Q2)
5. Computes MODE-SPECIFIC valid data fractions

Each visualization includes:
- Annual composite RGB (first vs last year) - same as model input
- Mode-specific valid data mask
- Ground truth mask
- Probability map
- Thresholded prediction (t=0.5)
- TP/FP/FN overlay

Usage:
    python qualitative_cv_analysis.py --num-examples 3
"""

import sys
import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from tqdm import tqdm
import rasterio
import pandas as pd

# Add paths for imports
script_dir = Path(__file__).resolve().parent
parent_dir = script_dir.parent.parent  # PART1_multi_temporal_experiments/
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(parent_dir.parent))  # NINA_fordypningsoppgave/
sys.path.insert(0, str(parent_dir / "scripts" / "modeling"))  # For models_multitemporal

from PART1_multi_temporal_experiments.config import (
    MT_EXPERIMENTS_DIR, DATA_DIR, YEARS, QUARTERS, SENTINEL2_BANDS
)
from PART1_multi_temporal_experiments.scripts.data_preparation.dataset_multitemporal import (
    MultiTemporalSentinel2Dataset, compute_normalization_stats
)
from models_multitemporal import create_multitemporal_model

# Experiment configurations (unified 400-epoch protocol)
EXPERIMENTS = {
    'annual': {
        'name': 'exp010_lstm7_no_es',
        'temporal_sampling': 'annual',
        'T': 7,
    },
    'bi_temporal': {
        'name': 'exp003_v3',
        'temporal_sampling': 'bi_temporal',
        'T': 2,
    },
    'bi_seasonal': {
        'name': 'exp002_v3',
        'temporal_sampling': 'quarterly',
        'T': 14,
    },
}

# Colorblind-safe colors for TP/FP/FN overlay
COLORS = {
    'TP': np.array([0.12, 0.47, 0.71]),    # Blue
    'FP': np.array([1.0, 0.5, 0.05]),      # Orange
    'FN': np.array([0.58, 0.40, 0.74]),    # Purple
}


# =============================================================================
# CACHING FOR EFFICIENCY
# =============================================================================

class ModelCache:
    """Cache for loaded models to avoid repeated I/O."""

    def __init__(self):
        self._models = {}

    def get_model(self, condition: str, fold: int, device: torch.device):
        """Get model from cache or load it."""
        key = (condition, fold)
        if key not in self._models:
            exp_config = EXPERIMENTS[condition]
            exp_name = exp_config['name']
            exp_dir = MT_EXPERIMENTS_DIR / f"{exp_name}_fold{fold}"
            config_path = exp_dir / "config.json"
            checkpoint_path = exp_dir / "best_model.pth"

            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

            with open(config_path) as f:
                config = json.load(f)

            model = create_multitemporal_model(
                config['model_name'],
                encoder_name=config['encoder_name'],
                encoder_weights=None,
                in_channels=9,
                classes=1,
                lstm_hidden_dim=config['lstm_hidden_dim'],
                lstm_num_layers=config['lstm_num_layers'],
                skip_aggregation=config['skip_aggregation'],
            )

            checkpoint = torch.load(checkpoint_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)

            model = model.to(device)
            model.eval()

            self._models[key] = (model, config)

        return self._models[key]


class NormStatsCache:
    """Cache for normalization stats to avoid repeated computation."""

    def __init__(self):
        self._stats = {}

    def get_stats(self, fold: int, train_refids: list):
        """Get norm stats from cache or compute them."""
        if fold not in self._stats:
            print(f"  Computing normalization stats for fold {fold}...")
            self._stats[fold] = compute_normalization_stats(train_refids)
        return self._stats[fold]


# Global caches
MODEL_CACHE = ModelCache()
NORM_STATS_CACHE = NormStatsCache()


# =============================================================================
# FOLD ASSIGNMENT (using same source as training)
# =============================================================================

def get_fold_assignments(refids: list, num_folds: int = 5, seed: int = 42):
    """
    Get fold assignments using the EXACT same logic and ordering as training.

    Loads train/val split files and concatenates them in the same order as
    get_dataloaders() in dataset_multitemporal.py, ensuring StratifiedKFold
    produces identical fold assignments.

    Returns:
        fold_assignments: dict {refid: fold_index}
        fold_train_refids: dict {fold_index: [train_refids]}
    """
    from sklearn.model_selection import StratifiedKFold

    base_dir = Path(__file__).resolve().parent.parent.parent.parent

    # Load split files in the SAME order as get_dataloaders()
    splits_dir = base_dir / "outputs" / "splits"
    train_refids_orig = [line.strip() for line in open(splits_dir / "train_refids.txt")]
    val_refids_orig = [line.strip() for line in open(splits_dir / "val_refids.txt")]
    trainval_refids = train_refids_orig + val_refids_orig

    # Load change levels (same file as dataset_multitemporal.py uses)
    change_level_path = base_dir / "PART1_multi_temporal_experiments" / "sample_change_levels.csv"

    if not change_level_path.exists():
        raise FileNotFoundError(
            f"Change level file not found: {change_level_path}\n"
            "This file is required for stratified fold assignment."
        )

    change_level_df = pd.read_csv(change_level_path)
    refid_to_level = dict(zip(change_level_df['refid'], change_level_df['change_level']))

    # Get change levels in trainval order (matching get_dataloaders)
    change_levels = [refid_to_level[refid] for refid in trainval_refids]

    # Create stratified k-fold splits (same as training)
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)

    fold_assignments = {}
    fold_train_refids = {}

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(trainval_refids, change_levels)):
        for i in val_idx:
            fold_assignments[trainval_refids[i]] = fold_idx
        fold_train_refids[fold_idx] = [trainval_refids[i] for i in train_idx]

    return fold_assignments, fold_train_refids


# =============================================================================
# DATA LOADING WITH PROPER HANDLING
# =============================================================================

def load_raw_sentinel2(refid: str, sentinel2_dir: Path = None) -> np.ndarray:
    """
    Load raw Sentinel-2 data (no normalization).

    Returns:
        Array of shape (14, 9, H, W) - all 14 quarterly time steps
    """
    if sentinel2_dir is None:
        sentinel2_dir = DATA_DIR / "Sentinel"

    s2_path = sentinel2_dir / f"{refid}_RGBNIRRSWIRQ_Mosaic.tif"

    with rasterio.open(s2_path) as src:
        data = src.read()  # (126, H, W)

    num_full_time_steps = len(YEARS) * len(QUARTERS)  # 14
    num_bands = len(SENTINEL2_BANDS)  # 9

    # Reshape to (14, 9, H, W)
    data = data.reshape(num_full_time_steps, num_bands, data.shape[1], data.shape[2])

    return data


def load_mask(refid: str, target_shape: tuple = None) -> np.ndarray:
    """
    Load and binarize mask, optionally resample to target shape.

    Args:
        refid: Reference ID
        target_shape: (H, W) to resample to if needed

    Returns:
        Binary mask (H, W) as float32 with values 0.0 or 1.0
    """
    from scipy.ndimage import zoom

    mask_path = DATA_DIR / "Land_take_masks" / f"{refid}_mask.tif"

    with rasterio.open(mask_path) as src:
        mask = src.read(1)

    # CRITICAL: Binarize mask (handles 0/255 or any other encoding)
    mask = (mask > 0).astype(np.float32)

    # Resample if needed
    if target_shape is not None and mask.shape != target_shape:
        zoom_factors = (target_shape[0] / mask.shape[0], target_shape[1] / mask.shape[1])
        mask = zoom(mask, zoom_factors, order=0)  # Nearest neighbor

    return mask


def compute_annual_composite(raw_data: np.ndarray, year_idx: int) -> np.ndarray:
    """
    Compute annual composite exactly as the model sees it.

    Uses Q2+Q3 mean with fallback logic (same as dataset_multitemporal.py lines 246-274).

    Args:
        raw_data: Shape (14, 9, H, W)
        year_idx: Year index (0-6 for 2018-2024)

    Returns:
        Composite array (9, H, W)
    """
    q2_idx = year_idx * 2
    q3_idx = year_idx * 2 + 1

    q2_data = raw_data[q2_idx]  # (9, H, W)
    q3_data = raw_data[q3_idx]  # (9, H, W)

    # Same fallback logic as dataset
    q2_nan_pct = np.isnan(q2_data).sum() / q2_data.size * 100
    q3_nan_pct = np.isnan(q3_data).sum() / q3_data.size * 100

    if q2_nan_pct > 50 and q3_nan_pct < 20:
        # Q2 is bad, use Q3
        return q3_data
    elif q3_nan_pct > 50 and q2_nan_pct < 20:
        # Q3 is bad, use Q2
        return q2_data
    else:
        # Average them
        return (q2_data + q3_data) / 2.0


def get_rgb_from_composite(composite: np.ndarray) -> np.ndarray:
    """
    Get RGB image from a composite with percentile stretch.

    Args:
        composite: Shape (9, H, W) or (C, H, W)

    Returns:
        RGB array (H, W, 3) normalized to [0, 1]
    """
    # Band indices: R=2, G=1, B=0 (based on RGBNIRRSWIRQ naming)
    rgb = np.stack([composite[2], composite[1], composite[0]], axis=-1)  # (H, W, 3)

    # Handle NaN
    rgb = np.nan_to_num(rgb, nan=0)

    # Percentile stretch (2-98%)
    for c in range(3):
        p2, p98 = np.percentile(rgb[:, :, c], [2, 98])
        if p98 > p2:
            rgb[:, :, c] = np.clip((rgb[:, :, c] - p2) / (p98 - p2), 0, 1)
        else:
            rgb[:, :, c] = 0

    return rgb


def compute_mode_specific_valid_fraction(raw_data: np.ndarray, mode: str) -> np.ndarray:
    """
    Compute valid data fraction for the specific frames used by each mode.

    Args:
        raw_data: Shape (14, 9, H, W) - all 14 quarterly frames
        mode: 'annual', 'bi_temporal', or 'bi_seasonal'

    Returns:
        Valid fraction (H, W) - fraction of relevant frames with valid data
    """
    if mode == 'bi_temporal':
        # Uses 2018 and 2024 annual composites (Q2+Q3 for each)
        # Check frames 0,1 (2018 Q2,Q3) and 12,13 (2024 Q2,Q3)
        frame_indices = [0, 1, 12, 13]
    elif mode == 'annual':
        # Uses all Q2+Q3 pairs for 7 years (all 14 frames)
        frame_indices = list(range(14))
    elif mode == 'bi_seasonal':
        # Uses all 14 quarterly frames
        frame_indices = list(range(14))
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Check for NaN across all bands for selected frames
    selected_data = raw_data[frame_indices]  # (N, 9, H, W)
    has_nan = np.isnan(selected_data).any(axis=1)  # (N, H, W)
    valid_fraction = (~has_nan).mean(axis=0)  # (H, W)

    return valid_fraction


# =============================================================================
# METRICS
# =============================================================================

def compute_sample_metrics(pred_prob: np.ndarray, mask: np.ndarray, threshold: float = 0.5):
    """
    Compute metrics for a single sample.

    Args:
        pred_prob: Probability map (H, W)
        mask: Ground truth mask (H, W) - must be binary 0/1
        threshold: Binary threshold

    Returns:
        dict with iou, f1, precision, recall, tp, fp, fn counts
    """
    pred_binary = (pred_prob > threshold).astype(float)

    # Ensure mask is binary
    mask_binary = (mask > 0).astype(float)

    tp = ((pred_binary == 1) & (mask_binary == 1)).sum()
    fp = ((pred_binary == 1) & (mask_binary == 0)).sum()
    fn = ((pred_binary == 0) & (mask_binary == 1)).sum()

    union = tp + fp + fn
    iou = tp / union if union > 0 else 1.0  # union=0 -> perfect match

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'iou': iou,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
    }


# =============================================================================
# TILE SELECTION
# =============================================================================

def select_tiles_by_mean_iou(iou_data: dict, num_examples: int = 3):
    """
    Select tiles based on MEAN IoU across all three conditions.

    This ensures the same tiles are used across all modes for apples-to-apples comparison.
    """
    refids = iou_data['refids']
    annual_iou = np.array(iou_data['annual_iou'])
    bi_temporal_iou = np.array(iou_data['bi_temporal_iou'])
    bi_seasonal_iou = np.array(iou_data['bi_seasonal_iou'])

    # Compute mean IoU across all conditions
    mean_iou = (annual_iou + bi_temporal_iou + bi_seasonal_iou) / 3

    # Compute percentiles based on mean IoU
    p25 = np.percentile(mean_iou, 25)
    p40 = np.percentile(mean_iou, 40)
    p60 = np.percentile(mean_iou, 60)
    p75 = np.percentile(mean_iou, 75)

    print(f"\nMean IoU Percentiles (across all conditions):")
    print(f"  25th: {p25*100:.1f}%")
    print(f"  40th: {p40*100:.1f}%")
    print(f"  60th: {p60*100:.1f}%")
    print(f"  75th: {p75*100:.1f}%")

    # Select examples for each category
    categories = {
        'good': mean_iou >= p75,
        'hard': (mean_iou >= p40) & (mean_iou <= p60),
        'failure': mean_iou <= p25,
    }

    selected_samples = {}
    for cat, mask in categories.items():
        indices = np.where(mask)[0]

        if cat == 'good':
            sorted_idx = indices[np.argsort(mean_iou[indices])[::-1]]
        elif cat == 'failure':
            sorted_idx = indices[np.argsort(mean_iou[indices])]
        else:
            sorted_idx = indices[np.argsort(np.abs(mean_iou[indices] - np.median(mean_iou[indices])))]

        selected_samples[cat] = sorted_idx[:num_examples]
        print(f"\n{cat.upper()} tiles ({len(indices)} total, selecting {len(selected_samples[cat])}):")
        for idx in selected_samples[cat]:
            print(f"  {refids[idx][:30]}...")
            print(f"    Mean: {mean_iou[idx]*100:.1f}% | Annual: {annual_iou[idx]*100:.1f}% | "
                  f"Bi-temp: {bi_temporal_iou[idx]*100:.1f}% | Bi-seas: {bi_seasonal_iou[idx]*100:.1f}%")

    return selected_samples


def select_divergence_tiles(iou_data: dict, num_examples: int = 2):
    """
    Select tiles where modes diverge significantly.

    These are cases where one mode performs well but another fails,
    which are informative for understanding mode-specific behavior.
    """
    refids = iou_data['refids']
    annual_iou = np.array(iou_data['annual_iou'])
    bi_temporal_iou = np.array(iou_data['bi_temporal_iou'])
    bi_seasonal_iou = np.array(iou_data['bi_seasonal_iou'])

    # Compute divergence metrics
    annual_vs_bitemporal = annual_iou - bi_temporal_iou
    annual_vs_biseasonal = annual_iou - bi_seasonal_iou
    max_minus_min = np.maximum.reduce([annual_iou, bi_temporal_iou, bi_seasonal_iou]) - \
                    np.minimum.reduce([annual_iou, bi_temporal_iou, bi_seasonal_iou])

    selected = {}

    # Tiles where Annual >> Bi-temporal (Annual wins)
    idx_annual_wins_bt = np.argsort(annual_vs_bitemporal)[::-1][:num_examples]
    selected['annual_beats_bitemporal'] = idx_annual_wins_bt
    print(f"\nDIVERGENCE: Annual >> Bi-temporal:")
    for idx in idx_annual_wins_bt:
        print(f"  {refids[idx][:30]}... | Annual: {annual_iou[idx]*100:.1f}% vs Bi-temp: {bi_temporal_iou[idx]*100:.1f}% (Δ={annual_vs_bitemporal[idx]*100:+.1f}pp)")

    # Tiles where Bi-temporal >> Annual (Bi-temporal wins)
    idx_bitemporal_wins = np.argsort(annual_vs_bitemporal)[:num_examples]
    selected['bitemporal_beats_annual'] = idx_bitemporal_wins
    print(f"\nDIVERGENCE: Bi-temporal >> Annual:")
    for idx in idx_bitemporal_wins:
        print(f"  {refids[idx][:30]}... | Bi-temp: {bi_temporal_iou[idx]*100:.1f}% vs Annual: {annual_iou[idx]*100:.1f}% (Δ={-annual_vs_bitemporal[idx]*100:+.1f}pp)")

    # Tiles with largest overall divergence (max - min)
    idx_max_divergence = np.argsort(max_minus_min)[::-1][:num_examples]
    selected['max_divergence'] = idx_max_divergence
    print(f"\nDIVERGENCE: Largest spread across modes:")
    for idx in idx_max_divergence:
        print(f"  {refids[idx][:30]}... | Annual: {annual_iou[idx]*100:.1f}% | Bi-temp: {bi_temporal_iou[idx]*100:.1f}% | Bi-seas: {bi_seasonal_iou[idx]*100:.1f}% (spread={max_minus_min[idx]*100:.1f}pp)")

    return selected


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_tpfpfn_overlay(pred_prob: np.ndarray, mask: np.ndarray,
                          rgb: np.ndarray, threshold: float = 0.5,
                          alpha: float = 0.7) -> np.ndarray:
    """Create TP/FP/FN overlay on RGB image."""
    pred_binary = pred_prob > threshold
    mask_binary = mask > 0

    tp_mask = pred_binary & mask_binary
    fp_mask = pred_binary & (~mask_binary)
    fn_mask = (~pred_binary) & mask_binary

    overlay = rgb.copy()

    overlay[tp_mask] = alpha * COLORS['TP'] + (1 - alpha) * overlay[tp_mask]
    overlay[fp_mask] = alpha * COLORS['FP'] + (1 - alpha) * overlay[fp_mask]
    overlay[fn_mask] = alpha * COLORS['FN'] + (1 - alpha) * overlay[fn_mask]

    return overlay


def plot_cross_mode_comparison(
    refid: str,
    raw_data: np.ndarray,
    predictions: dict,
    mask: np.ndarray,
    metrics: dict,
    category: str,
    output_dir: Path,
    threshold: float = 0.5,
):
    """
    Generate cross-mode comparison figure with context row.

    Layout: 4 rows x 3 columns.
      Row 0 (context): 2018 Composite | 2024 Composite | Ground Truth
      Row 1-3 (models): Probability | Binary Prediction | TP/FP/FN Overlay
    Labels (a)-(c) above context row, (d)-(f) below last model row.
    """
    fig = plt.figure(figsize=(8, 10.5))

    # Create grid: 1 context row + 3 model rows, with extra gap after context row
    gs = fig.add_gridspec(
        4, 3,
        hspace=0.08, wspace=0.08,
        height_ratios=[1, 1, 1, 1],
    )

    # Compute 2018 and 2024 RGB composites
    composite_2018 = compute_annual_composite(raw_data, year_idx=0)  # 2018
    rgb_2018 = get_rgb_from_composite(composite_2018)
    composite_2024 = compute_annual_composite(raw_data, year_idx=6)  # 2024
    rgb_2024 = get_rgb_from_composite(composite_2024)

    # Center crop (same as model)
    H, W = rgb_2024.shape[:2]
    start_h = (H - 64) // 2
    start_w = (W - 64) // 2

    assert mask.shape == (64, 64), f"Mask shape mismatch: expected (64, 64), got {mask.shape}"
    rgb_2018 = rgb_2018[start_h:start_h+64, start_w:start_w+64]
    rgb_2024 = rgb_2024[start_h:start_h+64, start_w:start_w+64]

    # --- Context row (row 0) ---
    context_labels = ['(a) 2018 Composite', '(b) 2024 Composite', '(c) Ground Truth']

    # Col 0: 2018 RGB
    ax_2018 = fig.add_subplot(gs[0, 0])
    ax_2018.imshow(rgb_2018)
    ax_2018.set_ylabel('Input', fontsize=10, fontweight='bold')
    ax_2018.set_xticks([])
    ax_2018.set_yticks([])
    ax_2018.set_title(context_labels[0], fontsize=9)

    # Col 1: 2024 RGB
    ax_2024 = fig.add_subplot(gs[0, 1])
    ax_2024.imshow(rgb_2024)
    ax_2024.set_xticks([])
    ax_2024.set_yticks([])
    ax_2024.set_title(context_labels[1], fontsize=9)

    # Col 2: Ground truth mask
    ax_gt = fig.add_subplot(gs[0, 2])
    ax_gt.imshow(mask, cmap='gray', vmin=0, vmax=1)
    ax_gt.set_xticks([])
    ax_gt.set_yticks([])
    ax_gt.set_title(context_labels[2], fontsize=9)

    # --- Model rows (rows 1-3) ---
    mode_order = ['bi_temporal', 'annual', 'bi_seasonal']
    mode_names = ['LSTM-2', 'LSTM-7', 'LSTM-14']
    model_labels = ['(d) Probability', '(e) Prediction', '(f) TP / FP / FN']
    ax_first_model_row = None  # track for separator positioning

    for row_idx, (condition, name) in enumerate(zip(mode_order, mode_names)):
        grid_row = row_idx + 1  # offset by context row
        pred_prob = predictions[condition]
        m = metrics[condition]
        iou_pct = m['iou'] * 100

        # Row label with IoU
        row_label = f"{name}\n({iou_pct:.1f}%)"

        # Column 0: Probability map
        ax_prob = fig.add_subplot(gs[grid_row, 0])
        ax_prob.imshow(pred_prob, cmap='viridis', vmin=0, vmax=1)
        ax_prob.set_ylabel(row_label, fontsize=10, fontweight='bold')
        ax_prob.set_xticks([])
        ax_prob.set_yticks([])
        if row_idx == 0:
            ax_first_model_row = ax_prob
        if row_idx == 2:  # Last model row
            ax_prob.text(0.5, -0.12, model_labels[0], transform=ax_prob.transAxes,
                        fontsize=9, ha='center')

        # Column 1: Binary prediction
        ax_pred = fig.add_subplot(gs[grid_row, 1])
        ax_pred.imshow(pred_prob > threshold, cmap='gray', vmin=0, vmax=1)
        ax_pred.set_xticks([])
        ax_pred.set_yticks([])
        if row_idx == 2:
            ax_pred.text(0.5, -0.12, model_labels[1], transform=ax_pred.transAxes,
                        fontsize=9, ha='center')

        # Column 2: TP/FP/FN overlay
        ax_overlay = fig.add_subplot(gs[grid_row, 2])
        overlay = create_tpfpfn_overlay(pred_prob, mask, rgb_2024, threshold=threshold)
        ax_overlay.imshow(overlay)
        ax_overlay.set_xticks([])
        ax_overlay.set_yticks([])
        if row_idx == 2:
            ax_overlay.text(0.5, -0.12, model_labels[2], transform=ax_overlay.transAxes,
                           fontsize=9, ha='center')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.06)

    # Visual separator between context row and model rows (after layout is finalized)
    row0_bottom = ax_gt.get_position().y0  # bottom of context row
    row1_top = ax_first_model_row.get_position().y1  # top of first model row
    sep_y = (row0_bottom + row1_top) / 2.0
    fig.add_artist(plt.Line2D(
        [0.06, 0.97], [sep_y, sep_y],
        transform=fig.transFigure,
        color='gray', linewidth=0.8, linestyle='--'
    ))

    # Save figure
    output_file = output_dir / f"crossmode_{category}_{refid[:30]}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return output_file, metrics


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def generate_figure_for_refid(
    refid: str,
    category: str,
    refids: list,
    fold_assignments: dict,
    fold_train_refids: dict,
    output_dir: Path,
    device: torch.device,
):
    """Generate a cross-mode comparison figure for a single refid."""
    fold = fold_assignments.get(refid, -1)
    if fold < 0:
        print(f"  Warning: Could not determine fold for {refid}, skipping")
        return None

    # Load raw data
    raw_data = load_raw_sentinel2(refid)
    H_raw, W_raw = raw_data.shape[2], raw_data.shape[3]

    # Load and binarize mask with proper alignment
    start_h = (H_raw - 64) // 2
    start_w = (W_raw - 64) // 2
    mask_full = load_mask(refid, target_shape=(H_raw, W_raw))
    mask = mask_full[start_h:start_h+64, start_w:start_w+64]
    assert mask.shape == (64, 64), f"Mask shape after crop: {mask.shape}"

    # Get normalization stats for this fold (cached)
    train_refids = fold_train_refids[fold]
    norm_stats = NORM_STATS_CACHE.get_stats(fold, train_refids)

    # Get predictions for each condition
    predictions = {}
    metrics = {}

    for condition, exp_config in EXPERIMENTS.items():
        temporal_sampling = exp_config['temporal_sampling']

        try:
            model, config = MODEL_CACHE.get_model(condition, fold, device)
        except FileNotFoundError as e:
            print(f"  Warning: {e}")
            continue

        from albumentations import Compose, CenterCrop

        transform = Compose([
            CenterCrop(config['image_size'], config['image_size']),
        ])

        dataset = MultiTemporalSentinel2Dataset(
            refids=[refid],
            temporal_sampling=temporal_sampling,
            normalization_stats=norm_stats,
            transform=transform,
            output_format="LSTM",
        )

        sample = dataset[0]
        images = sample['image'].unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(images)
            pred_prob = torch.sigmoid(output).cpu().numpy().squeeze()

        predictions[condition] = pred_prob
        metrics[condition] = compute_sample_metrics(pred_prob, mask, threshold=0.5)

    if len(predictions) == 3:
        output_file, _ = plot_cross_mode_comparison(
            refid=refid,
            raw_data=raw_data,
            predictions=predictions,
            mask=mask,
            metrics=metrics,
            category=category,
            output_dir=output_dir,
        )
        print(f"  Saved: {output_file.name}")
        return output_file
    else:
        print(f"  Warning: Missing predictions for {refid}, skipping")
        return None


def run_qualitative_analysis(
    num_examples: int = 3,
    output_dir: Path = None,
    device: torch.device = None,
    specific_refids: list = None,
):
    """
    Run cross-mode qualitative analysis.

    Args:
        specific_refids: If provided, only generate figures for these refids
                         (list of refid strings). Skips automatic tile selection.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if output_dir is None:
        output_dir = MT_EXPERIMENTS_DIR.parent / "analysis" / "cv_qualitative"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("CV QUALITATIVE ANALYSIS - CROSS-MODE COMPARISON")
    print(f"{'='*60}")

    # Load per-sample IoU from statistical analysis
    iou_file = MT_EXPERIMENTS_DIR.parent / "analysis" / "per_sample_iou.json"
    if not iou_file.exists():
        raise FileNotFoundError(f"Per-sample IoU file not found: {iou_file}\n"
                                "Run statistical_analysis_persample.py first.")

    with open(iou_file) as f:
        iou_data = json.load(f)

    refids = iou_data['refids']

    # Get fold assignments using same logic as training
    print("\nComputing fold assignments (using sample_change_levels.csv)...")
    fold_assignments, fold_train_refids = get_fold_assignments(refids)

    # --- Specific refids mode ---
    if specific_refids:
        print(f"\nGenerating figures for {len(specific_refids)} specific tile(s)...")
        for refid in specific_refids:
            # Match partial refid prefixes (filenames are truncated to 30 chars)
            matched = [r for r in refids if r.startswith(refid) or refid.startswith(r)]
            if not matched:
                print(f"  Warning: refid '{refid}' not found in dataset, skipping")
                continue
            full_refid = matched[0]
            print(f"  Processing: {full_refid}")
            generate_figure_for_refid(
                refid=full_refid,
                category="selected",
                refids=refids,
                fold_assignments=fold_assignments,
                fold_train_refids=fold_train_refids,
                output_dir=output_dir,
                device=device,
            )

        print(f"\n{'='*60}")
        print(f"Visualizations saved to: {output_dir}")
        print(f"{'='*60}")
        return output_dir

    # --- Automatic tile selection mode ---
    print("\n" + "-"*60)
    print("TILE SELECTION: Mean IoU-based")
    print("-"*60)
    selected_mean = select_tiles_by_mean_iou(iou_data, num_examples=num_examples)

    print("\n" + "-"*60)
    print("TILE SELECTION: Divergence-based")
    print("-"*60)
    selected_divergence = select_divergence_tiles(iou_data, num_examples=2)

    # Combine all selected tiles
    all_selected = {}
    for cat, indices in selected_mean.items():
        all_selected[cat] = indices
    for cat, indices in selected_divergence.items():
        all_selected[f"divergence_{cat}"] = indices

    # Generate visualizations
    print("\n" + "="*60)
    print("Generating cross-mode comparison figures...")
    print("="*60)

    for cat, indices in all_selected.items():
        print(f"\n--- {cat.upper()} ---")

        for idx in tqdm(indices, desc=f"{cat}"):
            refid = refids[idx]
            generate_figure_for_refid(
                refid=refid,
                category=cat,
                refids=refids,
                fold_assignments=fold_assignments,
                fold_train_refids=fold_train_refids,
                output_dir=output_dir,
                device=device,
            )

    print(f"\n{'='*60}")
    print(f"Visualizations saved to: {output_dir}")
    print(f"{'='*60}")

    return output_dir


def main():
    parser = argparse.ArgumentParser(description="CV Qualitative Analysis")
    parser.add_argument('--num-examples', type=int, default=3,
                        help='Number of examples per category (good/hard/failure)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--refids', type=str, nargs='+', default=None,
                        help='Generate figures for specific refid(s) only. '
                             'Prefix matching supported.')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    output_dir = Path(args.output_dir) if args.output_dir else None

    run_qualitative_analysis(
        num_examples=args.num_examples,
        output_dir=output_dir,
        device=device,
        specific_refids=args.refids,
    )

    print("\n" + "="*60)
    print("CV Qualitative Analysis Complete")
    print("="*60)
    print("\nReview the generated figures to check for:")
    print("  1. Correct spatial alignment between input, GT, and prediction")
    print("  2. Annual composite shows Q2/Q3 mean (same as model input)")
    print("  3. Mode-specific valid data reflects actual frames used")
    print("  4. Expected failure modes (clouds, bare soil, etc.)")
    print("  5. Divergence tiles show where modes differ significantly")
    print("\nIf sanity checks pass, proceed with test set evaluation.")


if __name__ == "__main__":
    main()
