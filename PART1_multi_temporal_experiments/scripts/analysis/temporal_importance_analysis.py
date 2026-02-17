#!/usr/bin/env python3
"""
Temporal Contribution Analysis: WHY does LSTM-7 outperform?

Three experiments testing three hypotheses:

1. Temporal Gradient Attribution (Hypothesis 1: temporal signal)
   - Which years does the trained LSTM-7 rely on most?
   - Computes mean |dL/dx_t| per timestep across OOF tiles.

2. Input Temporal Autocorrelation (Hypothesis 2: noise/redundancy)
   - How redundant are consecutive frames in T=14 vs T=7?
   - Computes pairwise Pearson correlation + NaN fractions.

3. Training Dynamics Comparison (Hypothesis 3: overfitting/capacity)
   - Does T=14 overfit more than T=7?
   - Extracts train/val IoU curves from existing history.json logs.

Usage:
    # Full run (Experiment 1 needs GPU)
    python temporal_importance_analysis.py

    # CPU-only experiments (2 & 3)
    python temporal_importance_analysis.py --skip-gradient
"""

import sys
import argparse
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import warnings

# Add paths for imports
script_dir = Path(__file__).resolve().parent
parent_dir = script_dir.parent.parent  # PART1_multi_temporal_experiments/
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(parent_dir.parent))  # NINA_fordypningsoppgave/
sys.path.insert(0, str(parent_dir / "scripts" / "modeling"))

from PART1_multi_temporal_experiments.config import (
    MT_EXPERIMENTS_DIR, DATA_DIR, YEARS, QUARTERS, SENTINEL2_BANDS
)
from PART1_multi_temporal_experiments.scripts.data_preparation.dataset_multitemporal import (
    MultiTemporalSentinel2Dataset, compute_normalization_stats
)
from models_multitemporal import create_multitemporal_model

# Reuse fold logic and caching from qualitative analysis
from PART1_multi_temporal_experiments.scripts.analysis.qualitative_cv_analysis import (
    get_fold_assignments, ModelCache, NormStatsCache, load_raw_sentinel2,
    load_mask, EXPERIMENTS,
)

# Global caches
MODEL_CACHE = ModelCache()
NORM_STATS_CACHE = NormStatsCache()


# =============================================================================
# EXPERIMENT 1: TEMPORAL GRADIENT ATTRIBUTION
# =============================================================================

def compute_temporal_gradient_importance(
    refids: list,
    fold_assignments: dict,
    fold_train_refids: dict,
    device: torch.device,
) -> dict:
    """
    Compute gradient-based temporal importance for the LSTM-7 model.

    For each OOF tile, computes importance_t = mean(|dL/dx_t|) over spatial
    dimensions and bands, giving a 7-element vector per tile.

    Returns:
        dict with:
            'per_tile': np.array (N, 7) - per-tile importance
            'mean': np.array (7,) - mean importance per year
            'std': np.array (7,) - std of importance per year
            'refids': list - corresponding refids
            'per_tile_iou': np.array (N,) - IoU per tile (for stratification)
    """
    from albumentations import Compose, CenterCrop

    condition = 'annual'
    exp_config = EXPERIMENTS[condition]
    T = exp_config['T']

    all_importance = []
    all_iou = []
    valid_refids = []

    criterion = nn.BCEWithLogitsLoss()

    for refid in tqdm(refids, desc="Gradient attribution"):
        fold = fold_assignments.get(refid)
        if fold is None:
            continue

        try:
            model, config = MODEL_CACHE.get_model(condition, fold, device)
        except FileNotFoundError:
            continue

        # Get normalization stats
        train_refids = fold_train_refids[fold]
        norm_stats = NORM_STATS_CACHE.get_stats(fold, train_refids)

        # Build single-sample dataset
        transform = Compose([CenterCrop(config['image_size'], config['image_size'])])
        dataset = MultiTemporalSentinel2Dataset(
            refids=[refid],
            temporal_sampling='annual',
            normalization_stats=norm_stats,
            transform=transform,
            output_format="LSTM",
        )

        sample = dataset[0]
        images = sample['image'].unsqueeze(0).to(device)   # (1, 7, 9, 64, 64)
        mask = sample['mask'].unsqueeze(0).to(device)       # (1, 64, 64)

        # Enable gradient on input
        images.requires_grad_(True)

        # Forward pass
        model.eval()
        output = model(images)  # (1, 1, 64, 64)
        loss = criterion(output.squeeze(1), mask)

        # Backward pass
        loss.backward()

        # Extract per-timestep importance: mean |gradient| over (batch, bands, H, W)
        grad = images.grad  # (1, 7, 9, 64, 64)
        importance = grad.abs().mean(dim=(0, 2, 3, 4)).detach().cpu().numpy()  # (7,)

        all_importance.append(importance)
        valid_refids.append(refid)

        # Also compute IoU for stratification
        with torch.no_grad():
            pred_prob = torch.sigmoid(output).cpu().numpy().squeeze()
            pred_binary = (pred_prob > 0.5).astype(float)
            mask_np = mask.cpu().numpy().squeeze()
            tp = ((pred_binary == 1) & (mask_np > 0)).sum()
            fp = ((pred_binary == 1) & (mask_np == 0)).sum()
            fn = ((pred_binary == 0) & (mask_np > 0)).sum()
            union = tp + fp + fn
            iou = tp / union if union > 0 else 1.0
            all_iou.append(iou)

        # Clear gradients
        images.grad = None
        model.zero_grad()

    all_importance = np.array(all_importance)  # (N, 7)
    all_iou = np.array(all_iou)

    return {
        'per_tile': all_importance,
        'mean': all_importance.mean(axis=0),
        'std': all_importance.std(axis=0),
        'refids': valid_refids,
        'per_tile_iou': all_iou,
    }


def plot_gradient_importance(importance_data: dict, output_dir: Path):
    """
    Plot temporal gradient importance as bar chart, stratified by tile quality.
    """
    years = YEARS  # [2018, ..., 2024]
    per_tile = importance_data['per_tile']       # (N, 7)
    per_tile_iou = importance_data['per_tile_iou']  # (N,)

    # Normalize per tile to relative importance (sum to 1)
    row_sums = per_tile.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    per_tile_rel = per_tile / row_sums

    # Overall mean
    overall_mean = per_tile_rel.mean(axis=0)
    overall_std = per_tile_rel.std(axis=0)

    # Stratify by IoU quartiles
    p25 = np.percentile(per_tile_iou, 25)
    p75 = np.percentile(per_tile_iou, 75)

    good_mask = per_tile_iou >= p75
    hard_mask = (per_tile_iou >= p25) & (per_tile_iou < p75)
    failure_mask = per_tile_iou < p25

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # --- Panel (a): Overall importance ---
    ax = axes[0]
    x = np.arange(len(years))
    bars = ax.bar(x, overall_mean, yerr=overall_std, capsize=3,
                  color='#4C72B0', edgecolor='black', linewidth=0.5, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(years, fontsize=9)
    ax.set_ylabel('Relative gradient importance', fontsize=10)
    ax.set_xlabel('Year', fontsize=10)
    ax.set_title('(a) Mean temporal importance (LSTM-7)', fontsize=11)
    ax.axhline(y=1.0 / len(years), color='gray', linestyle='--', linewidth=0.8,
               label='Uniform')
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)

    # --- Panel (b): Stratified by tile quality ---
    ax = axes[1]
    width = 0.25
    strata = [
        ('Good (IoU≥p75)', good_mask, '#55A868'),
        ('Medium', hard_mask, '#4C72B0'),
        ('Failure (IoU<p25)', failure_mask, '#C44E52'),
    ]

    for i, (label, mask, color) in enumerate(strata):
        if mask.sum() == 0:
            continue
        stratum_mean = per_tile_rel[mask].mean(axis=0)
        stratum_std = per_tile_rel[mask].std(axis=0)
        offset = (i - 1) * width
        ax.bar(x + offset, stratum_mean, width, yerr=stratum_std, capsize=2,
               label=f'{label} (n={mask.sum()})', color=color,
               edgecolor='black', linewidth=0.5, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(years, fontsize=9)
    ax.set_ylabel('Relative gradient importance', fontsize=10)
    ax.set_xlabel('Year', fontsize=10)
    ax.set_title('(b) Importance stratified by tile quality', fontsize=11)
    ax.axhline(y=1.0 / len(years), color='gray', linestyle='--', linewidth=0.8)
    ax.legend(fontsize=7, loc='upper left')
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    out_path = output_dir / "temporal_gradient_importance.pdf"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.savefig(out_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")

    return out_path


# =============================================================================
# EXPERIMENT 2: INPUT TEMPORAL AUTOCORRELATION
# =============================================================================

def compute_temporal_autocorrelation(refids: list) -> dict:
    """
    Compute pairwise Pearson correlation between consecutive frames
    for both T=14 (quarterly) and T=7 (annual) representations.

    Also computes per-quarter NaN fraction.

    Returns:
        dict with correlation matrices, NaN fractions, etc.
    """
    # We'll compute the full 14x14 and 7x7 correlation matrices
    all_corr_14 = []
    all_corr_7 = []
    nan_fractions_14 = []  # (N, 14)

    for refid in tqdm(refids, desc="Autocorrelation analysis"):
        try:
            raw_data = load_raw_sentinel2(refid)  # (14, 9, H, W)
        except Exception as e:
            print(f"  Skipping {refid}: {e}")
            continue

        T_full, C, H, W = raw_data.shape

        # --- NaN fractions per quarterly frame ---
        nan_frac = np.array([
            np.isnan(raw_data[t]).mean() for t in range(T_full)
        ])
        nan_fractions_14.append(nan_frac)

        # --- Quarterly (T=14) correlation ---
        # Flatten each frame to a vector, compute pairwise correlation
        frames_14 = []
        for t in range(T_full):
            frame = raw_data[t].flatten()  # (9*H*W,)
            frames_14.append(frame)
        frames_14 = np.stack(frames_14, axis=0)  # (14, 9*H*W)

        # Replace NaN with per-frame mean for correlation computation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for t in range(T_full):
                nan_mask = np.isnan(frames_14[t])
                if nan_mask.any():
                    frames_14[t, nan_mask] = np.nanmean(frames_14[t])

        # Pairwise Pearson correlation
        if not np.isnan(frames_14).any():
            corr_14 = np.corrcoef(frames_14)  # (14, 14)
            all_corr_14.append(corr_14)

        # --- Annual (T=7) correlation ---
        # Build annual composites (same logic as dataset)
        annual_frames = []
        for year_idx in range(len(YEARS)):
            q2_idx = year_idx * 2
            q3_idx = year_idx * 2 + 1
            q2_data = raw_data[q2_idx]
            q3_data = raw_data[q3_idx]

            q2_nan_pct = np.isnan(q2_data).sum() / q2_data.size * 100
            q3_nan_pct = np.isnan(q3_data).sum() / q3_data.size * 100

            if q2_nan_pct > 50 and q3_nan_pct < 20:
                year_data = q3_data
            elif q3_nan_pct > 50 and q2_nan_pct < 20:
                year_data = q2_data
            else:
                year_data = np.nanmean([q2_data, q3_data], axis=0)

            annual_frames.append(year_data.flatten())

        annual_frames = np.stack(annual_frames, axis=0)  # (7, 9*H*W)

        # Replace remaining NaN
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for t in range(len(YEARS)):
                nan_mask = np.isnan(annual_frames[t])
                if nan_mask.any():
                    mean_val = np.nanmean(annual_frames[t])
                    if np.isnan(mean_val):
                        mean_val = 0.0
                    annual_frames[t, nan_mask] = mean_val

        if not np.isnan(annual_frames).any():
            corr_7 = np.corrcoef(annual_frames)  # (7, 7)
            all_corr_7.append(corr_7)

    # Aggregate
    mean_corr_14 = np.mean(all_corr_14, axis=0) if all_corr_14 else None
    mean_corr_7 = np.mean(all_corr_7, axis=0) if all_corr_7 else None
    nan_fractions_14 = np.array(nan_fractions_14) if nan_fractions_14 else None

    # Consecutive-pair correlations
    consec_corr_14 = []
    if mean_corr_14 is not None:
        for t in range(13):
            consec_corr_14.append(mean_corr_14[t, t + 1])

    consec_corr_7 = []
    if mean_corr_7 is not None:
        for t in range(6):
            consec_corr_7.append(mean_corr_7[t, t + 1])

    return {
        'mean_corr_14': mean_corr_14,
        'mean_corr_7': mean_corr_7,
        'consec_corr_14': np.array(consec_corr_14),
        'consec_corr_7': np.array(consec_corr_7),
        'nan_fractions_14': nan_fractions_14,
        'num_tiles': len(all_corr_14),
    }


def plot_autocorrelation(autocorr_data: dict, output_dir: Path):
    """
    Plot autocorrelation analysis: heatmaps + NaN fraction bar chart.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Frame labels
    quarterly_labels = []
    for year in YEARS:
        for q in QUARTERS:
            quarterly_labels.append(f"{year}\nQ{q}")
    annual_labels = [str(y) for y in YEARS]

    # --- Panel (a): T=14 correlation heatmap ---
    ax = axes[0]
    if autocorr_data['mean_corr_14'] is not None:
        im = ax.imshow(autocorr_data['mean_corr_14'], cmap='RdYlBu_r',
                        vmin=0.5, vmax=1.0, aspect='equal')
        ax.set_xticks(range(14))
        ax.set_xticklabels(quarterly_labels, fontsize=5.5, rotation=90)
        ax.set_yticks(range(14))
        ax.set_yticklabels(quarterly_labels, fontsize=5.5)
        plt.colorbar(im, ax=ax, shrink=0.8, label='Pearson r')
    ax.set_title('(a) T=14 frame correlation', fontsize=11)

    # --- Panel (b): T=7 correlation heatmap ---
    ax = axes[1]
    if autocorr_data['mean_corr_7'] is not None:
        im = ax.imshow(autocorr_data['mean_corr_7'], cmap='RdYlBu_r',
                        vmin=0.5, vmax=1.0, aspect='equal')
        ax.set_xticks(range(7))
        ax.set_xticklabels(annual_labels, fontsize=9)
        ax.set_yticks(range(7))
        ax.set_yticklabels(annual_labels, fontsize=9)
        plt.colorbar(im, ax=ax, shrink=0.8, label='Pearson r')
    ax.set_title('(b) T=7 frame correlation', fontsize=11)

    # --- Panel (c): NaN fraction per quarter ---
    ax = axes[2]
    if autocorr_data['nan_fractions_14'] is not None:
        nan_mean = autocorr_data['nan_fractions_14'].mean(axis=0)
        nan_std = autocorr_data['nan_fractions_14'].std(axis=0)

        x = np.arange(14)
        colors = ['#4C72B0' if q == 2 else '#DD8452' for _ in YEARS for q in QUARTERS]
        bars = ax.bar(x, nan_mean * 100, yerr=nan_std * 100, capsize=2,
                      color=colors, edgecolor='black', linewidth=0.5, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(quarterly_labels, fontsize=5.5, rotation=90)
        ax.set_ylabel('NaN fraction (%)', fontsize=10)

        # Legend for quarters
        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(facecolor='#4C72B0', label='Q2 (Apr-Jun)'),
            Patch(facecolor='#DD8452', label='Q3 (Jul-Sep)'),
        ], fontsize=8)
    ax.set_title('(c) Cloud/nodata per quarter', fontsize=11)

    plt.tight_layout()
    out_path = output_dir / "temporal_autocorrelation.pdf"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.savefig(out_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")

    # Print summary statistics
    if autocorr_data['consec_corr_14'] is not None and len(autocorr_data['consec_corr_14']) > 0:
        print(f"\n  Consecutive-frame correlation summary:")
        cc14 = autocorr_data['consec_corr_14']
        cc7 = autocorr_data['consec_corr_7']

        # Separate intra-year (Q2->Q3) and inter-year (Q3->Q2) for T=14
        intra_year = cc14[0::2]  # Q2->Q3 within same year
        inter_year = cc14[1::2]  # Q3->Q2 across years
        print(f"    T=14 intra-year (Q2→Q3):  mean={intra_year.mean():.3f} ± {intra_year.std():.3f}")
        print(f"    T=14 inter-year (Q3→Q2'):  mean={inter_year.mean():.3f} ± {inter_year.std():.3f}")
        print(f"    T=7  consecutive (annual): mean={cc7.mean():.3f} ± {cc7.std():.3f}")

    return out_path


# =============================================================================
# EXPERIMENT 3: TRAINING DYNAMICS COMPARISON
# =============================================================================

def load_training_histories() -> dict:
    """
    Load training history (train/val IoU per epoch) for all three RQ1 conditions.

    Returns:
        dict {condition: {fold: {'train': [...], 'val': [...]}}}
    """
    histories = {}

    for condition, exp_config in EXPERIMENTS.items():
        exp_name = exp_config['name']
        histories[condition] = {}

        for fold in range(5):
            history_path = MT_EXPERIMENTS_DIR / f"{exp_name}_fold{fold}" / "history.json"
            if not history_path.exists():
                print(f"  Warning: Missing {history_path}")
                continue

            with open(history_path) as f:
                history = json.load(f)

            train_iou = [epoch['iou'] for epoch in history['train']]
            val_iou = [epoch['iou'] for epoch in history['val']]
            train_loss = [epoch['loss'] for epoch in history['train']]
            val_loss = [epoch['loss'] for epoch in history['val']]

            histories[condition][fold] = {
                'train_iou': train_iou,
                'val_iou': val_iou,
                'train_loss': train_loss,
                'val_loss': val_loss,
            }

    return histories


def plot_training_dynamics(histories: dict, output_dir: Path):
    """
    Plot training dynamics comparison: overlaid learning curves for T=2/T=7/T=14.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    conditions_display = {
        'bi_temporal': ('LSTM-2 (T=2)', '#C44E52'),
        'annual': ('LSTM-7 (T=7)', '#4C72B0'),
        'bi_seasonal': ('LSTM-14 (T=14)', '#55A868'),
    }

    # Collect summary stats for annotation
    summary = {}

    for condition, (label, color) in conditions_display.items():
        if condition not in histories:
            continue

        folds = histories[condition]
        if not folds:
            continue

        # Get max epoch count across folds
        max_epochs = max(len(folds[f]['train_iou']) for f in folds)

        # Pad shorter histories with last value
        def pad_to_length(arr, length):
            if len(arr) >= length:
                return arr[:length]
            return arr + [arr[-1]] * (length - len(arr))

        train_iou_all = np.array([
            pad_to_length(folds[f]['train_iou'], max_epochs) for f in folds
        ])
        val_iou_all = np.array([
            pad_to_length(folds[f]['val_iou'], max_epochs) for f in folds
        ])

        epochs = np.arange(1, max_epochs + 1)

        train_mean = train_iou_all.mean(axis=0)
        val_mean = val_iou_all.mean(axis=0)
        train_std = train_iou_all.std(axis=0)
        val_std = val_iou_all.std(axis=0)

        # Best val epoch (per fold, then average)
        best_epochs = [np.argmax(folds[f]['val_iou']) + 1 for f in folds]
        best_val_ious = [max(folds[f]['val_iou']) for f in folds]

        # Train-val gap at convergence (last 20% of epochs)
        last_20pct = max(1, max_epochs // 5)
        train_final = train_mean[-last_20pct:].mean()
        val_final = val_mean[-last_20pct:].mean()
        gap = train_final - val_final

        summary[condition] = {
            'best_epoch_mean': np.mean(best_epochs),
            'best_epoch_std': np.std(best_epochs),
            'best_val_iou_mean': np.mean(best_val_ious),
            'best_val_iou_std': np.std(best_val_ious),
            'train_val_gap': gap,
            'val_stability': val_std[-last_20pct:].mean(),
        }

        # --- Panel (a): IoU curves ---
        ax = axes[0]
        ax.plot(epochs, train_mean, color=color, linestyle='--', alpha=0.5, linewidth=1)
        ax.plot(epochs, val_mean, color=color, linestyle='-', linewidth=1.5, label=label)
        ax.fill_between(epochs, val_mean - val_std, val_mean + val_std,
                         alpha=0.1, color=color)

        # Mark best validation epoch
        best_ep_idx = int(np.mean(best_epochs)) - 1
        if 0 <= best_ep_idx < len(val_mean):
            ax.axvline(x=best_ep_idx + 1, color=color, linestyle=':', alpha=0.4, linewidth=0.8)

    # Finalize panel (a)
    ax = axes[0]
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('IoU', fontsize=10)
    ax.set_title('(a) Validation IoU curves (5-fold mean ± std)', fontsize=11)
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)
    # Add dashed line legend note
    ax.plot([], [], color='gray', linestyle='--', alpha=0.5, label='Train (dashed)')
    ax.legend(fontsize=8, loc='lower right')

    # --- Panel (b): Train-val gap summary ---
    ax = axes[1]
    conditions_order = ['bi_temporal', 'annual', 'bi_seasonal']
    labels_order = []
    gaps = []
    best_epochs_list = []
    colors_list = []
    val_stab = []

    for cond in conditions_order:
        if cond in summary:
            label, color = conditions_display[cond]
            labels_order.append(label)
            gaps.append(summary[cond]['train_val_gap'])
            best_epochs_list.append(summary[cond]['best_epoch_mean'])
            colors_list.append(color)
            val_stab.append(summary[cond]['val_stability'])

    x = np.arange(len(labels_order))
    bars = ax.bar(x, gaps, color=colors_list, edgecolor='black', linewidth=0.5, alpha=0.85)

    # Annotate with best epoch
    for i, (g, ep) in enumerate(zip(gaps, best_epochs_list)):
        ax.text(i, g + 0.002, f'best@{ep:.0f}', ha='center', fontsize=8, style='italic')

    ax.set_xticks(x)
    ax.set_xticklabels(labels_order, fontsize=10)
    ax.set_ylabel('Train - Val IoU gap', fontsize=10)
    ax.set_title('(b) Overfitting indicator (train-val gap)', fontsize=11)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    out_path = output_dir / "training_dynamics_comparison.pdf"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.savefig(out_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")

    # Print summary table
    print(f"\n  Training dynamics summary:")
    print(f"  {'Condition':<20} {'Best epoch':>12} {'Best val IoU':>14} {'Train-Val gap':>14} {'Val stability':>14}")
    print(f"  {'-'*74}")
    for cond in conditions_order:
        if cond not in summary:
            continue
        s = summary[cond]
        label = conditions_display[cond][0]
        print(f"  {label:<20} {s['best_epoch_mean']:>8.0f}±{s['best_epoch_std']:<3.0f}"
              f" {s['best_val_iou_mean']:>10.3f}±{s['best_val_iou_std']:<4.3f}"
              f" {s['train_val_gap']:>14.4f}"
              f" {s['val_stability']:>14.4f}")

    return out_path, summary


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Temporal Contribution Analysis")
    parser.add_argument('--skip-gradient', action='store_true',
                        help='Skip Experiment 1 (gradient attribution, needs GPU)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for figures')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = MT_EXPERIMENTS_DIR.parent / "analysis" / "temporal_contribution"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load per-sample IoU data (provides the list of OOF refids)
    iou_file = MT_EXPERIMENTS_DIR.parent / "analysis" / "per_sample_iou.json"
    if not iou_file.exists():
        raise FileNotFoundError(
            f"Per-sample IoU file not found: {iou_file}\n"
            "Run statistical_analysis_persample.py first."
        )

    with open(iou_file) as f:
        iou_data = json.load(f)
    refids = iou_data['refids']

    print(f"\n{'='*60}")
    print("TEMPORAL CONTRIBUTION ANALYSIS")
    print(f"{'='*60}")
    print(f"Total OOF tiles: {len(refids)}")

    # =========================================================================
    # EXPERIMENT 3: Training Dynamics (CPU, fast — run first)
    # =========================================================================
    print(f"\n{'='*60}")
    print("EXPERIMENT 3: Training Dynamics Comparison")
    print(f"{'='*60}")

    histories = load_training_histories()
    train_dyn_path, train_summary = plot_training_dynamics(histories, output_dir)

    # Save summary as JSON
    summary_path = output_dir / "training_dynamics_summary.json"
    # Convert numpy types for JSON serialization
    serializable_summary = {}
    for k, v in train_summary.items():
        serializable_summary[k] = {sk: float(sv) for sk, sv in v.items()}
    with open(summary_path, 'w') as f:
        json.dump(serializable_summary, f, indent=2)
    print(f"  Saved summary: {summary_path}")

    # =========================================================================
    # EXPERIMENT 2: Autocorrelation (CPU, moderate)
    # =========================================================================
    print(f"\n{'='*60}")
    print("EXPERIMENT 2: Input Temporal Autocorrelation")
    print(f"{'='*60}")

    autocorr_data = compute_temporal_autocorrelation(refids)
    autocorr_path = plot_autocorrelation(autocorr_data, output_dir)

    # Save correlation matrices
    np.savez(
        output_dir / "autocorrelation_data.npz",
        mean_corr_14=autocorr_data['mean_corr_14'],
        mean_corr_7=autocorr_data['mean_corr_7'],
        consec_corr_14=autocorr_data['consec_corr_14'],
        consec_corr_7=autocorr_data['consec_corr_7'],
        nan_fractions_14=autocorr_data['nan_fractions_14'],
    )
    print(f"  Saved data: {output_dir / 'autocorrelation_data.npz'}")

    # =========================================================================
    # EXPERIMENT 1: Gradient Attribution (GPU)
    # =========================================================================
    if args.skip_gradient:
        print(f"\n{'='*60}")
        print("EXPERIMENT 1: Skipped (--skip-gradient)")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print("EXPERIMENT 1: Temporal Gradient Attribution")
        print(f"{'='*60}")

        if device.type != 'cuda':
            print("  WARNING: No GPU detected. Gradient attribution will be slow.")

        # Get fold assignments
        fold_assignments, fold_train_refids = get_fold_assignments(refids)

        importance_data = compute_temporal_gradient_importance(
            refids, fold_assignments, fold_train_refids, device
        )

        grad_path = plot_gradient_importance(importance_data, output_dir)

        # Save raw importance data
        np.savez(
            output_dir / "gradient_importance_data.npz",
            per_tile=importance_data['per_tile'],
            mean=importance_data['mean'],
            std=importance_data['std'],
            per_tile_iou=importance_data['per_tile_iou'],
        )
        with open(output_dir / "gradient_importance_refids.json", 'w') as f:
            json.dump(importance_data['refids'], f)
        print(f"  Saved data: {output_dir / 'gradient_importance_data.npz'}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"\nFigures generated:")
    print(f"  - training_dynamics_comparison.pdf  (Experiment 3)")
    print(f"  - temporal_autocorrelation.pdf      (Experiment 2)")
    if not args.skip_gradient:
        print(f"  - temporal_gradient_importance.pdf  (Experiment 1)")
    print(f"\nInterpretation guide:")
    print(f"  Exp 1: If all years contribute ≈ equally → trajectory matters (H1 supported)")
    print(f"  Exp 2: High intra-year corr (Q2↔Q3) → quarterly is redundant (H2 supported)")
    print(f"  Exp 3: Larger train-val gap for T=14 → overfitting (H3 supported)")


if __name__ == "__main__":
    main()
