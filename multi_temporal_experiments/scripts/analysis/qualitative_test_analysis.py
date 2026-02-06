#!/usr/bin/env python3
"""
Qualitative analysis of TEST SET predictions.

Generates visualizations for all 8 test tiles across all temporal modes.
Uses CV ensemble (average of 5 fold models) same as final test evaluation.

Output:
1. Supplement figures: One detailed figure per test tile (all 8)
2. Main paper figure: 2-3 tiles selected by predefined criteria:
   - Median test-tile IoU under Annual (typical case)
   - Largest improvement Annual − Bi-seasonal (shows the "tail driver")
   - Failure case (IoU = 0 for all modes)

Design notes:
- Main figure: simplified to 2018/2024 RGB, GT, and TP/FP/FN overlay per mode
- Supplement: includes probability maps with colorblind-safe colormap (viridis)
- All figures note: CV ensemble, threshold=0.5, center crop 64x64
"""

import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from tqdm import tqdm
import rasterio
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# Add paths for imports
script_dir = Path(__file__).resolve().parent
parent_dir = script_dir.parent.parent  # multi_temporal_experiments/
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(parent_dir.parent))  # NINA_fordypningsoppgave/
sys.path.insert(0, str(parent_dir / "scripts" / "modeling"))  # For models_multitemporal

from multi_temporal_experiments.config import (
    MT_EXPERIMENTS_DIR, DATA_DIR, YEARS, QUARTERS, SENTINEL2_BANDS
)
from multi_temporal_experiments.scripts.data_preparation.dataset_multitemporal import (
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
    'FN': np.array([0.58, 0.40, 0.74]),    # Purple (magenta-ish)
}


class ModelCache:
    """Cache for loaded models."""
    def __init__(self):
        self._models = {}

    def get_model(self, condition: str, fold: int, device: torch.device):
        key = (condition, fold)
        if key not in self._models:
            exp_config = EXPERIMENTS[condition]
            exp_name = exp_config['name']
            exp_dir = MT_EXPERIMENTS_DIR / f"{exp_name}_fold{fold}"
            config_path = exp_dir / "config.json"
            checkpoint_path = exp_dir / "best_model.pth"

            with open(config_path) as f:
                config = json.load(f)

            model = create_multitemporal_model(
                model_name=config.get('model_name', 'lstm_unet'),
                classes=1,
                encoder_name=config.get('encoder_name', 'resnet50'),
                encoder_weights=None,
                in_channels=len(SENTINEL2_BANDS),
                lstm_hidden_dim=config.get('lstm_hidden_dim', 256),
                lstm_num_layers=config.get('lstm_num_layers', 2),
                skip_aggregation=config.get('skip_aggregation', 'max'),
            )

            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
            model.eval()

            self._models[key] = model

        return self._models[key]


class NormStatsCache:
    """Cache for normalization statistics."""
    def __init__(self):
        self._stats = {}

    def get_stats(self, condition: str, fold: int, train_refids: list):
        key = (condition, fold)
        if key not in self._stats:
            # compute_normalization_stats returns a dict with 'mean' and 'std' keys
            self._stats[key] = compute_normalization_stats(train_refids)
        return self._stats[key]


def load_raw_sentinel2(refid: str) -> np.ndarray:
    """Load raw Sentinel-2 data (no normalization)."""
    s2_path = DATA_DIR / "Sentinel" / f"{refid}_RGBNIRRSWIRQ_Mosaic.tif"
    with rasterio.open(s2_path) as src:
        data = src.read()
    num_full_time_steps = len(YEARS) * len(QUARTERS)
    num_bands = len(SENTINEL2_BANDS)
    data = data.reshape(num_full_time_steps, num_bands, data.shape[1], data.shape[2])
    return data


def load_mask(refid: str, target_shape: tuple = None) -> np.ndarray:
    """Load and binarize mask."""
    from scipy.ndimage import zoom
    mask_path = DATA_DIR / "Land_take_masks" / f"{refid}_mask.tif"
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
    mask = (mask > 0).astype(np.float32)
    if target_shape is not None and mask.shape != target_shape:
        zoom_factors = (target_shape[0] / mask.shape[0], target_shape[1] / mask.shape[1])
        mask = zoom(mask, zoom_factors, order=0)
    return mask


def compute_annual_composite(raw_data: np.ndarray, year_idx: int) -> np.ndarray:
    """Compute annual composite exactly as the model sees it."""
    q2_idx = year_idx * 2
    q3_idx = year_idx * 2 + 1
    q2_data = raw_data[q2_idx]
    q3_data = raw_data[q3_idx]

    q2_nan_pct = np.isnan(q2_data).sum() / q2_data.size * 100
    q3_nan_pct = np.isnan(q3_data).sum() / q3_data.size * 100

    if q2_nan_pct > 50 and q3_nan_pct < 20:
        return q3_data
    elif q3_nan_pct > 50 and q2_nan_pct < 20:
        return q2_data
    else:
        return (q2_data + q3_data) / 2.0


def get_rgb_from_composite(composite: np.ndarray) -> np.ndarray:
    """Get RGB image with percentile stretch."""
    rgb = np.stack([composite[2], composite[1], composite[0]], axis=-1)
    rgb = np.nan_to_num(rgb, nan=0)
    for c in range(3):
        p2, p98 = np.percentile(rgb[:, :, c], [2, 98])
        if p98 > p2:
            rgb[:, :, c] = np.clip((rgb[:, :, c] - p2) / (p98 - p2), 0, 1)
        else:
            rgb[:, :, c] = 0
    return rgb


def compute_sample_metrics(pred_prob: np.ndarray, mask: np.ndarray, threshold: float = 0.5):
    """Compute metrics for a single sample."""
    pred_binary = (pred_prob > threshold).astype(float)
    mask_binary = (mask > 0).astype(float)

    tp = ((pred_binary == 1) & (mask_binary == 1)).sum()
    fp = ((pred_binary == 1) & (mask_binary == 0)).sum()
    fn = ((pred_binary == 0) & (mask_binary == 1)).sum()

    union = tp + fp + fn
    iou = tp / union if union > 0 else 1.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {'iou': iou, 'f1': f1, 'precision': precision, 'recall': recall}


def create_tpfpfn_overlay(pred_prob: np.ndarray, mask: np.ndarray,
                          rgb: np.ndarray, threshold: float = 0.5,
                          alpha: float = 0.5) -> np.ndarray:
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


def get_cv_splits(all_refids: list, change_levels: list, n_folds: int = 5):
    """Get CV fold assignments (same as training)."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    folds = {}
    for fold, (train_idx, val_idx) in enumerate(skf.split(all_refids, change_levels)):
        folds[fold] = {
            'train': [all_refids[i] for i in train_idx],
            'val': [all_refids[i] for i in val_idx],
        }
    return folds


def predict_ensemble(refid: str, condition: str, folds: dict,
                     model_cache: ModelCache, norm_stats_cache: NormStatsCache,
                     device: torch.device) -> np.ndarray:
    """Get ensemble prediction (average of 5 fold models)."""
    import albumentations as A

    exp_config = EXPERIMENTS[condition]
    probs = []

    for fold in range(5):
        train_refids = folds[fold]['train']
        norm_stats = norm_stats_cache.get_stats(condition, fold, train_refids)

        # Create center crop transform (same as validation/test)
        transform = A.Compose([
            A.CenterCrop(64, 64),
        ])

        dataset = MultiTemporalSentinel2Dataset(
            refids=[refid],
            temporal_sampling=exp_config['temporal_sampling'],
            normalization_stats=norm_stats,
            transform=transform,
        )

        sample = dataset[0]
        x = sample['image'].unsqueeze(0).to(device)

        model = model_cache.get_model(condition, fold, device)
        with torch.no_grad():
            logits = model(x)
            prob = torch.sigmoid(logits).squeeze().cpu().numpy()

        probs.append(prob)

    return np.mean(probs, axis=0)


def plot_test_tile_supplement(refid: str, raw_data: np.ndarray, predictions: dict,
                               mask: np.ndarray, metrics: dict, output_dir: Path):
    """
    Plot detailed cross-mode comparison for supplement (one per test tile).

    Includes probability maps with colorblind-safe colormap (viridis).
    """
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(4, 4, height_ratios=[1, 1, 1, 1], hspace=0.25, wspace=0.15)

    # Get annual composites
    composite_2018 = compute_annual_composite(raw_data, year_idx=0)
    composite_2024 = compute_annual_composite(raw_data, year_idx=6)
    rgb_2018 = get_rgb_from_composite(composite_2018)
    rgb_2024 = get_rgb_from_composite(composite_2024)

    # Center crop
    H, W = rgb_2018.shape[:2]
    start_h = (H - 64) // 2
    start_w = (W - 64) // 2
    rgb_2018 = rgb_2018[start_h:start_h+64, start_w:start_w+64]
    rgb_2024 = rgb_2024[start_h:start_h+64, start_w:start_w+64]

    # Header row: 2018, 2024, GT, legend
    ax_2018 = fig.add_subplot(gs[0, 0])
    ax_2018.imshow(rgb_2018)
    ax_2018.set_title('2018 Composite', fontsize=10)
    ax_2018.axis('off')

    ax_2024 = fig.add_subplot(gs[0, 1])
    ax_2024.imshow(rgb_2024)
    ax_2024.set_title('2024 Composite', fontsize=10)
    ax_2024.axis('off')

    ax_gt = fig.add_subplot(gs[0, 2])
    ax_gt.imshow(mask, cmap='gray', vmin=0, vmax=1)
    ax_gt.set_title('Ground Truth', fontsize=10)
    ax_gt.axis('off')

    # Legend
    ax_legend = fig.add_subplot(gs[0, 3])
    ax_legend.axis('off')
    patches = [
        mpatches.Patch(color=COLORS['TP'], label='TP'),
        mpatches.Patch(color=COLORS['FP'], label='FP'),
        mpatches.Patch(color=COLORS['FN'], label='FN'),
    ]
    ax_legend.legend(handles=patches, loc='center', fontsize=10)
    ax_legend.text(0.5, 0.15, f'RefID: {refid[:25]}...', transform=ax_legend.transAxes,
                  ha='center', fontsize=8)

    # Mode rows
    modes = ['bi_temporal', 'annual', 'bi_seasonal']
    mode_labels = ['T=2 (Bi-temporal)', 'T=7 (Annual)', 'T=14 (Bi-seasonal)']

    for row, (mode, label) in enumerate(zip(modes, mode_labels), start=1):
        pred_prob = predictions[mode]
        m = metrics[mode]

        # Probability map (viridis - colorblind safe)
        ax_prob = fig.add_subplot(gs[row, 0])
        ax_prob.imshow(pred_prob, cmap='viridis', vmin=0, vmax=1)
        ax_prob.set_ylabel(label, fontsize=10, fontweight='bold')
        ax_prob.set_title('Probability' if row == 1 else '', fontsize=9)
        ax_prob.axis('off')

        # Prediction
        ax_pred = fig.add_subplot(gs[row, 1])
        ax_pred.imshow(pred_prob > 0.5, cmap='gray', vmin=0, vmax=1)
        ax_pred.set_title('Prediction' if row == 1 else '', fontsize=9)
        ax_pred.axis('off')

        # TP/FP/FN overlay
        overlay = create_tpfpfn_overlay(pred_prob, mask, rgb_2024)
        ax_overlay = fig.add_subplot(gs[row, 2])
        ax_overlay.imshow(overlay)
        ax_overlay.set_title('TP/FP/FN Overlay' if row == 1 else '', fontsize=9)
        ax_overlay.axis('off')

        # Metrics
        ax_metrics = fig.add_subplot(gs[row, 3])
        ax_metrics.axis('off')
        metrics_text = (f"IoU:  {m['iou']*100:.1f}%\n"
                       f"F1:   {m['f1']*100:.1f}%\n"
                       f"Prec: {m['precision']*100:.1f}%\n"
                       f"Rec:  {m['recall']*100:.1f}%")
        ax_metrics.text(0.5, 0.5, metrics_text, transform=ax_metrics.transAxes,
                       ha='center', va='center', fontsize=10, family='monospace')

    fig.suptitle(f'TEST TILE: {refid[:40]}...\n(CV ensemble, threshold=0.5, 64×64 center crop)',
                 fontsize=11, fontweight='bold', y=0.99)

    # Save
    safe_refid = refid.replace('/', '_').replace('\\', '_')[:50]
    output_path = output_dir / f"test_tile_{safe_refid}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return output_path


def plot_main_summary_figure(selected_tiles: list, all_data: dict, output_dir: Path):
    """
    Create clean summary figure for main paper.

    Shows 2-3 tiles with: 2018 RGB, 2024 RGB, GT, and TP/FP/FN overlay + IoU per mode.
    Minimal design - no titles, legends, or annotations (those go in caption).
    """
    n_tiles = len(selected_tiles)

    # Figure layout: n_tiles rows, 6 columns (2018, 2024, GT, T=2 overlay, T=7 overlay, T=14 overlay)
    fig, axes = plt.subplots(n_tiles, 6, figsize=(14, 3 * n_tiles))
    if n_tiles == 1:
        axes = axes.reshape(1, -1)

    col_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    last_row = n_tiles - 1

    for row, tile_info in enumerate(selected_tiles):
        refid = tile_info['refid']
        criterion = tile_info['criterion']
        data = all_data[refid]

        rgb_2018 = data['rgb_2018']
        rgb_2024 = data['rgb_2024']
        mask = data['mask']
        predictions = data['predictions']
        metrics = data['metrics']

        # Build row label with criterion and IoUs
        iou_2 = metrics['bi_temporal']['iou'] * 100
        iou_7 = metrics['annual']['iou'] * 100
        iou_14 = metrics['bi_seasonal']['iou'] * 100
        row_label = f"{criterion}\n({iou_2:.0f}% / {iou_7:.0f}% / {iou_14:.0f}%)"

        # Column 0: 2018 RGB
        axes[row, 0].imshow(rgb_2018)
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])
        axes[row, 0].set_ylabel(row_label, fontsize=9, fontweight='bold')
        if row == last_row:
            axes[row, 0].text(0.5, -0.1, col_labels[0], transform=axes[row, 0].transAxes,
                             fontsize=10, ha='center')

        # Column 1: 2024 RGB
        axes[row, 1].imshow(rgb_2024)
        axes[row, 1].set_xticks([])
        axes[row, 1].set_yticks([])
        if row == last_row:
            axes[row, 1].text(0.5, -0.1, col_labels[1], transform=axes[row, 1].transAxes,
                             fontsize=10, ha='center')

        # Column 2: Ground Truth
        axes[row, 2].imshow(mask, cmap='gray', vmin=0, vmax=1)
        axes[row, 2].set_xticks([])
        axes[row, 2].set_yticks([])
        if row == last_row:
            axes[row, 2].text(0.5, -0.1, col_labels[2], transform=axes[row, 2].transAxes,
                             fontsize=10, ha='center')

        # Columns 3-5: TP/FP/FN overlays
        for col, mode in enumerate(['bi_temporal', 'annual', 'bi_seasonal'], start=3):
            pred_prob = predictions[mode]
            overlay = create_tpfpfn_overlay(pred_prob, mask, rgb_2024)
            axes[row, col].imshow(overlay)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if row == last_row:
                axes[row, col].text(0.5, -0.1, col_labels[col], transform=axes[row, col].transAxes,
                                   fontsize=10, ha='center')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)

    output_path = output_dir / "test_summary_main.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')

    output_pdf = output_dir / "test_summary_main.pdf"
    plt.savefig(output_pdf, bbox_inches='tight', facecolor='white')

    plt.close()

    return output_path


def select_tiles_for_main_figure(all_results: list):
    """
    Select 2-3 tiles by predefined criteria (not visual appearance).

    Criteria:
    1. Median IoU under Annual (typical case)
    2. Largest Annual - Bi-seasonal improvement (tail driver)
    3. Failure case (IoU = 0 for all modes) if exists
    """
    selected = []

    # Convert to arrays for easier indexing
    annual_ious = np.array([r['annual_iou'] for r in all_results])
    bi_seasonal_ious = np.array([r['bi_seasonal_iou'] for r in all_results])
    bi_temporal_ious = np.array([r['bi_temporal_iou'] for r in all_results])

    # 1. Median IoU under Annual (typical case)
    median_iou = np.median(annual_ious)
    median_idx = np.argmin(np.abs(annual_ious - median_iou))
    selected.append({
        'refid': all_results[median_idx]['refid'],
        'criterion': 'Typical\n(median IoU)',
        'idx': median_idx
    })
    print(f"  Median case: {all_results[median_idx]['refid'][:30]}... "
          f"(Annual IoU={annual_ious[median_idx]*100:.1f}%)")

    # 2. Largest Annual - Bi-seasonal improvement (tail driver)
    delta = annual_ious - bi_seasonal_ious
    # Exclude already selected
    delta_masked = delta.copy()
    delta_masked[median_idx] = -np.inf
    max_delta_idx = np.argmax(delta_masked)
    selected.append({
        'refid': all_results[max_delta_idx]['refid'],
        'criterion': 'Tail driver\n(max Δ)',
        'idx': max_delta_idx
    })
    print(f"  Tail driver: {all_results[max_delta_idx]['refid'][:30]}... "
          f"(Δ Annual−Bi-seas={delta[max_delta_idx]*100:+.1f}pp)")

    # 3. Failure case (IoU = 0 for all modes) if exists
    for i, r in enumerate(all_results):
        if i not in [median_idx, max_delta_idx]:
            if r['annual_iou'] == 0 and r['bi_temporal_iou'] == 0 and r['bi_seasonal_iou'] == 0:
                selected.append({
                    'refid': r['refid'],
                    'criterion': 'Failure\n(all modes)',
                    'idx': i
                })
                print(f"  Failure case: {r['refid'][:30]}... (all IoU=0%)")
                break

    return selected


def main():
    print("=" * 70)
    print("TEST SET QUALITATIVE ANALYSIS")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load sample info
    sample_info_path = parent_dir / "sample_change_levels.csv"
    sample_info = pd.read_csv(sample_info_path)

    # Load test/train split from files
    splits_dir = parent_dir.parent / "outputs" / "splits"
    test_refids = [line.strip() for line in open(splits_dir / "test_refids.txt")]
    cv_refids = [line.strip() for line in open(splits_dir / "train_refids.txt")]

    # Get change levels for CV samples
    refid_to_level = dict(zip(sample_info['refid'], sample_info['change_level']))
    cv_change_levels = [refid_to_level[r] for r in cv_refids]

    print(f"\nCV samples: {len(cv_refids)}")
    print(f"Test samples: {len(test_refids)}")

    # Get CV folds (needed for normalization stats)
    folds = get_cv_splits(cv_refids, cv_change_levels)

    # Initialize caches
    model_cache = ModelCache()
    norm_stats_cache = NormStatsCache()

    # Output directories
    supplement_dir = MT_EXPERIMENTS_DIR / "outputs" / "analysis" / "test_qualitative"
    supplement_dir.mkdir(parents=True, exist_ok=True)

    main_dir = MT_EXPERIMENTS_DIR / "outputs" / "analysis"

    # Process each test tile and store data
    all_results = []
    all_data = {}  # Store data for main figure

    for refid in tqdm(test_refids, desc="Processing test tiles"):
        print(f"\n  Processing: {refid[:40]}...")

        # Load raw data and mask
        raw_data = load_raw_sentinel2(refid)
        H, W = raw_data.shape[2], raw_data.shape[3]
        start_h = (H - 64) // 2
        start_w = (W - 64) // 2
        mask = load_mask(refid, target_shape=(H, W))
        mask = mask[start_h:start_h+64, start_w:start_w+64]

        # Get RGB images
        composite_2018 = compute_annual_composite(raw_data, year_idx=0)
        composite_2024 = compute_annual_composite(raw_data, year_idx=6)
        rgb_2018 = get_rgb_from_composite(composite_2018)
        rgb_2024 = get_rgb_from_composite(composite_2024)
        rgb_2018 = rgb_2018[start_h:start_h+64, start_w:start_w+64]
        rgb_2024 = rgb_2024[start_h:start_h+64, start_w:start_w+64]

        # Get predictions for all modes
        predictions = {}
        metrics = {}

        for condition in EXPERIMENTS.keys():
            pred_prob = predict_ensemble(refid, condition, folds,
                                         model_cache, norm_stats_cache, device)
            predictions[condition] = pred_prob
            metrics[condition] = compute_sample_metrics(pred_prob, mask)

        # Store data for main figure
        all_data[refid] = {
            'rgb_2018': rgb_2018,
            'rgb_2024': rgb_2024,
            'mask': mask,
            'predictions': predictions,
            'metrics': metrics,
        }

        # Plot supplement figure (detailed, one per tile)
        output_path = plot_test_tile_supplement(refid, raw_data, predictions, mask, metrics, supplement_dir)
        print(f"    Saved supplement: {output_path.name}")

        # Store results
        all_results.append({
            'refid': refid,
            'annual_iou': metrics['annual']['iou'],
            'bi_temporal_iou': metrics['bi_temporal']['iou'],
            'bi_seasonal_iou': metrics['bi_seasonal']['iou'],
        })

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SET SUMMARY")
    print("=" * 70)
    print(f"{'RefID':<45} {'Annual':>8} {'Bi-temp':>8} {'Bi-seas':>8}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['refid'][:44]:<45} {r['annual_iou']*100:>7.1f}% {r['bi_temporal_iou']*100:>7.1f}% {r['bi_seasonal_iou']*100:>7.1f}%")

    # Select tiles for main figure by predefined criteria
    print("\n" + "=" * 70)
    print("SELECTING TILES FOR MAIN FIGURE")
    print("=" * 70)
    selected_tiles = select_tiles_for_main_figure(all_results)

    # Generate main summary figure
    print("\nGenerating main summary figure...")
    main_path = plot_main_summary_figure(selected_tiles, all_data, main_dir)
    print(f"✓ Main figure saved to: {main_path}")

    print(f"\n✓ Supplement figures saved to: {supplement_dir}")


if __name__ == "__main__":
    main()
