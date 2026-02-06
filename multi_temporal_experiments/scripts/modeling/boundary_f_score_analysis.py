#!/usr/bin/env python3
"""
Boundary F-score (BF) analysis for land-take detection experiments.

Computes boundary quality metrics to complement IoU (region accuracy).
Evaluates how well model predictions delineate change boundaries.

Metrics:
- Boundary Precision (BP): predicted boundary pixels within d of GT boundary
- Boundary Recall (BR): GT boundary pixels within d of predicted boundary
- Boundary F-score (BF): harmonic mean of BP and BR

Usage:
    python boundary_f_score_analysis.py --tolerance 2
"""

import sys
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy import ndimage
from scipy import stats
import warnings

# Add paths for imports
script_dir = Path(__file__).resolve().parent
parent_dir = script_dir.parent.parent
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(parent_dir.parent))

# Import multi-temporal modules
from models_multitemporal import create_multitemporal_model
from multi_temporal_experiments.scripts.data_preparation.dataset_multitemporal import get_dataloaders
from multi_temporal_experiments.config import MT_EXPERIMENTS_DIR


# Experiment configurations
EXPERIMENTS = {
    'annual': {
        'name': 'exp010_lstm7_no_es',
        'temporal_sampling': 'annual',
        'convlstm_kernel_size': 3,
        'description': 'Annual (T=7)',
    },
    'bi_seasonal': {
        'name': 'exp002_v3',
        'temporal_sampling': 'quarterly',
        'convlstm_kernel_size': 3,
        'description': 'Bi-seasonal (T=14)',
    },
    'bi_temporal': {
        'name': 'exp003_v3',
        'temporal_sampling': 'bi_temporal',
        'convlstm_kernel_size': 3,
        'description': 'Bi-temporal (T=2)',
    },
    'k1x1': {
        'name': 'exp004_v2',
        'temporal_sampling': 'annual',
        'convlstm_kernel_size': 1,
        'description': '1×1 kernel (T=7)',
    },
    'early_fusion': {
        'name': 'exp005_early_fusion',
        'temporal_sampling': 'bi_temporal',
        'model_name': 'early_fusion_unet',
        'description': 'Early-Fusion U-Net (T=2, stacked)',
    },
    'late_fusion': {
        'name': 'exp006_late_fusion',
        'temporal_sampling': 'bi_temporal',
        'model_name': 'late_fusion_concat',
        'description': 'Late-Fusion Concat (T=2)',
    },
    'late_fusion_pool': {
        'name': 'exp007_late_fusion_pool',
        'temporal_sampling': 'annual',
        'model_name': 'late_fusion_pool',
        'description': 'Late-Fusion Pool (T=7)',
    },
    'conv3d_fusion': {
        'name': 'exp008_conv3d_fusion',
        'temporal_sampling': 'annual',
        'model_name': 'conv3d_fusion',
        'description': '3D Conv Fusion (T=7)',
    },
    'lstm_lite': {
        'name': 'exp009_lstm_lite',
        'temporal_sampling': 'bi_temporal',
        'convlstm_kernel_size': 3,
        'description': 'ConvLSTM-lite (T=2, 1-layer, h=32)',
    },
    'lstm7_no_es': {
        'name': 'exp010_lstm7_no_es',
        'temporal_sampling': 'annual',
        'convlstm_kernel_size': 3,
        'description': 'LSTM-7 no ES (T=7, full 400 epochs)',
    },
    'lstm7_lite': {
        'name': 'exp011_lstm7_lite',
        'temporal_sampling': 'annual',
        'convlstm_kernel_size': 3,
        'description': 'LSTM-7-lite (T=7, 1-layer, h=32)',
    },
}


def extract_boundary(mask: np.ndarray) -> np.ndarray:
    """
    Extract boundary pixels from a binary mask.

    Uses morphological erosion: boundary = mask XOR erode(mask)

    Args:
        mask: Binary mask (H, W) with values 0 or 1

    Returns:
        boundary: Binary boundary map (H, W)
    """
    # 3x3 structuring element for erosion
    struct = ndimage.generate_binary_structure(2, 1)

    # Erode the mask
    eroded = ndimage.binary_erosion(mask, structure=struct)

    # Boundary = mask XOR eroded mask
    boundary = np.logical_xor(mask, eroded).astype(np.float32)

    return boundary


def compute_boundary_f_score(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    tolerance: int = 2,
    epsilon: float = 1e-7
) -> dict:
    """
    Compute Boundary F-score between predicted and ground-truth masks.

    Args:
        pred_mask: Predicted binary mask (H, W)
        gt_mask: Ground-truth binary mask (H, W)
        tolerance: Distance tolerance in pixels (d)
        epsilon: Small constant to avoid division by zero

    Returns:
        dict with boundary_precision, boundary_recall, boundary_f_score
    """
    # Extract boundaries
    pred_boundary = extract_boundary(pred_mask)
    gt_boundary = extract_boundary(gt_mask)

    # Count boundary pixels
    n_pred_boundary = np.sum(pred_boundary)
    n_gt_boundary = np.sum(gt_boundary)

    # Handle edge cases
    # Case A: No GT boundary and no predicted boundary -> perfect match
    if n_gt_boundary == 0 and n_pred_boundary == 0:
        return {
            'boundary_precision': 1.0,
            'boundary_recall': 1.0,
            'boundary_f_score': 1.0,
            'n_pred_boundary': 0,
            'n_gt_boundary': 0,
            'case': 'A_both_empty',
        }

    # Case B: No GT boundary but predicted boundary exists -> false positives
    if n_gt_boundary == 0 and n_pred_boundary > 0:
        return {
            'boundary_precision': 0.0,
            'boundary_recall': 1.0,  # Vacuously true (no GT to miss)
            'boundary_f_score': 0.0,
            'n_pred_boundary': int(n_pred_boundary),
            'n_gt_boundary': 0,
            'case': 'B_false_boundary',
        }

    # Case C: GT boundary exists but no predicted boundary -> missed boundary
    if n_gt_boundary > 0 and n_pred_boundary == 0:
        return {
            'boundary_precision': 1.0,  # Vacuously true (no pred to be wrong)
            'boundary_recall': 0.0,
            'boundary_f_score': 0.0,
            'n_pred_boundary': 0,
            'n_gt_boundary': int(n_gt_boundary),
            'case': 'C_missed_boundary',
        }

    # Normal case: Both have boundaries
    # Compute distance transforms
    # Distance from each pixel to nearest GT boundary pixel
    gt_boundary_bool = gt_boundary.astype(bool)
    pred_boundary_bool = pred_boundary.astype(bool)

    # Distance transform: distance to nearest True pixel
    # We need distance to boundary, so invert: distance to nearest boundary pixel
    dist_to_gt = ndimage.distance_transform_edt(~gt_boundary_bool)
    dist_to_pred = ndimage.distance_transform_edt(~pred_boundary_bool)

    # Boundary precision: fraction of predicted boundary pixels within tolerance of GT
    pred_boundary_pixels = pred_boundary_bool
    dist_at_pred = dist_to_gt[pred_boundary_pixels]
    bp = np.sum(dist_at_pred <= tolerance) / (n_pred_boundary + epsilon)

    # Boundary recall: fraction of GT boundary pixels within tolerance of predicted
    gt_boundary_pixels = gt_boundary_bool
    dist_at_gt = dist_to_pred[gt_boundary_pixels]
    br = np.sum(dist_at_gt <= tolerance) / (n_gt_boundary + epsilon)

    # Boundary F-score
    bf = (2 * bp * br) / (bp + br + epsilon)

    return {
        'boundary_precision': float(bp),
        'boundary_recall': float(br),
        'boundary_f_score': float(bf),
        'n_pred_boundary': int(n_pred_boundary),
        'n_gt_boundary': int(n_gt_boundary),
        'case': 'D_normal',
    }


def compute_sample_iou(pred_mask: np.ndarray, gt_mask: np.ndarray, epsilon: float = 1e-7) -> float:
    """Compute IoU for a single sample."""
    tp = np.sum((pred_mask == 1) & (gt_mask == 1))
    fp = np.sum((pred_mask == 1) & (gt_mask == 0))
    fn = np.sum((pred_mask == 0) & (gt_mask == 1))

    union = tp + fp + fn
    if union == 0:
        return 1.0

    return float(tp / (union + epsilon))


def load_model_from_checkpoint(checkpoint_path: Path, config: dict, device: torch.device):
    """Load a trained model from checkpoint."""
    model = create_multitemporal_model(
        config.get('model_name', 'lstm_unet'),
        encoder_name=config.get('encoder_name', 'resnet50'),
        encoder_weights=None,
        in_channels=9,
        classes=1,
        lstm_hidden_dim=config.get('lstm_hidden_dim', 256),
        lstm_num_layers=config.get('lstm_num_layers', 2),
        convlstm_kernel_size=config.get('convlstm_kernel_size', 3),
        skip_aggregation=config.get('skip_aggregation', 'max'),
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    return model


def get_predictions_and_metrics(
    experiment_config: dict,
    num_folds: int,
    device: torch.device,
    threshold: float = 0.5,
    tolerance: int = 2
):
    """
    Get per-sample IoU and BF from out-of-fold predictions.
    """
    exp_name = experiment_config['name']
    temporal_sampling = experiment_config['temporal_sampling']

    per_sample_metrics = {}

    for fold in range(num_folds):
        exp_dir = MT_EXPERIMENTS_DIR / f"{exp_name}_fold{fold}"

        if not exp_dir.exists():
            print(f"  WARNING: {exp_dir} not found, skipping fold {fold}")
            continue

        # Load config
        config_path = exp_dir / "config.json"
        with open(config_path) as f:
            config = json.load(f)

        # Load model
        checkpoint_path = exp_dir / "best_model.pth"
        if not checkpoint_path.exists():
            print(f"  WARNING: {checkpoint_path} not found, skipping fold {fold}")
            continue

        model = load_model_from_checkpoint(checkpoint_path, config, device)

        # Create dataloader for this fold's validation set
        dataloaders = get_dataloaders(
            temporal_sampling=temporal_sampling,
            batch_size=1,
            num_workers=4,
            image_size=config['image_size'],
            output_format="LSTM",
            fold=fold,
            num_folds=num_folds,
            seed=config.get('seed', 42),
        )
        val_loader = dataloaders['val']

        # Evaluate on validation set
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                masks = batch['mask']
                refids = batch['refid']

                outputs = model(images)

                for i, refid in enumerate(refids):
                    # Get prediction and ground truth
                    pred_logits = outputs[i].squeeze().cpu().numpy()
                    gt_mask = masks[i].squeeze().cpu().numpy()

                    # Convert to binary
                    pred_prob = 1 / (1 + np.exp(-pred_logits))  # sigmoid
                    pred_mask = (pred_prob >= threshold).astype(np.float32)
                    gt_mask = gt_mask.astype(np.float32)

                    # Compute IoU
                    iou = compute_sample_iou(pred_mask, gt_mask)

                    # Compute BF
                    bf_metrics = compute_boundary_f_score(pred_mask, gt_mask, tolerance=tolerance)

                    per_sample_metrics[refid] = {
                        'iou': iou,
                        'boundary_precision': bf_metrics['boundary_precision'],
                        'boundary_recall': bf_metrics['boundary_recall'],
                        'boundary_f_score': bf_metrics['boundary_f_score'],
                        'n_pred_boundary': bf_metrics['n_pred_boundary'],
                        'n_gt_boundary': bf_metrics['n_gt_boundary'],
                        'case': bf_metrics['case'],
                    }

    return per_sample_metrics


def permutation_test(differences: np.ndarray, n_permutations: int = 10000, seed: int = 42):
    """Perform permutation test on paired differences."""
    rng = np.random.RandomState(seed)
    observed_mean = np.mean(differences)
    n = len(differences)

    permuted_means = np.zeros(n_permutations)
    for i in range(n_permutations):
        signs = rng.choice([-1, 1], size=n)
        permuted_diff = differences * signs
        permuted_means[i] = np.mean(permuted_diff)

    p_value = np.mean(np.abs(permuted_means) >= np.abs(observed_mean))
    return observed_mean, p_value


def bootstrap_ci(differences: np.ndarray, n_bootstrap: int = 10000, ci: float = 0.95, seed: int = 42):
    """Compute bootstrap confidence interval for mean difference."""
    rng = np.random.RandomState(seed)
    n = len(differences)

    bootstrap_means = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        indices = rng.choice(n, size=n, replace=True)
        bootstrap_sample = differences[indices]
        bootstrap_means[i] = np.mean(bootstrap_sample)

    alpha = 1 - ci
    lower = np.percentile(bootstrap_means, alpha / 2 * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)
    return lower, upper


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Cliff's delta effect size for paired data.

    delta = (# pairs where x > y - # pairs where x < y) / n

    Interpretation (Romano et al., 2006):
    - |delta| < 0.147: negligible
    - 0.147 <= |delta| < 0.33: small
    - 0.33 <= |delta| < 0.474: medium
    - |delta| >= 0.474: large

    Args:
        x: First condition values
        y: Second condition values

    Returns:
        float: Cliff's delta
    """
    differences = x - y
    n = len(differences)
    n_greater = np.sum(differences > 0)
    n_less = np.sum(differences < 0)
    return (n_greater - n_less) / n


def interpret_cliffs_delta(delta: float) -> str:
    """Interpret Cliff's delta magnitude."""
    abs_delta = abs(delta)
    if abs_delta < 0.147:
        return "negligible"
    elif abs_delta < 0.33:
        return "small"
    elif abs_delta < 0.474:
        return "medium"
    else:
        return "large"


def holm_bonferroni_correction(p_values: list) -> list:
    """
    Apply Holm-Bonferroni correction for multiple comparisons.

    Identical logic to statistical_analysis_persample.holm_bonferroni_correction.

    Args:
        p_values: List of raw p-values

    Returns:
        list: Adjusted p-values
    """
    n = len(p_values)

    # Sort p-values and track original indices
    indexed_pvalues = [(p, i) for i, p in enumerate(p_values)]
    indexed_pvalues.sort(key=lambda x: x[0])

    adjusted = [None] * n
    cumulative_min = 1.0

    # Process from largest to smallest (reverse order for Holm)
    for rank in range(n, 0, -1):
        p, orig_idx = indexed_pvalues[rank - 1]
        # Multiply by (n - rank + 1)
        adjusted_p = min(p * (n - rank + 1), 1.0)
        # Ensure monotonicity
        cumulative_min = min(cumulative_min, adjusted_p)
        adjusted[orig_idx] = cumulative_min

    return adjusted


def main():
    parser = argparse.ArgumentParser(description="Boundary F-score analysis")
    parser.add_argument('--tolerance', type=int, default=2,
                        help='Boundary tolerance in pixels (default: 2 = 20m)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Binary threshold for predictions')
    parser.add_argument('--num-folds', type=int, default=5)
    parser.add_argument('--n-permutations', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    print("\n" + "="*80)
    print(f"BOUNDARY F-SCORE ANALYSIS (tolerance d={args.tolerance} pixels = {args.tolerance*10}m)")
    print("="*80)

    # Collect metrics for all experiments
    all_metrics = {}

    for exp_key, exp_config in EXPERIMENTS.items():
        print(f"\n--- {exp_config['description']} ({exp_config['name']}) ---")

        metrics = get_predictions_and_metrics(
            exp_config, args.num_folds, device,
            threshold=args.threshold, tolerance=args.tolerance
        )
        all_metrics[exp_key] = metrics

        if len(metrics) > 0:
            ious = [m['iou'] for m in metrics.values()]
            bfs = [m['boundary_f_score'] for m in metrics.values()]
            print(f"  Samples: {len(metrics)}")
            print(f"  Mean IoU: {np.mean(ious)*100:.2f}%")
            print(f"  Mean BF@{args.tolerance}: {np.mean(bfs)*100:.2f}%")
        else:
            print(f"  No samples collected!")

    # Verify sample alignment
    refids = set(all_metrics['annual'].keys())
    for exp_key in all_metrics:
        if set(all_metrics[exp_key].keys()) != refids:
            raise ValueError(
                f"Sample mismatch for {exp_key}: has {len(all_metrics[exp_key])} samples "
                f"vs {len(refids)} in reference (annual). "
                f"Missing: {refids - set(all_metrics[exp_key].keys())}, "
                f"Extra: {set(all_metrics[exp_key].keys()) - refids}"
            )

    n_samples = len(refids)
    refid_list = sorted(refids)
    print(f"\n✓ All experiments have {n_samples} matching samples")

    # Create aligned arrays
    def get_aligned_array(exp_key, metric):
        return np.array([all_metrics[exp_key][r][metric] for r in refid_list])

    # ==========================================================================
    # TEMPORAL SAMPLING COMPARISONS (IoU + BF)
    # ==========================================================================
    print("\n" + "="*80)
    print("TEMPORAL SAMPLING COMPARISONS")
    print("="*80)

    temporal_comparisons = [
        ('Annual vs Bi-temporal', 'annual', 'bi_temporal'),
        ('Annual vs Bi-seasonal', 'annual', 'bi_seasonal'),
    ]

    temporal_results = []

    for name, cond1, cond2 in temporal_comparisons:
        print(f"\n--- {name} ---")

        for metric in ['iou', 'boundary_f_score']:
            metric_label = 'IoU' if metric == 'iou' else f'BF@{args.tolerance}'

            arr1 = get_aligned_array(cond1, metric)
            arr2 = get_aligned_array(cond2, metric)
            diff = arr1 - arr2

            mean_diff = np.mean(diff)
            _, p_value = permutation_test(diff, n_permutations=args.n_permutations, seed=args.seed)
            ci_lower, ci_upper = bootstrap_ci(diff, seed=args.seed)

            delta = cliffs_delta(arr1, arr2)
            delta_interp = interpret_cliffs_delta(delta)

            sig = "*" if p_value < 0.05 else ""
            print(f"  {metric_label}: Δ = {mean_diff*100:+.2f} pp, p = {p_value:.3f}{sig}, "
                  f"95% CI [{ci_lower*100:+.2f}, {ci_upper*100:+.2f}], "
                  f"Cliff's δ = {delta:+.3f} ({delta_interp})")

            temporal_results.append({
                'comparison': name,
                'metric': metric_label,
                'mean_diff_pp': mean_diff * 100,
                'p_value': p_value,
                'ci_lower_pp': ci_lower * 100,
                'ci_upper_pp': ci_upper * 100,
                'cliffs_delta': delta,
                'cliffs_delta_interp': delta_interp,
            })

    # ==========================================================================
    # 1D vs 2D COMPARISON (IoU + BF)
    # ==========================================================================
    print("\n" + "="*80)
    print("1D vs 2D TEMPORAL MODELING (RQ2c)")
    print("="*80)
    print("\n--- 3×3 (patch-based) vs 1×1 (per-pixel) ---")

    kernel_results = []

    for metric in ['iou', 'boundary_f_score']:
        metric_label = 'IoU' if metric == 'iou' else f'BF@{args.tolerance}'

        arr_3x3 = get_aligned_array('annual', metric)  # exp001_v2 uses 3x3
        arr_1x1 = get_aligned_array('k1x1', metric)    # exp004_1x1 uses 1x1
        diff = arr_3x3 - arr_1x1

        mean_diff = np.mean(diff)
        _, p_value = permutation_test(diff, n_permutations=args.n_permutations, seed=args.seed)
        ci_lower, ci_upper = bootstrap_ci(diff, seed=args.seed)

        n_3x3_wins = np.sum(diff > 0)
        prob_3x3_better = n_3x3_wins / n_samples

        delta = cliffs_delta(arr_3x3, arr_1x1)
        delta_interp = interpret_cliffs_delta(delta)

        sig = "*" if p_value < 0.05 else ""
        print(f"  {metric_label}: Δ = {mean_diff*100:+.2f} pp, p = {p_value:.3f}{sig}, "
              f"95% CI [{ci_lower*100:+.2f}, {ci_upper*100:+.2f}], "
              f"3×3 wins {prob_3x3_better*100:.1f}%, "
              f"Cliff's δ = {delta:+.3f} ({delta_interp})")

        kernel_results.append({
            'metric': metric_label,
            'mean_3x3': float(np.mean(arr_3x3)),
            'mean_1x1': float(np.mean(arr_1x1)),
            'mean_diff_pp': mean_diff * 100,
            'p_value': p_value,
            'ci_lower_pp': ci_lower * 100,
            'ci_upper_pp': ci_upper * 100,
            'prob_3x3_better': prob_3x3_better,
            'cliffs_delta': delta,
            'cliffs_delta_interp': delta_interp,
        })

    # ==========================================================================
    # BASELINE COMPARISONS (RQ0: No-temporal baselines)
    # ==========================================================================
    print("\n" + "="*80)
    print("BASELINE COMPARISONS (RQ0: No-Temporal Baselines)")
    print("="*80)

    baseline_comparisons = [
        ('Annual (LSTM) vs Early-Fusion', 'annual', 'early_fusion'),
        ('Annual (LSTM) vs Late-Fusion', 'annual', 'late_fusion'),
        ('Bi-temporal (LSTM) vs Early-Fusion', 'bi_temporal', 'early_fusion'),
        ('Bi-temporal (LSTM) vs Late-Fusion', 'bi_temporal', 'late_fusion'),
        ('Late-Fusion vs Early-Fusion', 'late_fusion', 'early_fusion'),
        ('Annual (LSTM) vs Late-Fusion Pool', 'annual', 'late_fusion_pool'),
        ('Annual (LSTM) vs Conv3D Fusion', 'annual', 'conv3d_fusion'),
        ('Bi-temporal (LSTM) vs LSTM-lite', 'bi_temporal', 'lstm_lite'),
        ('LSTM-lite vs Early-Fusion', 'lstm_lite', 'early_fusion'),
        ('LSTM-lite vs Late-Fusion Concat', 'lstm_lite', 'late_fusion'),
        ('Annual LSTM (no ES) vs Pool-7', 'lstm7_no_es', 'late_fusion_pool'),
        ('Annual LSTM (no ES) vs Conv3D-7', 'lstm7_no_es', 'conv3d_fusion'),
        ('Annual LSTM (no ES) vs LSTM-7-lite', 'lstm7_no_es', 'lstm7_lite'),
        ('LSTM-7-lite vs Pool-7', 'lstm7_lite', 'late_fusion_pool'),
        ('LSTM-7-lite vs Conv3D-7', 'lstm7_lite', 'conv3d_fusion'),
    ]

    baseline_results = []

    for name, cond1, cond2 in baseline_comparisons:
        print(f"\n--- {name} ---")

        for metric in ['iou', 'boundary_f_score']:
            metric_label = 'IoU' if metric == 'iou' else f'BF@{args.tolerance}'

            arr1 = get_aligned_array(cond1, metric)
            arr2 = get_aligned_array(cond2, metric)
            diff = arr1 - arr2

            mean_diff = np.mean(diff)
            _, p_value = permutation_test(diff, n_permutations=args.n_permutations, seed=args.seed)
            ci_lower, ci_upper = bootstrap_ci(diff, seed=args.seed)

            n_cond1_wins = np.sum(diff > 0)
            prob_cond1_better = n_cond1_wins / n_samples

            delta = cliffs_delta(arr1, arr2)
            delta_interp = interpret_cliffs_delta(delta)

            sig = "*" if p_value < 0.05 else ""
            print(f"  {metric_label}: Δ = {mean_diff*100:+.2f} pp, p = {p_value:.3f}{sig}, "
                  f"95% CI [{ci_lower*100:+.2f}, {ci_upper*100:+.2f}], "
                  f"{cond1} wins {prob_cond1_better*100:.1f}%, "
                  f"Cliff's δ = {delta:+.3f} ({delta_interp})")

            baseline_results.append({
                'comparison': name,
                'metric': metric_label,
                'mean_diff_pp': mean_diff * 100,
                'p_value': p_value,
                'ci_lower_pp': ci_lower * 100,
                'ci_upper_pp': ci_upper * 100,
                'prob_cond1_better': prob_cond1_better,
                'cliffs_delta': delta,
                'cliffs_delta_interp': delta_interp,
            })

    # ==========================================================================
    # HOLM-BONFERRONI CORRECTION (per-family, per-metric)
    # ==========================================================================
    #
    # Families (matching statistical_analysis_persample.py and Baselines.tex):
    #   Family 1 — Temporal regime (RQ1): temporal_comparisons (indices 0-1)
    #   Family 2 — Bi-temporal baselines (RQ0, T=2): baseline_comparisons
    #              indices 0-4, 7-9
    #   Family 3 — Extended baselines (RQ0, T=7): baseline_comparisons
    #              indices 5-6
    #
    # IoU and BF are corrected independently within each family.
    # ==========================================================================

    print("\n" + "="*80)
    print("MULTIPLE COMPARISON CORRECTION (Holm-Bonferroni, per-family, per-metric)")
    print("="*80)

    # --- Family 1: Temporal regime (RQ1) ---
    # temporal_results has 2 comparisons × 2 metrics = 4 entries,
    # stored as [comp0_iou, comp0_bf, comp1_iou, comp1_bf]
    for metric in ['IoU', f'BF@{args.tolerance}']:
        family_entries = [r for r in temporal_results if r['metric'] == metric]
        raw_pvals = [r['p_value'] for r in family_entries]
        adj_pvals = holm_bonferroni_correction(raw_pvals)
        for r, p_adj in zip(family_entries, adj_pvals):
            r['p_adj'] = p_adj
            r['family'] = 'Temporal regime (RQ1)'
            r['family_m'] = len(raw_pvals)

    # --- Family 2 & 3: Baseline comparisons ---
    # baseline_results has 10 comparisons × 2 metrics = 20 entries,
    # stored as [comp0_iou, comp0_bf, comp1_iou, comp1_bf, ...]
    # Baseline comparison indices:
    #   T=2 family: 0,1,2,3,4,7,8,9  (8 comparisons)
    #   T=7 family: 5,6               (2 comparisons)
    t2_comparison_indices = {0, 1, 2, 3, 4, 7, 8, 9}
    t7_comparison_indices = {5, 6, 10, 11, 12, 13, 14}

    for metric in ['IoU', f'BF@{args.tolerance}']:
        # Each comparison produces 2 entries (IoU, BF); map comparison index
        # to the correct baseline_results entries for this metric.
        metric_entries = []
        for comp_idx, (name, _, _) in enumerate(baseline_comparisons):
            entry = [r for r in baseline_results
                     if r['comparison'] == name and r['metric'] == metric]
            if entry:
                metric_entries.append((comp_idx, entry[0]))

        # T=2 family
        t2_entries = [(idx, r) for idx, r in metric_entries if idx in t2_comparison_indices]
        t2_raw = [r['p_value'] for _, r in t2_entries]
        t2_adj = holm_bonferroni_correction(t2_raw)
        for (_, r), p_adj in zip(t2_entries, t2_adj):
            r['p_adj'] = p_adj
            r['family'] = 'Bi-temporal baselines (RQ0, T=2)'
            r['family_m'] = len(t2_raw)

        # T=7 family
        t7_entries = [(idx, r) for idx, r in metric_entries if idx in t7_comparison_indices]
        t7_raw = [r['p_value'] for _, r in t7_entries]
        t7_adj = holm_bonferroni_correction(t7_raw)
        for (_, r), p_adj in zip(t7_entries, t7_adj):
            r['p_adj'] = p_adj
            r['family'] = 'Extended baselines (RQ0, T=7)'
            r['family_m'] = len(t7_raw)

    # --- Print correction summary ---
    all_corrected = temporal_results + baseline_results
    families_seen = []
    for r in all_corrected:
        fam = r.get('family')
        if fam and fam not in families_seen:
            families_seen.append(fam)

    for fam in families_seen:
        fam_entries = [r for r in all_corrected if r.get('family') == fam]
        m = fam_entries[0]['family_m'] if fam_entries else '?'
        print(f"\n  Family: {fam} (m={m} per metric)")
        for r in fam_entries:
            sig = "✓ SIGNIFICANT" if r['p_adj'] < 0.05 else "not significant"
            delta_str = ""
            if 'cliffs_delta' in r:
                delta_str = f", Cliff's δ = {r['cliffs_delta']:+.3f} ({r['cliffs_delta_interp']})"
            print(f"    {r['comparison']} [{r['metric']}]:")
            print(f"      p_raw = {r['p_value']:.4f}  →  p_adj = {r['p_adj']:.4f} ({sig} at α=0.05){delta_str}")

    # ==========================================================================
    # SUMMARY TABLES
    # ==========================================================================
    print("\n" + "="*80)
    print("SUMMARY TABLES FOR EXPERIMENTS.TEX")
    print("="*80)

    # Table 1: Descriptive statistics
    print("\n% Table: Descriptive statistics (IoU and BF)")
    print("\\begin{table}[h]")
    print("\\centering")
    print(f"\\caption{{Region and boundary metrics by experiment (BF tolerance = {args.tolerance*10}m).}}")
    print("\\begin{tabular}{lcccc}")
    print("\\toprule")
    print("\\textbf{Condition} & \\textbf{Mean IoU} & \\textbf{Std IoU} & \\textbf{Mean BF} & \\textbf{Std BF} \\\\")
    print("\\midrule")

    for exp_key, exp_config in EXPERIMENTS.items():
        if exp_key == 'k1x1':
            continue  # Skip for main temporal table
        ious = get_aligned_array(exp_key, 'iou')
        bfs = get_aligned_array(exp_key, 'boundary_f_score')
        print(f"{exp_config['description']} & {np.mean(ious)*100:.1f}\\% & {np.std(ious, ddof=1)*100:.1f}\\% & "
              f"{np.mean(bfs)*100:.1f}\\% & {np.std(bfs, ddof=1)*100:.1f}\\% \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

    # Table 2: 1D vs 2D comparison
    print("\n% Table: 1D vs 2D comparison")
    print("\\begin{table}[h]")
    print("\\centering")
    print(f"\\caption{{1D vs 2D temporal modeling: region and boundary metrics (n={n_samples}).}}")
    print("\\begin{tabular}{lccccc}")
    print("\\toprule")
    print("\\textbf{Kernel} & \\textbf{Mean IoU} & \\textbf{Mean BF} & \\textbf{$\\Delta$IoU} & \\textbf{$\\Delta$BF} & \\textbf{p-value} \\\\")
    print("\\midrule")

    iou_3x3 = get_aligned_array('annual', 'iou')
    iou_1x1 = get_aligned_array('k1x1', 'iou')
    bf_3x3 = get_aligned_array('annual', 'boundary_f_score')
    bf_1x1 = get_aligned_array('k1x1', 'boundary_f_score')

    print(f"3×3 (patch) & {np.mean(iou_3x3)*100:.1f}\\% & {np.mean(bf_3x3)*100:.1f}\\% & --- & --- & --- \\\\")

    iou_diff = np.mean(iou_1x1) - np.mean(iou_3x3)
    bf_diff = np.mean(bf_1x1) - np.mean(bf_3x3)
    _, p_iou = permutation_test(iou_3x3 - iou_1x1, seed=args.seed)
    _, p_bf = permutation_test(bf_3x3 - bf_1x1, seed=args.seed)

    print(f"1×1 (pixel) & {np.mean(iou_1x1)*100:.1f}\\% & {np.mean(bf_1x1)*100:.1f}\\% & "
          f"{iou_diff*100:+.1f} pp & {bf_diff*100:+.1f} pp & {max(p_iou, p_bf):.3f} \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

    # Save results
    output_dir = MT_EXPERIMENTS_DIR.parent / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'tolerance_pixels': args.tolerance,
        'tolerance_meters': args.tolerance * 10,
        'threshold': args.threshold,
        'n_samples': n_samples,
        'seed': args.seed,
        'experiment_summaries': {},
        'temporal_comparisons': temporal_results,
        'baseline_comparisons': baseline_results,
        'kernel_comparisons': kernel_results,
        'per_sample': {exp_key: all_metrics[exp_key] for exp_key in all_metrics},
    }

    for exp_key, exp_config in EXPERIMENTS.items():
        ious = get_aligned_array(exp_key, 'iou')
        bfs = get_aligned_array(exp_key, 'boundary_f_score')
        results['experiment_summaries'][exp_key] = {
            'description': exp_config['description'],
            'mean_iou': float(np.mean(ious)),
            'std_iou': float(np.std(ious, ddof=1)),
            'mean_bf': float(np.mean(bfs)),
            'std_bf': float(np.std(bfs, ddof=1)),
        }

    output_file = output_dir / f"boundary_f_score_d{args.tolerance}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to: {output_file}")


if __name__ == "__main__":
    main()
