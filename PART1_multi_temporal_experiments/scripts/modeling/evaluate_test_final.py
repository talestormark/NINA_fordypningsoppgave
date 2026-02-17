#!/usr/bin/env python3
"""
Final Test Set Evaluation - Multi-temporal Land-Take Detection

This script performs the final, locked evaluation on the held-out test set (8 tiles).
All decisions are frozen before looking at test results.

PROTOCOL (locked before test evaluation):
=========================================
1. Model selection: CV ensemble (average probabilities from 5 fold models)
2. Checkpoint selection: Best validation IoU checkpoint for each fold
3. Threshold: Fixed at 0.5 (no tuning on test set)
4. Preprocessing: Per-fold training normalization stats
5. Cropping: Center crop to 64x64 for test samples
6. Ensemble method: Mean of predicted probabilities across folds

METRICS REPORTED:
=================
- Per-tile IoU: Mean, Median, Std, 95% CI (bootstrap)
- Micro IoU: Aggregated over all test pixels
- F1, Precision, Recall at threshold 0.5
- Per-tile breakdown for transparency

Usage:
    python evaluate_test_final.py --condition annual
    python evaluate_test_final.py --condition bi_temporal
    python evaluate_test_final.py --condition bi_seasonal
    python evaluate_test_final.py --condition early_fusion
    python evaluate_test_final.py --all-conditions
    python evaluate_test_final.py --architecture-variants
"""

import sys
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# Add paths for imports
script_dir = Path(__file__).resolve().parent
parent_dir = script_dir.parent.parent
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(parent_dir.parent))

# Import baseline utilities
sys.path.insert(0, str(parent_dir.parent / "scripts" / "modeling"))

# Import multi-temporal modules
from models_multitemporal import create_multitemporal_model
from PART1_multi_temporal_experiments.scripts.data_preparation.dataset_multitemporal import get_dataloaders
from PART1_multi_temporal_experiments.config import MT_EXPERIMENTS_DIR


# LOCKED CONFIGURATION - DO NOT MODIFY AFTER TEST EVALUATION BEGINS
EXPERIMENTS = {
    # Temporal sampling conditions (RQ1) - all use unified 400-epoch protocol
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
    # Architecture variants (RQ2)
    'early_fusion': {
        'name': 'exp005_early_fusion',
        'temporal_sampling': 'bi_temporal',
        'T': 2,
    },
    'late_fusion': {
        'name': 'exp006_late_fusion',
        'temporal_sampling': 'bi_temporal',
        'T': 2,
    },
    'late_fusion_pool': {
        'name': 'exp007_late_fusion_pool',
        'temporal_sampling': 'annual',
        'T': 7,
    },
    'conv3d_fusion': {
        'name': 'exp008_conv3d_fusion',
        'temporal_sampling': 'annual',
        'T': 7,
    },
    'lstm_lite_t2': {
        'name': 'exp009_lstm_lite',
        'temporal_sampling': 'bi_temporal',
        'T': 2,
    },
    'k1x1': {
        'name': 'exp004_v2',
        'temporal_sampling': 'annual',
        'T': 7,
    },
    # Extended experiments
    'lstm7_lite': {
        'name': 'exp011_lstm7_lite',
        'temporal_sampling': 'annual',
        'T': 7,
    },
}

# Original three temporal conditions
TEMPORAL_CONDITIONS = ['annual', 'bi_temporal', 'bi_seasonal']
# Architecture variant conditions
ARCHITECTURE_CONDITIONS = [
    'early_fusion', 'late_fusion', 'late_fusion_pool', 'conv3d_fusion',
    'lstm_lite_t2', 'k1x1', 'lstm7_lite',
]

THRESHOLD = 0.5  # LOCKED - do not change
NUM_FOLDS = 5
SEED = 42
N_BOOTSTRAP = 10000


def compute_per_tile_metrics(pred_prob: np.ndarray, mask: np.ndarray, threshold: float = 0.5):
    """
    Compute metrics for a single tile.

    Convention: If union=0 (both prediction and GT empty), IoU=1 (perfect match).
    In practice, all test tiles have land-take, so this case doesn't occur.
    """
    pred_binary = (pred_prob > threshold).astype(np.float32)

    pred_flat = pred_binary.flatten()
    mask_flat = mask.flatten()

    tp = ((pred_flat == 1) & (mask_flat == 1)).sum()
    fp = ((pred_flat == 1) & (mask_flat == 0)).sum()
    fn = ((pred_flat == 0) & (mask_flat == 1)).sum()
    tn = ((pred_flat == 0) & (mask_flat == 0)).sum()

    union = tp + fp + fn
    if union == 0:
        iou = 1.0  # Perfect match (both empty)
    else:
        iou = tp / union

    eps = 1e-7
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    return {
        'iou': float(iou),
        'f1': float(f1),
        'precision': float(precision),
        'recall': float(recall),
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn),
    }


def compute_micro_metrics(all_tp: int, all_fp: int, all_fn: int):
    """Compute micro-averaged metrics over all pixels."""
    union = all_tp + all_fp + all_fn
    if union == 0:
        iou = 1.0
    else:
        iou = all_tp / union

    eps = 1e-7
    precision = all_tp / (all_tp + all_fp + eps)
    recall = all_tp / (all_tp + all_fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    return {
        'iou': float(iou),
        'f1': float(f1),
        'precision': float(precision),
        'recall': float(recall),
    }


def bootstrap_ci(values: np.ndarray, n_bootstrap: int = 10000, ci: float = 0.95, seed: int = 42):
    """Compute bootstrap confidence interval."""
    rng = np.random.RandomState(seed)
    n = len(values)

    bootstrap_means = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        indices = rng.choice(n, size=n, replace=True)
        bootstrap_means[i] = np.mean(values[indices])

    alpha = 1 - ci
    lower = np.percentile(bootstrap_means, alpha / 2 * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)

    return lower, upper


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


def evaluate_condition(condition_name: str, device: torch.device, verbose: bool = True):
    """
    Evaluate a single condition using CV ensemble on test set.

    Returns:
        dict: Complete evaluation results
    """
    config = EXPERIMENTS[condition_name]
    exp_name = config['name']
    temporal_sampling = config['temporal_sampling']

    if verbose:
        print(f"\n{'='*70}")
        print(f"EVALUATING: {condition_name.upper()} (T={config['T']})")
        print(f"Experiment: {exp_name}")
        print(f"{'='*70}")

    # Collect predictions from all folds
    all_fold_predictions = {}

    for fold in range(NUM_FOLDS):
        exp_dir = MT_EXPERIMENTS_DIR / f"{exp_name}_fold{fold}"

        if not exp_dir.exists():
            raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

        # Load config
        config_path = exp_dir / "config.json"
        with open(config_path) as f:
            model_config = json.load(f)

        # Load model (best validation IoU checkpoint)
        checkpoint_path = exp_dir / "best_model.pth"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        if verbose:
            print(f"\n  Fold {fold}: Loading {checkpoint_path.name}")

        model = load_model_from_checkpoint(checkpoint_path, model_config, device)

        # Create test dataloader with this fold's normalization stats
        dataloaders = get_dataloaders(
            temporal_sampling=temporal_sampling,
            batch_size=1,
            num_workers=4,
            image_size=model_config['image_size'],
            output_format="LSTM",
            fold=fold,
            num_folds=NUM_FOLDS,
            seed=SEED,
        )
        test_loader = dataloaders['test']

        if verbose and fold == 0:
            print(f"  Test samples: {len(test_loader.dataset)}")

        # Run inference on test set
        fold_predictions = {}
        with torch.no_grad():
            for batch in test_loader:
                images = batch['image'].to(device)
                masks = batch['mask']
                refids = batch['refid']

                outputs = model(images)
                probs = torch.sigmoid(outputs)

                for i, refid in enumerate(refids):
                    fold_predictions[refid] = {
                        'prob': probs[i].cpu().numpy().squeeze(),
                        'mask': masks[i].numpy(),
                    }

        all_fold_predictions[fold] = fold_predictions

    # Verify all folds have same test samples
    refids = sorted(all_fold_predictions[0].keys())
    for fold in range(1, NUM_FOLDS):
        if sorted(all_fold_predictions[fold].keys()) != refids:
            raise ValueError(f"Test sample mismatch between folds!")

    if verbose:
        print(f"\n  Ensembling {NUM_FOLDS} fold models...")

    # Compute ensemble predictions and metrics
    per_tile_metrics = {}
    total_tp, total_fp, total_fn = 0, 0, 0

    for refid in refids:
        # Average probabilities across all folds
        probs = [all_fold_predictions[fold][refid]['prob'] for fold in range(NUM_FOLDS)]
        avg_prob = np.mean(probs, axis=0)

        # Ground truth (same across folds)
        mask = all_fold_predictions[0][refid]['mask']

        # Compute per-tile metrics
        tile_metrics = compute_per_tile_metrics(avg_prob, mask, THRESHOLD)
        per_tile_metrics[refid] = tile_metrics

        # Accumulate for micro metrics
        total_tp += tile_metrics['tp']
        total_fp += tile_metrics['fp']
        total_fn += tile_metrics['fn']

    # Compute summary statistics
    iou_values = np.array([m['iou'] for m in per_tile_metrics.values()])
    f1_values = np.array([m['f1'] for m in per_tile_metrics.values()])
    precision_values = np.array([m['precision'] for m in per_tile_metrics.values()])
    recall_values = np.array([m['recall'] for m in per_tile_metrics.values()])

    # Bootstrap CIs
    iou_ci = bootstrap_ci(iou_values, N_BOOTSTRAP, seed=SEED)
    f1_ci = bootstrap_ci(f1_values, N_BOOTSTRAP, seed=SEED)

    # Micro metrics
    micro_metrics = compute_micro_metrics(total_tp, total_fp, total_fn)

    results = {
        'condition': condition_name,
        'experiment': exp_name,
        'T': config['T'],
        'n_test_tiles': len(refids),
        'n_folds': NUM_FOLDS,
        'threshold': THRESHOLD,
        'ensemble_method': 'mean_probability',
        'checkpoint_selection': 'best_validation_iou',

        # Per-tile statistics
        'per_tile': {
            'iou': {
                'mean': float(np.mean(iou_values)),
                'median': float(np.median(iou_values)),
                'std': float(np.std(iou_values)),
                'ci_lower': float(iou_ci[0]),
                'ci_upper': float(iou_ci[1]),
            },
            'f1': {
                'mean': float(np.mean(f1_values)),
                'median': float(np.median(f1_values)),
                'std': float(np.std(f1_values)),
                'ci_lower': float(f1_ci[0]),
                'ci_upper': float(f1_ci[1]),
            },
            'precision': {
                'mean': float(np.mean(precision_values)),
                'std': float(np.std(precision_values)),
            },
            'recall': {
                'mean': float(np.mean(recall_values)),
                'std': float(np.std(recall_values)),
            },
        },

        # Micro metrics (aggregated over all pixels)
        'micro': micro_metrics,

        # Per-tile breakdown
        'tiles': {refid: per_tile_metrics[refid] for refid in refids},

        # Metadata
        'evaluation_timestamp': datetime.now().isoformat(),
        'seed': SEED,
        'n_bootstrap': N_BOOTSTRAP,
    }

    if verbose:
        print(f"\n  RESULTS ({condition_name}):")
        print(f"  {'-'*50}")
        print(f"  Per-tile IoU:")
        print(f"    Mean:   {results['per_tile']['iou']['mean']*100:.2f}%")
        print(f"    Median: {results['per_tile']['iou']['median']*100:.2f}%")
        print(f"    Std:    {results['per_tile']['iou']['std']*100:.2f}%")
        print(f"    95% CI: [{results['per_tile']['iou']['ci_lower']*100:.2f}%, {results['per_tile']['iou']['ci_upper']*100:.2f}%]")
        print(f"  Micro IoU: {results['micro']['iou']*100:.2f}%")
        print(f"  F1:        {results['per_tile']['f1']['mean']*100:.2f}%")
        print(f"  Precision: {results['per_tile']['precision']['mean']*100:.2f}%")
        print(f"  Recall:    {results['per_tile']['recall']['mean']*100:.2f}%")

        print(f"\n  Per-tile breakdown:")
        print(f"  {'RefID':<45} {'IoU':<8} {'F1':<8} {'Prec':<8} {'Rec':<8}")
        print(f"  {'-'*77}")
        for refid in sorted(per_tile_metrics.keys()):
            m = per_tile_metrics[refid]
            print(f"  {refid:<45} {m['iou']*100:<8.2f} {m['f1']*100:<8.2f} "
                  f"{m['precision']*100:<8.2f} {m['recall']*100:<8.2f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Final test set evaluation")
    parser.add_argument('--condition', type=str, choices=list(EXPERIMENTS.keys()),
                        help='Condition to evaluate')
    parser.add_argument('--all-conditions', action='store_true',
                        help='Evaluate all conditions (temporal + architecture)')
    parser.add_argument('--temporal-only', action='store_true',
                        help='Evaluate only the 3 temporal conditions')
    parser.add_argument('--architecture-variants', action='store_true',
                        help='Evaluate only the architecture variant conditions')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for results')
    args = parser.parse_args()

    if not args.condition and not args.all_conditions and not args.temporal_only and not args.architecture_variants:
        parser.error("Must specify --condition, --all-conditions, --temporal-only, or --architecture-variants")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\n" + "="*70)
    print("FINAL TEST SET EVALUATION")
    print("="*70)
    print("\nPROTOCOL (LOCKED):")
    print(f"  - Ensemble: CV ensemble ({NUM_FOLDS} fold models, mean probability)")
    print(f"  - Checkpoint: Best validation IoU")
    print(f"  - Threshold: {THRESHOLD} (fixed)")
    print(f"  - Normalization: Per-fold training stats")
    print(f"  - Bootstrap samples: {N_BOOTSTRAP}")

    # Determine conditions to evaluate
    if args.all_conditions:
        conditions = list(EXPERIMENTS.keys())
    elif args.temporal_only:
        conditions = TEMPORAL_CONDITIONS
    elif args.architecture_variants:
        conditions = ARCHITECTURE_CONDITIONS
    else:
        conditions = [args.condition]

    # Evaluate
    all_results = {}
    for condition in conditions:
        results = evaluate_condition(condition, device)
        all_results[condition] = results

    # Summary comparison (if multiple conditions)
    if len(all_results) > 1:
        print("\n" + "="*70)
        print("SUMMARY COMPARISON (Test Set)")
        print("="*70)
        print(f"\n{'Condition':<15} {'T':<4} {'Mean IoU':<12} {'95% CI':<20} {'Micro IoU':<12}")
        print("-"*70)
        for cond, res in all_results.items():
            pt = res['per_tile']['iou']
            micro = res['micro']['iou']
            ci_str = f"[{pt['ci_lower']*100:.1f}%, {pt['ci_upper']*100:.1f}%]"
            print(f"{cond:<15} {res['T']:<4} {pt['mean']*100:<12.2f} {ci_str:<20} {micro*100:<12.2f}")

    # Save results
    output_dir = Path(args.output_dir) if args.output_dir else (
        MT_EXPERIMENTS_DIR.parent / "analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / "test_set_evaluation.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Results saved to: {results_file}")

    # Print LaTeX table
    print("\n" + "="*70)
    print("LATEX TABLE")
    print("="*70)

    # Display name mapping for LaTeX
    DISPLAY_NAMES = {
        'annual': 'Annual (LSTM-7)',
        'bi_temporal': 'Bi-temporal (LSTM-2)',
        'bi_seasonal': 'Bi-seasonal (LSTM-14)',
        'early_fusion': 'Early-Fusion (T=2)',
        'late_fusion': 'Late-Fusion (T=2)',
        'late_fusion_pool': 'Pool-7 (T=7)',
        'conv3d_fusion': 'Conv3D-7 (T=7)',
        'lstm_lite_t2': 'LSTM-2-lite (T=2)',
        'k1x1': 'LSTM-1x1 (T=7)',
        'lstm7_no_es': 'LSTM-7 no ES (T=7)',
        'lstm7_lite': 'LSTM-7-lite (T=7)',
    }

    print("""
\\begin{table}[h]
\\centering
\\caption{Test set results (8 held-out tiles, CV ensemble).}
\\label{tab:test-results}
\\begin{tabular}{lccccc}
\\toprule
\\textbf{Condition} & $T$ & \\textbf{Mean IoU} & \\textbf{95\\% CI} & \\textbf{Micro IoU} & \\textbf{F1} \\\\
\\midrule""")
    for cond in all_results:
        res = all_results[cond]
        pt = res['per_tile']['iou']
        cond_name = DISPLAY_NAMES.get(cond, cond.replace('_', '-').title())
        print(f"{cond_name} & {res['T']} & {pt['mean']*100:.1f}\\% & "
              f"$[{pt['ci_lower']*100:.1f}, {pt['ci_upper']*100:.1f}]$ & "
              f"{res['micro']['iou']*100:.1f}\\% & {res['per_tile']['f1']['mean']*100:.1f}\\% \\\\")
    print("""\\bottomrule
\\end{tabular}
\\end{table}""")


if __name__ == "__main__":
    main()
