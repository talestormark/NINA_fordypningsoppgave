#!/usr/bin/env python3
"""
Test set evaluation script for multi-temporal land-take detection models.

Evaluates trained models on the held-out test set (8 samples).
Supports:
- Single fold evaluation
- Ensemble evaluation (average predictions from all folds)
- Per-sample metrics for qualitative example selection

Usage:
    # Evaluate single fold
    python evaluate_test_set.py --experiment exp001_v2 --fold 0

    # Evaluate all folds and compute ensemble
    python evaluate_test_set.py --experiment exp001_v2 --all-folds
"""

import sys
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# Add paths for imports
script_dir = Path(__file__).resolve().parent
parent_dir = script_dir.parent.parent
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(parent_dir.parent))

# Import baseline utilities
sys.path.insert(0, str(parent_dir.parent / "scripts" / "modeling"))
from train import Metrics

# Import multi-temporal modules
from models_multitemporal import create_multitemporal_model
from multi_temporal_experiments.scripts.data_preparation.dataset_multitemporal import get_dataloaders
from multi_temporal_experiments.config import MT_EXPERIMENTS_DIR


def compute_sample_metrics(pred_logits: torch.Tensor, mask: torch.Tensor, threshold: float = 0.5):
    """
    Compute metrics for a single sample.

    Args:
        pred_logits: Model output logits (1, H, W) or (H, W)
        mask: Ground truth mask (H, W)
        threshold: Binary threshold for predictions

    Returns:
        dict: Per-sample metrics (iou, f1, precision, recall)
    """
    # Ensure correct shapes
    if pred_logits.dim() == 3:
        pred_logits = pred_logits.squeeze(0)
    if mask.dim() == 3:
        mask = mask.squeeze(0)

    # Convert to binary predictions
    pred_prob = torch.sigmoid(pred_logits)
    pred_binary = (pred_prob > threshold).float()

    # Flatten for metric computation
    pred_flat = pred_binary.view(-1)
    mask_flat = mask.view(-1)

    # Compute TP, FP, FN, TN
    tp = ((pred_flat == 1) & (mask_flat == 1)).sum().float()
    fp = ((pred_flat == 1) & (mask_flat == 0)).sum().float()
    fn = ((pred_flat == 0) & (mask_flat == 1)).sum().float()
    tn = ((pred_flat == 0) & (mask_flat == 0)).sum().float()

    # Compute metrics with epsilon to avoid division by zero
    eps = 1e-7

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)

    return {
        'iou': iou.item(),
        'f1': f1.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'tp': int(tp.item()),
        'fp': int(fp.item()),
        'fn': int(fn.item()),
        'tn': int(tn.item()),
    }


def load_model_from_checkpoint(checkpoint_path: Path, config: dict, device: torch.device):
    """
    Load a trained model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint (.pth file)
        config: Model configuration dict
        device: Device to load model to

    Returns:
        Loaded model in eval mode
    """
    # Create model with same architecture
    model = create_multitemporal_model(
        config['model_name'],
        encoder_name=config['encoder_name'],
        encoder_weights=None,  # Don't load ImageNet weights, we'll load trained weights
        in_channels=9,
        classes=1,
        lstm_hidden_dim=config['lstm_hidden_dim'],
        lstm_num_layers=config['lstm_num_layers'],
        skip_aggregation=config['skip_aggregation'],
    )

    # Load trained weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    return model


def evaluate_single_fold(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    return_predictions: bool = False,
):
    """
    Evaluate model on test set.

    Args:
        model: Trained model
        test_loader: Test dataloader
        device: Device
        return_predictions: Whether to return raw predictions for ensemble

    Returns:
        dict: Evaluation metrics (aggregated)
        dict: Per-sample metrics
        dict (optional): Predictions per sample if return_predictions=True
    """
    model.eval()
    metrics = Metrics()
    predictions = {}
    per_sample_metrics = {}

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            images = batch['image'].to(device)  # (B, T, C, H, W)
            masks = batch['mask'].to(device)    # (B, H, W)
            refids = batch['refid']             # List of refids

            # Forward pass
            outputs = model(images)  # (B, 1, H, W)

            # Update aggregated metrics
            metrics.update(outputs, masks)

            # Compute per-sample metrics
            for i, refid in enumerate(refids):
                sample_metrics = compute_sample_metrics(
                    outputs[i], masks[i]
                )
                per_sample_metrics[refid] = sample_metrics

            # Store predictions for ensemble
            if return_predictions:
                probs = torch.sigmoid(outputs)
                for i, refid in enumerate(refids):
                    predictions[refid] = {
                        'prob': probs[i].cpu().numpy(),
                        'mask': masks[i].cpu().numpy(),
                        'logits': outputs[i].cpu().numpy(),
                    }

    results = metrics.compute()

    if return_predictions:
        return results, per_sample_metrics, predictions
    return results, per_sample_metrics


def evaluate_ensemble(all_predictions: dict, threshold: float = 0.5):
    """
    Compute ensemble predictions by averaging probabilities.

    Args:
        all_predictions: Dict mapping fold -> {refid -> {'prob': ..., 'mask': ...}}
        threshold: Binary threshold for final prediction

    Returns:
        dict: Ensemble evaluation metrics (aggregated)
        dict: Per-sample ensemble metrics
    """
    # Get all refids from first fold
    refids = list(all_predictions[0].keys())

    # Aggregate predictions
    metrics = Metrics()
    per_sample_metrics = {}

    for refid in refids:
        # Average probabilities across folds
        probs = [all_predictions[fold][refid]['prob'] for fold in all_predictions]
        avg_prob = np.mean(probs, axis=0)

        # Get ground truth (same across folds)
        mask = all_predictions[0][refid]['mask']

        # Convert to tensors for metrics computation
        mask_tensor = torch.from_numpy(mask)

        # Metrics.update expects logits, so convert prob back to logits
        # logit = log(p / (1-p))
        eps = 1e-7
        avg_prob_clipped = np.clip(avg_prob, eps, 1 - eps)
        logits = np.log(avg_prob_clipped / (1 - avg_prob_clipped))
        logit_tensor = torch.from_numpy(logits).float()

        # Update aggregated metrics
        metrics.update(logit_tensor.unsqueeze(0), mask_tensor.unsqueeze(0))

        # Compute per-sample metrics for ensemble
        sample_metrics = compute_sample_metrics(logit_tensor, mask_tensor)
        per_sample_metrics[refid] = sample_metrics

    return metrics.compute(), per_sample_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate models on test set")
    parser.add_argument('--experiment', type=str, required=True,
                        help='Experiment name (e.g., exp001_v2)')
    parser.add_argument('--fold', type=int, default=None,
                        help='Specific fold to evaluate (0-4)')
    parser.add_argument('--all-folds', action='store_true',
                        help='Evaluate all folds and compute ensemble')
    parser.add_argument('--num-folds', type=int, default=5,
                        help='Number of folds')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for evaluation')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for results (default: experiment dir)')
    args = parser.parse_args()

    if args.fold is None and not args.all_folds:
        parser.error("Must specify either --fold or --all-folds")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Determine folds to evaluate
    if args.all_folds:
        folds = list(range(args.num_folds))
    else:
        folds = [args.fold]

    # Results storage
    all_results = {}
    all_per_sample = {}
    all_predictions = {} if args.all_folds else None

    # Evaluate each fold
    for fold in folds:
        print(f"\n{'='*60}")
        print(f"Evaluating {args.experiment} - Fold {fold}")
        print('='*60)

        # Find experiment directory
        exp_dir = MT_EXPERIMENTS_DIR / "outputs" / "experiments" / f"{args.experiment}_fold{fold}"
        if not exp_dir.exists():
            print(f"WARNING: Experiment directory not found: {exp_dir}")
            continue

        # Load config
        config_path = exp_dir / "config.json"
        with open(config_path) as f:
            config = json.load(f)

        print(f"  Model: {config['model_name']}")
        print(f"  Temporal sampling: {config['temporal_sampling']}")

        # Load model
        checkpoint_path = exp_dir / "best_model.pth"
        if not checkpoint_path.exists():
            print(f"WARNING: Checkpoint not found: {checkpoint_path}")
            continue

        print(f"  Loading checkpoint: {checkpoint_path}")
        model = load_model_from_checkpoint(checkpoint_path, config, device)

        # Create dataloader with same fold configuration (for correct normalization)
        print(f"  Creating test dataloader (fold {fold} normalization)...")
        dataloaders = get_dataloaders(
            temporal_sampling=config['temporal_sampling'],
            batch_size=args.batch_size,
            num_workers=4,
            image_size=config['image_size'],
            output_format="LSTM",
            fold=fold,
            num_folds=args.num_folds,
            seed=config.get('seed', 42),
        )
        test_loader = dataloaders['test']
        print(f"  Test samples: {len(test_loader.dataset)}")

        # Evaluate
        if args.all_folds:
            results, per_sample, predictions = evaluate_single_fold(
                model, test_loader, device, return_predictions=True
            )
            all_predictions[fold] = predictions
        else:
            results, per_sample = evaluate_single_fold(model, test_loader, device)

        all_results[fold] = results
        all_per_sample[fold] = per_sample

        # Print aggregated results
        print(f"\n  Fold {fold} Test Results (Aggregated):")
        print(f"    IoU:       {results['iou']*100:.2f}%")
        print(f"    F1:        {results['f1']*100:.2f}%")
        print(f"    Precision: {results['precision']*100:.2f}%")
        print(f"    Recall:    {results['recall']*100:.2f}%")

        # Print per-sample results
        print(f"\n  Per-Sample Results:")
        print(f"    {'RefID':<45} {'IoU':<8} {'F1':<8} {'Prec':<8} {'Rec':<8}")
        print(f"    {'-'*77}")
        for refid in sorted(per_sample.keys()):
            m = per_sample[refid]
            print(f"    {refid:<45} {m['iou']*100:<8.2f} {m['f1']*100:<8.2f} "
                  f"{m['precision']*100:<8.2f} {m['recall']*100:<8.2f}")

    # Summary statistics
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("Summary Across Folds")
        print('='*60)

        # Per-fold table
        print(f"\n{'Fold':<6} {'IoU':<10} {'F1':<10} {'Precision':<10} {'Recall':<10}")
        print('-' * 50)
        for fold in sorted(all_results.keys()):
            r = all_results[fold]
            print(f"{fold:<6} {r['iou']*100:<10.2f} {r['f1']*100:<10.2f} "
                  f"{r['precision']*100:<10.2f} {r['recall']*100:<10.2f}")

        # Mean ± std
        metrics_names = ['iou', 'f1', 'precision', 'recall']
        print('-' * 50)
        for metric in metrics_names:
            values = [all_results[f][metric] for f in all_results]
            mean = np.mean(values) * 100
            std = np.std(values) * 100
            print(f"Mean {metric.upper()}: {mean:.2f}% ± {std:.2f}%")

        # Ensemble evaluation
        if args.all_folds and all_predictions:
            print(f"\n{'='*60}")
            print("Ensemble Evaluation (Average Probabilities)")
            print('='*60)

            ensemble_results, ensemble_per_sample = evaluate_ensemble(all_predictions)
            print(f"\n  Aggregated:")
            print(f"    IoU:       {ensemble_results['iou']*100:.2f}%")
            print(f"    F1:        {ensemble_results['f1']*100:.2f}%")
            print(f"    Precision: {ensemble_results['precision']*100:.2f}%")
            print(f"    Recall:    {ensemble_results['recall']*100:.2f}%")

            # Print per-sample ensemble results
            print(f"\n  Per-Sample (Ensemble):")
            print(f"    {'RefID':<45} {'IoU':<8} {'F1':<8} {'Prec':<8} {'Rec':<8}")
            print(f"    {'-'*77}")
            for refid in sorted(ensemble_per_sample.keys()):
                m = ensemble_per_sample[refid]
                print(f"    {refid:<45} {m['iou']*100:<8.2f} {m['f1']*100:<8.2f} "
                      f"{m['precision']*100:<8.2f} {m['recall']*100:<8.2f}")

    # Save results
    output_dir = Path(args.output_dir) if args.output_dir else (
        MT_EXPERIMENTS_DIR / "outputs" / "experiments" / args.experiment
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / "test_results.json"

    # Convert numpy values to Python floats for JSON serialization
    serializable_results = {
        'experiment': args.experiment,
        'folds': {
            str(fold): {
                'aggregated': {k: float(v) for k, v in all_results[fold].items()},
                'per_sample': {
                    refid: {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                            for k, v in metrics.items()}
                    for refid, metrics in all_per_sample[fold].items()
                }
            }
            for fold in all_results.keys()
        }
    }

    if len(all_results) > 1:
        # Add summary stats
        for metric in metrics_names:
            values = [all_results[f][metric] for f in all_results]
            serializable_results[f'mean_{metric}'] = float(np.mean(values))
            serializable_results[f'std_{metric}'] = float(np.std(values))

        # Add ensemble results if computed
        if args.all_folds and all_predictions:
            serializable_results['ensemble'] = {
                'aggregated': {k: float(v) for k, v in ensemble_results.items()},
                'per_sample': {
                    refid: {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                            for k, v in metrics.items()}
                    for refid, metrics in ensemble_per_sample.items()
                }
            }

    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
