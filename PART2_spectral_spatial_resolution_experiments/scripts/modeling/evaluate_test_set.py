#!/usr/bin/env python3
"""
Test set evaluation script for Part II spectral/spatial resolution experiments.

Evaluates trained models on the held-out test set (8 tiles).
Supports single fold, all folds, and ensemble evaluation.
Saves per-tile predictions to disk for boundary metrics and qualitative analysis.

Usage:
    # Evaluate single fold
    python evaluate_test_set.py --experiment A3_s2_9band --fold 0

    # Evaluate all folds and compute ensemble
    python evaluate_test_set.py --experiment A3_s2_9band --all-folds

    # Save predictions for boundary metrics
    python evaluate_test_set.py --experiment A3_s2_9band --all-folds --save-predictions
"""

import sys
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import importlib.util

# ---------------------------------------------------------------------------
# Path setup for cross-part imports
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
PART1_DIR = REPO_ROOT / "PART1_multi_temporal_experiments"
PART2_DIR = REPO_ROOT / "PART2_spectral_spatial_resolution_experiments"

# Baseline utilities
sys.path.insert(0, str(REPO_ROOT / "scripts" / "modeling"))
from train import Metrics

# Part I models
sys.path.insert(0, str(PART1_DIR / "scripts" / "modeling"))
from models_multitemporal import create_multitemporal_model

# Part II dataset (importlib to avoid collision with baseline dataset.py)
_p2_spec = importlib.util.spec_from_file_location(
    "p2_dataset", PART2_DIR / "scripts" / "data_preparation" / "dataset.py"
)
_p2_dataset = importlib.util.module_from_spec(_p2_spec)
_p2_spec.loader.exec_module(_p2_dataset)
EXPERIMENT_CONFIGS = _p2_dataset.EXPERIMENT_CONFIGS
get_dataloaders = _p2_dataset.get_dataloaders

EXPERIMENTS_DIR = PART2_DIR / "outputs" / "experiments"


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def compute_sample_metrics(pred_logits: torch.Tensor, mask: torch.Tensor, threshold: float = 0.5):
    """Compute metrics for a single sample."""
    if pred_logits.dim() == 3:
        pred_logits = pred_logits.squeeze(0)
    if mask.dim() == 3:
        mask = mask.squeeze(0)

    pred_prob = torch.sigmoid(pred_logits)
    pred_binary = (pred_prob > threshold).float()

    pred_flat = pred_binary.view(-1)
    mask_flat = mask.view(-1)

    tp = ((pred_flat == 1) & (mask_flat == 1)).sum().float()
    fp = ((pred_flat == 1) & (mask_flat == 0)).sum().float()
    fn = ((pred_flat == 0) & (mask_flat == 1)).sum().float()
    tn = ((pred_flat == 0) & (mask_flat == 0)).sum().float()

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


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model_from_checkpoint(checkpoint_path: Path, config: dict, device: torch.device):
    """Load a trained model from checkpoint."""
    # Derive in_channels from experiment config
    cfg = EXPERIMENT_CONFIGS[config['experiment']]
    _, C, _, _ = cfg["expected_shape"]

    model = create_multitemporal_model(
        config['model_name'],
        encoder_name=config['encoder_name'],
        encoder_weights=None,  # Load trained weights, not pretrained
        in_channels=C,
        classes=1,
        lstm_hidden_dim=config.get('lstm_hidden_dim', 512),
        lstm_num_layers=config.get('lstm_num_layers', 2),
        convlstm_kernel_size=config.get('convlstm_kernel_size', 3),
        skip_aggregation=config.get('skip_aggregation', 'max'),
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_single_fold(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    return_predictions: bool = False,
):
    """Evaluate model on test set, return aggregated + per-sample metrics."""
    model.eval()
    metrics = Metrics()
    predictions = {}
    per_sample_metrics = {}

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            refids = batch['refid']

            outputs = model(images)

            metrics.update(outputs.detach(), masks)

            for i, refid in enumerate(refids):
                sample_metrics = compute_sample_metrics(outputs[i], masks[i])
                per_sample_metrics[refid] = sample_metrics

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
    """Compute ensemble predictions by averaging probabilities across folds."""
    refids = list(all_predictions[0].keys())

    metrics = Metrics()
    per_sample_metrics = {}
    ensemble_probs = {}

    for refid in refids:
        probs = [all_predictions[fold][refid]['prob'] for fold in all_predictions]
        avg_prob = np.mean(probs, axis=0)
        mask = all_predictions[0][refid]['mask']

        mask_tensor = torch.from_numpy(mask)

        eps = 1e-7
        avg_prob_clipped = np.clip(avg_prob, eps, 1 - eps)
        logits = np.log(avg_prob_clipped / (1 - avg_prob_clipped))
        logit_tensor = torch.from_numpy(logits).float()

        metrics.update(logit_tensor.unsqueeze(0), mask_tensor.unsqueeze(0))

        sample_metrics = compute_sample_metrics(logit_tensor, mask_tensor)
        per_sample_metrics[refid] = sample_metrics
        ensemble_probs[refid] = avg_prob

    return metrics.compute(), per_sample_metrics, ensemble_probs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Evaluate Part II models on test set")
    parser.add_argument('--experiment', type=str, required=True,
                        help='Experiment name (e.g., A3_s2_9band)')
    parser.add_argument('--fold', type=int, default=None,
                        help='Specific fold to evaluate (0-4)')
    parser.add_argument('--all-folds', action='store_true',
                        help='Evaluate all folds and compute ensemble')
    parser.add_argument('--num-folds', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--save-predictions', action='store_true',
                        help='Save prediction masks to disk (for boundary metrics)')
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()

    if args.fold is None and not args.all_folds:
        parser.error("Must specify either --fold or --all-folds")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if args.all_folds:
        folds = list(range(args.num_folds))
    else:
        folds = [args.fold]

    all_results = {}
    all_per_sample = {}
    all_predictions = {} if args.all_folds else None

    for fold in folds:
        print(f"\n{'='*60}")
        print(f"Evaluating {args.experiment} - Fold {fold}")
        print('='*60)

        exp_dir = EXPERIMENTS_DIR / f"{args.experiment}_fold{fold}"
        if not exp_dir.exists():
            print(f"WARNING: Experiment directory not found: {exp_dir}")
            continue

        # Load config
        with open(exp_dir / "config.json") as f:
            config = json.load(f)

        print(f"  Model: {config['model_name']}")
        print(f"  Experiment: {config['experiment']}")

        # Load model
        checkpoint_path = exp_dir / "best_model.pth"
        if not checkpoint_path.exists():
            print(f"WARNING: Checkpoint not found: {checkpoint_path}")
            continue

        print(f"  Loading checkpoint: {checkpoint_path}")
        model = load_model_from_checkpoint(checkpoint_path, config, device)

        # Create test dataloader
        print(f"  Creating test dataloader (fold {fold} normalization)...")
        dataloaders = get_dataloaders(
            experiment=config['experiment'],
            batch_size=args.batch_size,
            num_workers=4,
            image_size=config.get('image_size', 64),
            fold=fold,
            num_folds=args.num_folds,
            seed=config.get('seed', 42),
        )
        test_loader = dataloaders['test']
        print(f"  Test samples: {len(test_loader.dataset)}")

        # Evaluate
        need_preds = args.all_folds or args.save_predictions
        if need_preds:
            results, per_sample, predictions = evaluate_single_fold(
                model, test_loader, device, return_predictions=True
            )
            if args.all_folds:
                all_predictions[fold] = predictions
        else:
            results, per_sample = evaluate_single_fold(model, test_loader, device)
            predictions = None

        all_results[fold] = results
        all_per_sample[fold] = per_sample

        # Save per-fold predictions if requested
        if args.save_predictions and predictions is not None:
            pred_dir = (EXPERIMENTS_DIR / f"{args.experiment}_fold{fold}" / "predictions")
            pred_dir.mkdir(parents=True, exist_ok=True)
            for refid, pred_data in predictions.items():
                np.savez_compressed(
                    pred_dir / f"{refid}.npz",
                    prob=pred_data['prob'].astype(np.float16),
                    mask=pred_data['mask'].astype(np.uint8),
                )
            print(f"  Predictions saved to: {pred_dir}")

        # Print results
        print(f"\n  Fold {fold} Test Results (Aggregated):")
        print(f"    IoU:       {results['iou']*100:.2f}%")
        print(f"    F1:        {results['f1']*100:.2f}%")
        print(f"    Precision: {results['precision']*100:.2f}%")
        print(f"    Recall:    {results['recall']*100:.2f}%")

        print(f"\n  Per-Sample Results:")
        print(f"    {'RefID':<50} {'IoU':<8} {'F1':<8} {'Prec':<8} {'Rec':<8}")
        print(f"    {'-'*82}")
        for refid in sorted(per_sample.keys()):
            m = per_sample[refid]
            print(f"    {refid:<50} {m['iou']*100:<8.2f} {m['f1']*100:<8.2f} "
                  f"{m['precision']*100:<8.2f} {m['recall']*100:<8.2f}")

    # Summary across folds
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("Summary Across Folds")
        print('='*60)

        print(f"\n{'Fold':<6} {'IoU':<10} {'F1':<10} {'Precision':<10} {'Recall':<10}")
        print('-' * 50)
        for fold in sorted(all_results.keys()):
            r = all_results[fold]
            print(f"{fold:<6} {r['iou']*100:<10.2f} {r['f1']*100:<10.2f} "
                  f"{r['precision']*100:<10.2f} {r['recall']*100:<10.2f}")

        metrics_names = ['iou', 'f1', 'precision', 'recall']
        print('-' * 50)
        for metric in metrics_names:
            values = [all_results[f][metric] for f in all_results]
            mean_val = np.mean(values) * 100
            std_val = np.std(values) * 100
            print(f"Mean {metric.upper()}: {mean_val:.2f}% +/- {std_val:.2f}%")

        # Ensemble evaluation
        if args.all_folds and all_predictions:
            print(f"\n{'='*60}")
            print("Ensemble Evaluation (Average Probabilities)")
            print('='*60)

            ensemble_results, ensemble_per_sample, ensemble_probs = evaluate_ensemble(all_predictions)
            print(f"\n  Aggregated:")
            print(f"    IoU:       {ensemble_results['iou']*100:.2f}%")
            print(f"    F1:        {ensemble_results['f1']*100:.2f}%")
            print(f"    Precision: {ensemble_results['precision']*100:.2f}%")
            print(f"    Recall:    {ensemble_results['recall']*100:.2f}%")

            print(f"\n  Per-Sample (Ensemble):")
            print(f"    {'RefID':<50} {'IoU':<8} {'F1':<8} {'Prec':<8} {'Rec':<8}")
            print(f"    {'-'*82}")
            for refid in sorted(ensemble_per_sample.keys()):
                m = ensemble_per_sample[refid]
                print(f"    {refid:<50} {m['iou']*100:<8.2f} {m['f1']*100:<8.2f} "
                      f"{m['precision']*100:<8.2f} {m['recall']*100:<8.2f}")

            # Save ensemble predictions
            if args.save_predictions:
                ens_dir = EXPERIMENTS_DIR / args.experiment / "ensemble_predictions"
                ens_dir.mkdir(parents=True, exist_ok=True)
                for refid, prob in ensemble_probs.items():
                    mask = all_predictions[0][refid]['mask']
                    np.savez_compressed(
                        ens_dir / f"{refid}.npz",
                        prob=prob.astype(np.float16),
                        mask=mask.astype(np.uint8),
                    )
                print(f"\n  Ensemble predictions saved to: {ens_dir}")

    # Save results JSON
    output_dir = Path(args.output_dir) if args.output_dir else (
        EXPERIMENTS_DIR / args.experiment
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    serializable_results = {
        'experiment': args.experiment,
        'folds': {
            str(fold): {
                'aggregated': {k: float(v) for k, v in all_results[fold].items()},
                'per_sample': {
                    refid: {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                            for k, v in m.items()}
                    for refid, m in all_per_sample[fold].items()
                }
            }
            for fold in all_results.keys()
        }
    }

    if len(all_results) > 1:
        for metric in metrics_names:
            values = [all_results[f][metric] for f in all_results]
            serializable_results[f'mean_{metric}'] = float(np.mean(values))
            serializable_results[f'std_{metric}'] = float(np.std(values))

        if args.all_folds and all_predictions:
            serializable_results['ensemble'] = {
                'aggregated': {k: float(v) for k, v in ensemble_results.items()},
                'per_sample': {
                    refid: {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                            for k, v in m.items()}
                    for refid, m in ensemble_per_sample.items()
                }
            }

    results_file = output_dir / "test_results.json"
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    # Save CSV summary
    csv_lines = ["experiment,fold,refid,iou,f1,precision,recall"]
    for fold in sorted(all_per_sample.keys()):
        for refid in sorted(all_per_sample[fold].keys()):
            m = all_per_sample[fold][refid]
            csv_lines.append(
                f"{args.experiment},{fold},{refid},"
                f"{m['iou']:.6f},{m['f1']:.6f},{m['precision']:.6f},{m['recall']:.6f}"
            )
    csv_file = output_dir / "test_results.csv"
    with open(csv_file, 'w') as f:
        f.write('\n'.join(csv_lines) + '\n')
    print(f"CSV saved to: {csv_file}")


if __name__ == "__main__":
    main()
