#!/usr/bin/env python3
"""
Evaluation script for baseline land-take detection models on test set.

Performs:
- Inference on test set (8 held-out tiles)
- Per-tile and overall metrics (F1, IoU, precision, recall)
- Prediction mask generation
- Visualization of predictions vs ground truth
- Results stratified by change level

Usage:
    python3 scripts/modeling/evaluate.py \
        --checkpoint outputs/training/siam_conc_resnet50_20251113_094511/best_model.pth \
        --output-dir outputs/evaluation/siam_conc_resnet50
"""

import argparse
import torch
import torch.nn as nn
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd

# Import local modules
from models import create_model
from dataset import create_dataloaders
from train import Metrics
import wandb


@torch.no_grad()
def evaluate_model(model, dataloader, device, save_predictions=True, output_dir=None):
    """
    Evaluate model on test set.

    Args:
        model: PyTorch model
        dataloader: Test dataloader
        device: Device to run on
        save_predictions: Whether to save prediction masks
        output_dir: Directory to save predictions

    Returns:
        Dictionary with overall metrics and per-tile results
    """
    model.eval()

    # Overall metrics
    overall_metrics = Metrics()

    # Per-tile results
    per_tile_results = []
    all_predictions = []
    all_targets = []
    all_tile_ids = []

    print(f"\nEvaluating on {len(dataloader)} batches...")

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Test set evaluation")):
        # Get data
        if 'image' in batch:
            # Early Fusion mode
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            outputs = model(images)
        else:
            # Siamese mode
            images_2018 = batch['image_2018'].to(device)
            images_2025 = batch['image_2025'].to(device)
            masks = batch['mask'].to(device)
            outputs = model(images_2018, images_2025)

        # Get tile IDs if available
        tile_ids = batch.get('tile_id', [f'tile_{batch_idx:03d}'] * masks.shape[0])

        # Convert outputs to predictions
        preds_binary = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
        targets_binary = masks.cpu().numpy()

        # Store predictions and targets
        for i in range(len(tile_ids)):
            pred = preds_binary[i, 0]  # Remove channel dimension
            target = targets_binary[i] if targets_binary.ndim == 3 else targets_binary[i, 0]
            tile_id = tile_ids[i]

            all_predictions.append(pred)
            all_targets.append(target)
            all_tile_ids.append(tile_id)

            # Compute per-tile metrics
            tile_metrics = Metrics()
            tile_metrics.update(
                torch.sigmoid(outputs[i:i+1]).cpu(),
                masks[i:i+1].cpu()
            )
            tile_result = tile_metrics.compute()
            tile_result['tile_id'] = tile_id
            tile_result['change_ratio'] = target.mean()

            per_tile_results.append(tile_result)

        # Update overall metrics
        overall_metrics.update(outputs, masks)

    # Compute overall metrics
    overall_results = overall_metrics.compute()

    # Save predictions if requested
    if save_predictions and output_dir is not None:
        pred_dir = Path(output_dir) / 'predictions'
        pred_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nSaving predictions to {pred_dir}...")
        for tile_id, pred, target in zip(all_tile_ids, all_predictions, all_targets):
            # Save as numpy arrays
            np.save(pred_dir / f'{tile_id}_pred.npy', pred)
            np.save(pred_dir / f'{tile_id}_target.npy', target)

    return {
        'overall': overall_results,
        'per_tile': per_tile_results,
        'predictions': all_predictions,
        'targets': all_targets,
        'tile_ids': all_tile_ids,
    }


def stratify_results_by_change(per_tile_results):
    """
    Stratify results by change level (low/moderate/high).

    Args:
        per_tile_results: List of per-tile result dictionaries

    Returns:
        Dictionary with stratified metrics
    """
    # Convert to DataFrame for easier processing
    df = pd.DataFrame(per_tile_results)

    # Define change level bins
    df['change_level'] = pd.cut(
        df['change_ratio'],
        bins=[0, 0.05, 0.30, 1.0],
        labels=['low', 'moderate', 'high']
    )

    # Compute metrics per change level
    stratified = {}
    for level in ['low', 'moderate', 'high']:
        level_df = df[df['change_level'] == level]
        if len(level_df) > 0:
            stratified[level] = {
                'count': len(level_df),
                'f1': level_df['f1'].mean(),
                'iou': level_df['iou'].mean(),
                'precision': level_df['precision'].mean(),
                'recall': level_df['recall'].mean(),
                'f1_std': level_df['f1'].std(),
                'iou_std': level_df['iou'].std(),
            }
        else:
            stratified[level] = {'count': 0}

    return stratified


def visualize_predictions(predictions, targets, tile_ids, output_dir, num_samples=6):
    """
    Create visualization comparing predictions vs ground truth.

    Args:
        predictions: List of prediction masks
        targets: List of target masks
        tile_ids: List of tile IDs
        output_dir: Directory to save visualizations
        num_samples: Number of samples to visualize
    """
    viz_dir = Path(output_dir) / 'visualizations'
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Select diverse samples (low, moderate, high change)
    change_ratios = [target.mean() for target in targets]
    indices = np.argsort(change_ratios)

    # Select evenly spaced samples
    sample_indices = indices[::len(indices)//num_samples][:num_samples]

    print(f"\nGenerating visualizations for {len(sample_indices)} samples...")

    for idx in sample_indices:
        tile_id = tile_ids[idx]
        pred = predictions[idx]
        target = targets[idx]

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Ground truth
        axes[0].imshow(target, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title(f'Ground Truth\nChange: {target.mean()*100:.1f}%')
        axes[0].axis('off')

        # Prediction
        axes[1].imshow(pred, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title(f'Prediction')
        axes[1].axis('off')

        # Difference (TP=green, FP=red, FN=blue)
        diff = np.zeros((*pred.shape, 3))
        diff[np.logical_and(pred == 1, target == 1)] = [0, 1, 0]  # TP: green
        diff[np.logical_and(pred == 1, target == 0)] = [1, 0, 0]  # FP: red
        diff[np.logical_and(pred == 0, target == 1)] = [0, 0, 1]  # FN: blue

        axes[2].imshow(diff)
        axes[2].set_title('Difference\n(TP=green, FP=red, FN=blue)')
        axes[2].axis('off')

        plt.suptitle(f'Tile: {tile_id}')
        plt.tight_layout()

        # Save
        plt.savefig(viz_dir / f'{tile_id}_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()

    print(f"Visualizations saved to {viz_dir}")


def create_results_report(results, output_dir, checkpoint_path):
    """
    Create comprehensive results report.

    Args:
        results: Dictionary with evaluation results
        output_dir: Directory to save report
        checkpoint_path: Path to model checkpoint
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Overall results
    overall = results['overall']
    per_tile = results['per_tile']

    # Stratified results
    stratified = stratify_results_by_change(per_tile)

    # Create markdown report
    report_path = output_dir / 'evaluation_report.md'

    with open(report_path, 'w') as f:
        f.write("# Test Set Evaluation Report\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Model Checkpoint**: `{checkpoint_path}`\n\n")
        f.write("---\n\n")

        # Overall metrics
        f.write("## Overall Performance\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| **F1-Score** | {overall['f1']:.4f} |\n")
        f.write(f"| **IoU** | {overall['iou']:.4f} |\n")
        f.write(f"| **Precision** | {overall['precision']:.4f} |\n")
        f.write(f"| **Recall** | {overall['recall']:.4f} |\n")
        f.write(f"| **Accuracy** | {overall['accuracy']:.4f} |\n")
        f.write("\n---\n\n")

        # Stratified results
        f.write("## Performance by Change Level\n\n")
        f.write("| Change Level | Count | F1 | IoU | Precision | Recall |\n")
        f.write("|--------------|-------|----|----|-----------|--------|\n")
        for level in ['low', 'moderate', 'high']:
            if stratified[level]['count'] > 0:
                s = stratified[level]
                f.write(f"| **{level.capitalize()}** (<5%/5-30%/≥30%) | "
                       f"{s['count']} | {s['f1']:.4f} ± {s['iou_std']:.4f} | "
                       f"{s['iou']:.4f} ± {s['iou_std']:.4f} | "
                       f"{s['precision']:.4f} | {s['recall']:.4f} |\n")
        f.write("\n---\n\n")

        # Per-tile results
        f.write("## Per-Tile Results\n\n")
        f.write("| Tile ID | Change % | F1 | IoU | Precision | Recall |\n")
        f.write("|---------|----------|----|----|-----------|--------|\n")

        # Sort by F1 score
        sorted_tiles = sorted(per_tile, key=lambda x: x['f1'], reverse=True)
        for tile in sorted_tiles:
            f.write(f"| {tile['tile_id']} | {tile['change_ratio']*100:.1f}% | "
                   f"{tile['f1']:.4f} | {tile['iou']:.4f} | "
                   f"{tile['precision']:.4f} | {tile['recall']:.4f} |\n")

        f.write("\n---\n\n")

        # Best and worst performing tiles
        f.write("## Analysis\n\n")
        f.write("### Top 3 Best Performing Tiles\n\n")
        for i, tile in enumerate(sorted_tiles[:3], 1):
            f.write(f"{i}. **{tile['tile_id']}** - F1: {tile['f1']:.4f}, "
                   f"IoU: {tile['iou']:.4f}, Change: {tile['change_ratio']*100:.1f}%\n")

        f.write("\n### Top 3 Worst Performing Tiles\n\n")
        for i, tile in enumerate(sorted_tiles[-3:], 1):
            f.write(f"{i}. **{tile['tile_id']}** - F1: {tile['f1']:.4f}, "
                   f"IoU: {tile['iou']:.4f}, Change: {tile['change_ratio']*100:.1f}%\n")

        f.write("\n---\n\n")
        f.write(f"**Report generated**: {datetime.now().isoformat()}\n")

    print(f"\nReport saved to {report_path}")

    # Save results as JSON (convert numpy types to Python types)
    json_path = output_dir / 'results.json'

    # Helper function to convert numpy types to Python types
    def convert_to_python_type(obj):
        if isinstance(obj, dict):
            return {k: convert_to_python_type(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_python_type(v) for v in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    results_json = {
        'overall': convert_to_python_type(overall),
        'per_tile': convert_to_python_type(per_tile),
        'stratified': convert_to_python_type(stratified),
        'checkpoint': str(checkpoint_path),
        'timestamp': datetime.now().isoformat(),
    }

    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)

    print(f"Results JSON saved to {json_path}")

    # Save per-tile results as CSV
    csv_path = output_dir / 'per_tile_results.csv'
    df = pd.DataFrame(per_tile)
    df.to_csv(csv_path, index=False)
    print(f"Per-tile CSV saved to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate baseline model on test set')

    # Model configuration
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (best_model.pth)')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save evaluation results')

    # Data configuration
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for evaluation (default: 1 for per-tile metrics)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--image-size', type=int, default=512,
                        help='Input image size')

    # Evaluation options
    parser.add_argument('--save-predictions', action='store_true', default=True,
                        help='Save prediction masks')
    parser.add_argument('--visualize', action='store_true', default=True,
                        help='Generate visualization plots')
    parser.add_argument('--num-viz-samples', type=int, default=6,
                        help='Number of samples to visualize')

    # Wandb options
    parser.add_argument('--wandb', action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--wandb-run-id', type=str, default=None,
                        help='Wandb run ID to resume (adds test metrics to training run)')
    parser.add_argument('--wandb-project', type=str, default='landtake-detection',
                        help='Weights & Biases project name')
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help='Weights & Biases team/user name (optional)')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("TEST SET EVALUATION")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output directory: {output_dir}")
    print("=" * 80 + "\n")

    # Load checkpoint
    print("Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    config = checkpoint.get('config', {})

    # Extract model configuration from checkpoint
    model_name = config.get('model_name', 'siam_conc')
    encoder_name = config.get('encoder_name', 'resnet50')
    encoder_weights = config.get('encoder_weights', 'imagenet')

    print(f"Model: {model_name}")
    print(f"Encoder: {encoder_name}")
    print(f"Encoder weights: {encoder_weights}\n")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Create model
    print("Creating model...")
    model = create_model(
        model_name,
        encoder_name=encoder_name,
        encoder_weights=None,  # Don't load pretrained weights, using checkpoint
    )

    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"Model loaded from checkpoint\n")

    # Create test dataloader
    print("Loading test dataset...")
    return_separate = model_name in ['siam_diff', 'siam_conc']

    dataloaders = create_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        return_separate_images=return_separate,
    )

    test_loader = dataloaders['test']
    print(f"Test set: {len(test_loader)} batches\n")

    # Initialize wandb if requested
    if args.wandb:
        if args.wandb_run_id:
            # Resume existing run
            print(f"Resuming wandb run: {args.wandb_run_id}")
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                id=args.wandb_run_id,
                resume="must",
            )
            print(f"Resumed run: {wandb_run.url}\n")
        else:
            # Create new run for evaluation
            print("Creating new wandb run for evaluation")
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=f"eval_{Path(args.checkpoint).parent.name}",
                config=config,
                tags=["evaluation"],
            )
            print(f"New run: {wandb_run.url}\n")
    else:
        wandb_run = None

    # Run evaluation
    results = evaluate_model(
        model=model,
        dataloader=test_loader,
        device=device,
        save_predictions=args.save_predictions,
        output_dir=output_dir,
    )

    # Print overall results
    print("\n" + "=" * 80)
    print("OVERALL TEST RESULTS")
    print("=" * 80)
    overall = results['overall']
    print(f"F1-Score:  {overall['f1']:.4f}")
    print(f"IoU:       {overall['iou']:.4f}")
    print(f"Precision: {overall['precision']:.4f}")
    print(f"Recall:    {overall['recall']:.4f}")
    print(f"Accuracy:  {overall['accuracy']:.4f}")
    print("=" * 80 + "\n")

    # Generate visualizations
    if args.visualize:
        visualize_predictions(
            results['predictions'],
            results['targets'],
            results['tile_ids'],
            output_dir,
            num_samples=args.num_viz_samples,
        )

    # Create comprehensive report
    create_results_report(results, output_dir, args.checkpoint)

    # Log to wandb if enabled
    if wandb_run is not None:
        print("\nLogging results to wandb...")

        # Log test metrics
        overall = results['overall']
        stratified = stratify_results_by_change(results['per_tile'])

        wandb_run.log({
            'test/f1': overall['f1'],
            'test/iou': overall['iou'],
            'test/precision': overall['precision'],
            'test/recall': overall['recall'],
            'test/accuracy': overall['accuracy'],
        })

        # Log stratified metrics
        for level in ['low', 'moderate', 'high']:
            if stratified[level]['count'] > 0:
                wandb_run.log({
                    f'test/{level}_f1': stratified[level]['f1'],
                    f'test/{level}_iou': stratified[level]['iou'],
                    f'test/{level}_count': stratified[level]['count'],
                })

        # Log visualizations
        if args.visualize:
            viz_dir = Path(output_dir) / 'visualizations'
            for viz_file in sorted(viz_dir.glob('*.png')):
                wandb_run.log({
                    f"test_viz/{viz_file.stem}": wandb.Image(str(viz_file))
                })

        # Create summary table
        per_tile_table = wandb.Table(
            columns=['tile_id', 'change_ratio', 'f1', 'iou', 'precision', 'recall'],
            data=[[t['tile_id'], t['change_ratio'], t['f1'], t['iou'],
                   t['precision'], t['recall']] for t in results['per_tile']]
        )
        wandb_run.log({'test/per_tile_results': per_tile_table})

        # Finish wandb run
        wandb_run.finish()
        print("Wandb logging complete")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {output_dir}")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
