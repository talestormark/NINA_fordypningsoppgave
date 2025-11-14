#!/usr/bin/env python3
"""
Upload existing training runs to Weights & Biases retroactively.

This script parses existing training outputs (history.json, config.json)
and uploads them to wandb for visualization and comparison.

Usage:
    python3 scripts/modeling/upload_existing_runs.py \
        --project landtake-detection \
        --run-dirs outputs/training/siam_conc_resnet50_*
"""

import argparse
import json
import wandb
from pathlib import Path
from typing import Dict, Any, List
from logger import create_run_name, create_tags


def parse_run_directory(run_dir: Path) -> Dict[str, Any]:
    """
    Parse a training run directory to extract config and history.

    Args:
        run_dir: Path to training run directory

    Returns:
        Dictionary with 'config', 'history', and 'name' keys
    """
    config_file = run_dir / 'config.json'
    history_file = run_dir / 'history.json'

    if not config_file.exists():
        raise FileNotFoundError(f"config.json not found in {run_dir}")
    if not history_file.exists():
        raise FileNotFoundError(f"history.json not found in {run_dir}")

    # Load config
    with open(config_file, 'r') as f:
        config = json.load(f)

    # Load history
    with open(history_file, 'r') as f:
        history = json.load(f)

    # Extract run name from directory or config
    run_name = run_dir.name

    return {
        'config': config,
        'history': history,
        'name': run_name,
        'directory': str(run_dir),
    }


def upload_run(
    run_data: Dict[str, Any],
    project: str,
    entity: str = None,
    dry_run: bool = False,
):
    """
    Upload a single training run to wandb.

    Args:
        run_data: Dictionary with config and history from parse_run_directory()
        project: wandb project name
        entity: wandb team/user name (optional)
        dry_run: If True, print what would be uploaded without actually uploading
    """
    config = run_data['config']
    history = run_data['history']
    run_name = run_data['name']

    print(f"\n{'='*80}")
    print(f"Processing: {run_name}")
    print(f"{'='*80}")

    # Extract key info from config
    model_name = config.get('model_name', 'unknown')
    encoder_name = config.get('encoder_name', 'unknown')
    seed = config.get('seed', 42)

    # Create tags
    tags = create_tags(
        model_name=model_name,
        encoder_name=encoder_name,
        loss=config.get('loss'),
    )

    print(f"Model: {model_name}")
    print(f"Encoder: {encoder_name}")
    print(f"Seed: {seed}")
    print(f"Epochs: {len(history)}")
    print(f"Tags: {tags}")

    if dry_run:
        print("DRY RUN - Would upload:")
        print(f"  Config keys: {list(config.keys())}")
        print(f"  History entries: {len(history)}")
        if history:
            print(f"  First epoch metrics: {list(history[0].keys())}")
        return

    # Initialize wandb run
    run = wandb.init(
        project=project,
        name=run_name,
        config=config,
        entity=entity,
        tags=tags,
        notes=f"Retroactively uploaded from {run_data['directory']}",
    )

    print(f"✓ Initialized wandb run: {run.url}")

    # Upload history epoch by epoch
    for entry in history:
        epoch = entry['epoch']
        lr = entry.get('lr', 0)
        train_metrics = entry.get('train', {})
        val_metrics = entry.get('val', {})

        # Log metrics
        metrics = {
            "epoch": epoch,
            "lr": lr,
            # Training metrics
            "train/loss": train_metrics.get("loss", 0),
            "train/f1": train_metrics.get("f1", 0),
            "train/iou": train_metrics.get("iou", 0),
            "train/precision": train_metrics.get("precision", 0),
            "train/recall": train_metrics.get("recall", 0),
            "train/accuracy": train_metrics.get("accuracy", 0),
            # Validation metrics
            "val/loss": val_metrics.get("loss", 0),
            "val/f1": val_metrics.get("f1", 0),
            "val/iou": val_metrics.get("iou", 0),
            "val/precision": val_metrics.get("precision", 0),
            "val/recall": val_metrics.get("recall", 0),
            "val/accuracy": val_metrics.get("accuracy", 0),
        }

        run.log(metrics, step=epoch)

    # Log the best model checkpoint if it exists
    best_model_path = Path(run_data['directory']) / 'best_model.pth'
    if best_model_path.exists():
        artifact = wandb.Artifact(
            name=f"{run_name}_best_model",
            type="model",
        )
        artifact.add_file(str(best_model_path))
        run.log_artifact(artifact)
        print(f"✓ Uploaded best model checkpoint")

    # Finish run
    run.finish()
    print(f"✓ Completed upload for {run_name}")


def main():
    parser = argparse.ArgumentParser(
        description='Upload existing training runs to Weights & Biases'
    )

    parser.add_argument(
        '--project',
        type=str,
        default='landtake-detection',
        help='wandb project name'
    )
    parser.add_argument(
        '--entity',
        type=str,
        default=None,
        help='wandb team/user name (optional)'
    )
    parser.add_argument(
        '--run-dirs',
        type=str,
        nargs='+',
        required=True,
        help='Paths to training run directories (supports wildcards)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print what would be uploaded without actually uploading'
    )

    args = parser.parse_args()

    # Expand paths (handle wildcards)
    run_dirs = []
    for pattern in args.run_dirs:
        matched = list(Path().glob(pattern))
        if not matched:
            print(f"Warning: No directories matched pattern: {pattern}")
        run_dirs.extend(matched)

    # Filter for directories only
    run_dirs = [d for d in run_dirs if d.is_dir()]

    if not run_dirs:
        print("Error: No valid run directories found")
        return

    print(f"\nFound {len(run_dirs)} run directories to upload")
    print(f"Project: {args.project}")
    if args.dry_run:
        print("DRY RUN MODE - No actual uploads will be made\n")

    # Process each run
    for run_dir in sorted(run_dirs):
        try:
            run_data = parse_run_directory(run_dir)
            upload_run(
                run_data,
                project=args.project,
                entity=args.entity,
                dry_run=args.dry_run,
            )
        except Exception as e:
            print(f"\n✗ Error processing {run_dir}: {e}")
            continue

    print(f"\n{'='*80}")
    print(f"Upload complete!")
    print(f"{'='*80}")
    if not args.dry_run:
        print(f"View your runs at: https://wandb.ai/{args.entity or 'your-username'}/{args.project}")


if __name__ == '__main__':
    main()
