#!/usr/bin/env python3
"""
Training script for Part II spectral/spatial resolution experiments.

Adapted from Part I's train_multitemporal.py with one key change:
the --experiment flag drives everything (data config + input channels),
replacing Part I's --temporal-sampling + hardcoded in_channels=9.

Reuses from baseline/Part I:
- FocalLoss, Metrics (from baseline train.py)
- WandbLogger (from baseline logger.py)
- create_multitemporal_model, count_parameters (from Part I models)
- EXPERIMENT_CONFIGS, get_dataloaders (from Part II dataset.py)
"""

import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime

# ---------------------------------------------------------------------------
# Path setup for cross-part imports
# ---------------------------------------------------------------------------
import importlib.util

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
PART1_DIR = REPO_ROOT / "PART1_multi_temporal_experiments"
PART2_DIR = REPO_ROOT / "PART2_spectral_spatial_resolution_experiments"

# 1. Baseline utilities — needs scripts/modeling on sys.path for internal imports
sys.path.insert(0, str(REPO_ROOT / "scripts" / "modeling"))
from train import FocalLoss, DiceLoss, FocalDiceLoss, Metrics
from logger import WandbLogger

# 2. Part I models — needs Part I modeling dir on sys.path (for convlstm import)
sys.path.insert(0, str(PART1_DIR / "scripts" / "modeling"))
from models_multitemporal import create_multitemporal_model, count_parameters

# 3. Part II dataset — use importlib to avoid collision with baseline dataset.py
_p2_spec = importlib.util.spec_from_file_location(
    "p2_dataset", PART2_DIR / "scripts" / "data_preparation" / "dataset.py"
)
_p2_dataset = importlib.util.module_from_spec(_p2_spec)
_p2_spec.loader.exec_module(_p2_dataset)
EXPERIMENT_CONFIGS = _p2_dataset.EXPERIMENT_CONFIGS
get_dataloaders = _p2_dataset.get_dataloaders


# ---------------------------------------------------------------------------
# Training & validation loops (identical to Part I)
# ---------------------------------------------------------------------------


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, total_epochs, accumulation_steps=1):
    """Train for one epoch with optional gradient accumulation."""
    model.train()
    metrics = Metrics()
    total_loss = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs} [Train]")
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)  # (B, T, C, H, W)
        masks = batch['mask'].to(device)    # (B, H, W)

        outputs = model(images)  # (B, 1, H, W)

        if torch.isnan(outputs).any():
            print(f"\nWARNING: NaN detected in model outputs!")
            print(f"  Input stats: min={images.min():.3f}, max={images.max():.3f}, mean={images.mean():.3f}")

        loss = criterion(outputs, masks) / accumulation_steps

        if torch.isnan(loss):
            print(f"\nWARNING: NaN detected in loss!")
            print(f"  Outputs contain NaN: {torch.isnan(outputs).any()}")

        loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps
        metrics.update(outputs.detach(), masks)
        pbar.set_postfix({'loss': f'{loss.item() * accumulation_steps:.4f}'})

    avg_loss = total_loss / len(dataloader)
    epoch_metrics = metrics.compute()
    epoch_metrics['loss'] = avg_loss
    return epoch_metrics


def validate(model, dataloader, criterion, device, epoch, total_epochs):
    """Validate model."""
    model.eval()
    metrics = Metrics()
    total_loss = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs} [Val]  ")

    with torch.no_grad():
        for batch in pbar:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            total_loss += loss.item()
            metrics.update(outputs, masks)
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(dataloader)
    epoch_metrics = metrics.compute()
    epoch_metrics['loss'] = avg_loss
    return epoch_metrics


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def train(args):
    """Main training function."""
    # Reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Derive in_channels from experiment config
    cfg = EXPERIMENT_CONFIGS[args.experiment]
    T, C, H_exp, W_exp = cfg["expected_shape"]
    print(f"\nExperiment: {args.experiment}")
    print(f"  Expected shape: T={T}, C={C}, H={H_exp}, W={W_exp}")

    # Create model
    print(f"\nCreating model: {args.model_name}")
    model = create_multitemporal_model(
        args.model_name,
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        in_channels=C,
        classes=1,
        lstm_hidden_dim=args.lstm_hidden_dim,
        lstm_num_layers=args.lstm_num_layers,
        convlstm_kernel_size=args.convlstm_kernel_size,
        skip_aggregation=args.skip_aggregation,
    )
    model = model.to(device)

    param_stats = count_parameters(model)
    print(f"Model parameters: {param_stats['total_millions']:.2f}M")
    print(f"  Trainable: {param_stats['trainable_millions']:.2f}M")
    print(f"  Input channels: {C}")

    # Dataloaders
    print("\nCreating dataloaders...")
    print(f"  Experiment: {args.experiment}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Accumulation steps: {args.accumulation_steps}")
    effective_batch_size = args.batch_size * args.accumulation_steps
    print(f"  Effective batch size: {effective_batch_size}")
    print(f"  Image size: {args.image_size}")

    dataloaders = get_dataloaders(
        experiment=args.experiment,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        fold=args.fold,
        num_folds=args.num_folds,
        seed=args.seed,
    )

    train_loader = dataloaders['train']
    val_loader = dataloaders['val']

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    # Loss function
    if args.loss == 'focal':
        criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
        print(f"\nLoss: Focal Loss (alpha={args.focal_alpha}, gamma={args.focal_gamma})")
    elif args.loss == 'focal_dice':
        criterion = FocalDiceLoss(
            focal_alpha=args.focal_alpha, focal_gamma=args.focal_gamma,
            lambda_focal=args.lambda_focal, lambda_dice=args.lambda_dice,
        )
        print(f"\nLoss: Focal+Dice (alpha={args.focal_alpha}, gamma={args.focal_gamma}, "
              f"lambda_focal={args.lambda_focal}, lambda_dice={args.lambda_dice})")
    elif args.loss == 'bce':
        criterion = nn.BCEWithLogitsLoss()
        print("\nLoss: Binary Cross Entropy")
    else:
        raise ValueError(f"Unknown loss: {args.loss}")

    # Optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        print(f"Optimizer: SGD (lr={args.lr}, momentum={args.momentum}, weight_decay={args.weight_decay})")
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        print(f"Optimizer: AdamW (lr={args.lr}, weight_decay={args.weight_decay})")
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    # LR scheduler (with optional linear warmup)
    if args.scheduler == 'linear':
        main_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.01,
            total_iters=args.epochs,
        )
        sched_desc = f"Linear decay (1.0 -> 0.01 over {args.epochs} epochs)"
    elif args.scheduler == 'cosine':
        main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs - args.warmup_epochs,
            eta_min=args.lr * 0.01,
        )
        sched_desc = f"Cosine annealing (T_max={args.epochs - args.warmup_epochs}, eta_min={args.lr * 0.01})"
    else:
        raise ValueError(f"Unknown scheduler: {args.scheduler}")

    if args.warmup_epochs > 0:
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-3,  # start at lr * 0.001
            end_factor=1.0,
            total_iters=args.warmup_epochs,
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[args.warmup_epochs],
        )
        print(f"LR Scheduler: {args.warmup_epochs}-epoch linear warmup -> {sched_desc}")
    else:
        scheduler = main_scheduler
        print(f"LR Scheduler: {sched_desc}")

    # WandB logger
    # Derive block tag from experiment name (e.g., "A3_s2_9band" -> "block_A")
    block_tag = f"block_{args.experiment[0]}"

    name_parts = [args.model_name, args.experiment, args.encoder_name, f"seed{args.seed}"]
    if args.fold is not None:
        name_parts.append(f"fold{args.fold}")

    run_name = "_".join(name_parts)
    tags = [
        args.model_name,
        args.experiment,
        block_tag,
        args.encoder_name,
        f"seed{args.seed}",
    ]
    if args.fold is not None:
        tags.extend([f"fold{args.fold}", f"{args.num_folds}fold_cv"])

    logger = WandbLogger(
        project=args.wandb_project,
        name=run_name,
        config=vars(args),
        entity=args.wandb_entity,
        enabled=args.wandb,
        tags=tags,
    )

    if args.wandb:
        print(f"\nWandB: Enabled (project={args.wandb_project})")
    else:
        print("\nWandB: Disabled")

    # Resume from checkpoint
    start_epoch = 1
    best_val_iou = 0.0
    history = {'train': [], 'val': []}

    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_iou = checkpoint.get('best_val_iou', 0.0)
        print(f"  Resuming from epoch {start_epoch}")
        print(f"  Best validation IoU: {best_val_iou:.4f}")

    # Save configuration
    config = vars(args)
    config['timestamp'] = datetime.now().isoformat()
    config['device'] = str(device)
    config['model_parameters'] = param_stats
    config['effective_batch_size'] = effective_batch_size
    config['in_channels'] = C
    config['num_timesteps'] = T

    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    best_epoch = 0

    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)

    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 80)

        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.epochs,
            accumulation_steps=args.accumulation_steps,
        )

        val_metrics = validate(
            model, val_loader, criterion, device, epoch, args.epochs,
        )

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        print(f"  Train - Loss: {train_metrics['loss']:.4f}, IoU: {train_metrics['iou']:.4f}, "
              f"F1: {train_metrics['f1']:.4f}, Prec: {train_metrics['precision']:.4f}, Rec: {train_metrics['recall']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, IoU: {val_metrics['iou']:.4f}, "
              f"F1: {val_metrics['f1']:.4f}, Prec: {val_metrics['precision']:.4f}, Rec: {val_metrics['recall']:.4f}")
        print(f"  LR: {current_lr:.6f}")

        logger.log_epoch(epoch, train_metrics, val_metrics, current_lr)

        history['train'].append(train_metrics)
        history['val'].append(val_metrics)

        with open(output_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

        if val_metrics['iou'] > best_val_iou:
            best_val_iou = val_metrics['iou']
            best_epoch = epoch
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_iou': best_val_iou,
                'val_metrics': val_metrics,
                'config': config,
            }
            torch.save(checkpoint, output_dir / 'best_model.pth')
            print(f"  > Saved best model (IoU: {best_val_iou:.4f})")

    # Save final model
    final_checkpoint = {
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_iou': best_val_iou,
        'val_metrics': val_metrics,
        'config': config,
    }
    torch.save(final_checkpoint, output_dir / 'final_model.pth')

    logger.finish()

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best validation IoU: {best_val_iou:.4f} (epoch {best_epoch})")
    print(f"Final epoch: {epoch}")
    print(f"Checkpoints saved to: {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description='Train Part II spectral/spatial resolution experiments'
    )

    # Experiment (required)
    parser.add_argument('--experiment', type=str, required=True,
                        choices=list(EXPERIMENT_CONFIGS.keys()),
                        help='Experiment key (e.g., A3_s2_9band)')

    # Model configuration
    parser.add_argument('--model-name', type=str, default='late_fusion_pool',
                        choices=['lstm_unet', 'early_fusion_unet', 'late_fusion_concat',
                                 'late_fusion_pool', 'conv3d_fusion'],
                        help='Model architecture (default: late_fusion_pool)')
    parser.add_argument('--encoder-name', type=str, default='resnet50',
                        help='Encoder backbone')
    parser.add_argument('--encoder-weights', type=str, default='imagenet',
                        help='Pretrained encoder weights')

    # LSTM-specific (for sanity check only)
    parser.add_argument('--lstm-hidden-dim', type=int, default=512,
                        help='ConvLSTM hidden dimension')
    parser.add_argument('--lstm-num-layers', type=int, default=2,
                        help='Number of ConvLSTM layers')
    parser.add_argument('--convlstm-kernel-size', type=int, default=3,
                        choices=[1, 3],
                        help='ConvLSTM spatial kernel size')
    parser.add_argument('--skip-aggregation', type=str, default='max',
                        choices=['max', 'mean', 'last'],
                        help='Skip connection temporal aggregation')

    # Data configuration
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--accumulation-steps', type=int, default=1,
                        help='Gradient accumulation steps')
    parser.add_argument('--image-size', type=int, default=64,
                        help='Crop size (default: 64 for Part II)')
    parser.add_argument('--num-workers', type=int, default=4)

    # Training configuration (Part II defaults)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (only used with --optimizer sgd)')
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['sgd', 'adamw'])
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['linear', 'cosine'])
    parser.add_argument('--warmup-epochs', type=int, default=5,
                        help='Linear LR warmup epochs (0 to disable, default: 5)')

    # Loss configuration
    parser.add_argument('--loss', type=str, default='focal_dice',
                        choices=['focal', 'focal_dice', 'bce'])
    parser.add_argument('--focal-alpha', type=float, default=0.75)
    parser.add_argument('--focal-gamma', type=float, default=2.0)
    parser.add_argument('--lambda-focal', type=float, default=1.0)
    parser.add_argument('--lambda-dice', type=float, default=1.0)

    # Output and logging
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (auto-generated if not specified)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    # WandB
    parser.add_argument('--wandb', action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='landtake-spectral-spatial')
    parser.add_argument('--wandb-entity', type=str, default=None)

    # K-Fold CV
    parser.add_argument('--fold', type=int, default=None,
                        help='Fold index (0 to num-folds-1)')
    parser.add_argument('--num-folds', type=int, default=5)

    # Data overrides
    parser.add_argument('--splits-csv', type=str, default=None,
                        help='Override splits CSV path (default: use dataset.py SPLITS_CSV)')

    args = parser.parse_args()

    # Override splits CSV if specified
    if args.splits_csv:
        _p2_dataset.SPLITS_CSV = Path(args.splits_csv)

    # Auto-generate output directory if not specified
    if args.output_dir is None:
        fold_suffix = f"_fold{args.fold}" if args.fold is not None else ""
        args.output_dir = str(
            PART2_DIR / "outputs" / "experiments" / f"{args.experiment}{fold_suffix}"
        )

    train(args)


if __name__ == "__main__":
    main()
