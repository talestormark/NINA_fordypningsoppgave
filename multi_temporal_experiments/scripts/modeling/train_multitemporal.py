#!/usr/bin/env python3
"""
Training script for multi-temporal land-take detection models.

Implements LSTM-UNet training on Sentinel-2 time series data.

Reuses from baseline:
- Focal Loss for class imbalance
- SGD optimizer with momentum and linear LR decay
- F1-score, IoU, precision, recall metrics
- Checkpointing and WandB logging

New for multi-temporal:
- MultiTemporalSentinel2Dataset with temporal sampling modes
- LSTM-UNet architecture
- (B, T, C, H, W) input format
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

# Add paths for imports
script_dir = Path(__file__).resolve().parent
parent_dir = script_dir.parent.parent
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(parent_dir.parent))

# Import baseline utilities (reuse as-is)
sys.path.insert(0, str(parent_dir.parent / "scripts" / "modeling"))
from train import FocalLoss, Metrics
from logger import WandbLogger, create_run_name, create_tags

# Import multi-temporal modules
from models_multitemporal import create_multitemporal_model, count_parameters
from multi_temporal_experiments.scripts.data_preparation.dataset_multitemporal import get_dataloaders
from multi_temporal_experiments.config import MT_EXPERIMENTS_DIR


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, total_epochs):
    """
    Train for one epoch.

    Args:
        model: LSTM-UNet model
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device (cuda/cpu)
        epoch: Current epoch
        total_epochs: Total epochs

    Returns:
        dict: Training metrics
    """
    model.train()
    metrics = Metrics()
    total_loss = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs} [Train]")

    for batch in pbar:
        # Get data
        images = batch['image'].to(device)  # (B, T, C, H, W)
        masks = batch['mask'].to(device)    # (B, H, W)

        # Forward pass - model handles temporal dimension
        outputs = model(images)  # (B, 1, H, W)

        # Debug: Check for NaN in outputs
        if torch.isnan(outputs).any():
            print(f"\nWARNING: NaN detected in model outputs!")
            print(f"  Input stats: min={images.min():.3f}, max={images.max():.3f}, mean={images.mean():.3f}")
            print(f"  Output stats: min={outputs.min():.3f}, max={outputs.max():.3f}, mean={outputs.mean():.3f}")
            print(f"  NaN count: {torch.isnan(outputs).sum().item()}/{outputs.numel()}")

        # Compute loss
        loss = criterion(outputs, masks)

        # Debug: Check for NaN in loss
        if torch.isnan(loss):
            print(f"\nWARNING: NaN detected in loss!")
            print(f"  Outputs contain NaN: {torch.isnan(outputs).any()}")
            print(f"  Masks contain NaN: {torch.isnan(masks).any()}")
            print(f"  Mask stats: min={masks.min():.3f}, max={masks.max():.3f}, unique={torch.unique(masks)}")

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Update metrics
        total_loss += loss.item()
        metrics.update(outputs.detach(), masks)

        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    # Compute epoch metrics
    avg_loss = total_loss / len(dataloader)
    epoch_metrics = metrics.compute()
    epoch_metrics['loss'] = avg_loss

    return epoch_metrics


def validate(model, dataloader, criterion, device, epoch, total_epochs):
    """
    Validate model.

    Args:
        model: LSTM-UNet model
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device (cuda/cpu)
        epoch: Current epoch
        total_epochs: Total epochs

    Returns:
        dict: Validation metrics
    """
    model.eval()
    metrics = Metrics()
    total_loss = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs} [Val]  ")

    with torch.no_grad():
        for batch in pbar:
            # Get data
            images = batch['image'].to(device)  # (B, T, C, H, W)
            masks = batch['mask'].to(device)    # (B, H, W)

            # Forward pass
            outputs = model(images)  # (B, 1, H, W)

            # Compute loss
            loss = criterion(outputs, masks)

            # Update metrics
            total_loss += loss.item()
            metrics.update(outputs, masks)

            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    # Compute epoch metrics
    avg_loss = total_loss / len(dataloader)
    epoch_metrics = metrics.compute()
    epoch_metrics['loss'] = avg_loss

    return epoch_metrics


def train(args):
    """
    Main training function.

    Args:
        args: Command-line arguments
    """
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create model
    print(f"\nCreating model: {args.model_name}")
    model = create_multitemporal_model(
        args.model_name,
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        in_channels=9,  # Sentinel-2 bands
        classes=1,  # Binary segmentation
        lstm_hidden_dim=args.lstm_hidden_dim,
        lstm_num_layers=args.lstm_num_layers,
        skip_aggregation=args.skip_aggregation,
    )
    model = model.to(device)

    # Count parameters
    param_stats = count_parameters(model)
    print(f"Model parameters: {param_stats['total_millions']:.2f}M")
    print(f"  Trainable: {param_stats['trainable_millions']:.2f}M")

    # Create dataloaders
    print("\nCreating dataloaders...")
    print(f"  Temporal sampling: {args.temporal_sampling}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Image size: {args.image_size}")

    dataloaders = get_dataloaders(
        temporal_sampling=args.temporal_sampling,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        output_format="LSTM",  # (B, T, C, H, W)
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

    # Learning rate scheduler
    if args.scheduler == 'linear':
        scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.01,  # Decay to 1% of initial LR
            total_iters=args.epochs,
        )
        print(f"LR Scheduler: Linear decay (1.0 → 0.01 over {args.epochs} epochs)")
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 0.01,  # Minimum LR = 1% of initial
        )
        print(f"LR Scheduler: Cosine annealing (T_max={args.epochs}, eta_min={args.lr * 0.01})")
    else:
        raise ValueError(f"Unknown scheduler: {args.scheduler}")

    # WandB logger
    if args.fold is not None:
        run_name = f"{args.model_name}_{args.temporal_sampling}_{args.encoder_name}_seed{args.seed}_fold{args.fold}"
        tags = [
            args.model_name,
            args.temporal_sampling,
            args.encoder_name,
            f"seed{args.seed}",
            f"fold{args.fold}",
            f"{args.num_folds}fold_cv",
        ]
    else:
        run_name = f"{args.model_name}_{args.temporal_sampling}_{args.encoder_name}_seed{args.seed}"
        tags = [
            args.model_name,
            args.temporal_sampling,
            args.encoder_name,
            f"seed{args.seed}",
        ]

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

    # Resume from checkpoint if specified
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

    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)

    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 80)

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.epochs
        )

        # Validate
        val_metrics = validate(
            model, val_loader, criterion, device, epoch, args.epochs
        )

        # Step scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Print metrics
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, IoU: {train_metrics['iou']:.4f}, "
              f"F1: {train_metrics['f1']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, IoU: {val_metrics['iou']:.4f}, "
              f"F1: {val_metrics['f1']:.4f}")
        print(f"  LR: {current_lr:.6f}")

        # Log to wandb
        logger.log_epoch(epoch, train_metrics, val_metrics, current_lr)

        # Save history
        history['train'].append(train_metrics)
        history['val'].append(val_metrics)

        with open(output_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

        # Save best model
        if val_metrics['iou'] > best_val_iou:
            best_val_iou = val_metrics['iou']
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
            print(f"  ✓ Saved best model (IoU: {best_val_iou:.4f})")

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

    # Finish wandb
    logger.finish()

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best validation IoU: {best_val_iou:.4f}")
    print(f"Checkpoints saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Train multi-temporal land-take detection models'
    )

    # Model configuration
    parser.add_argument('--model-name', type=str, default='lstm_unet',
                        choices=['lstm_unet', 'unet_3d'],
                        help='Model architecture')
    parser.add_argument('--encoder-name', type=str, default='resnet50',
                        help='Encoder architecture (for LSTM-UNet)')
    parser.add_argument('--encoder-weights', type=str, default='imagenet',
                        help='Pretrained encoder weights')

    # LSTM-specific parameters
    parser.add_argument('--lstm-hidden-dim', type=int, default=512,
                        help='ConvLSTM hidden dimension')
    parser.add_argument('--lstm-num-layers', type=int, default=2,
                        help='Number of ConvLSTM layers')
    parser.add_argument('--skip-aggregation', type=str, default='max',
                        choices=['max', 'mean', 'last'],
                        help='Skip connection temporal aggregation')

    # Temporal sampling
    parser.add_argument('--temporal-sampling', type=str, default='annual',
                        choices=['bi_temporal', 'annual', 'quarterly'],
                        help='Temporal sampling mode')

    # Data configuration
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--image-size', type=int, default=512,
                        help='Image size (square)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')

    # Training configuration
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='Weight decay (L2 regularization)')
    parser.add_argument('--optimizer', type=str, default='sgd',
                        choices=['sgd', 'adamw'],
                        help='Optimizer type')
    parser.add_argument('--scheduler', type=str, default='linear',
                        choices=['linear', 'cosine'],
                        help='Learning rate scheduler type')

    # Loss configuration
    parser.add_argument('--loss', type=str, default='focal',
                        choices=['focal', 'bce'],
                        help='Loss function')
    parser.add_argument('--focal-alpha', type=float, default=0.25,
                        help='Focal loss alpha parameter')
    parser.add_argument('--focal-gamma', type=float, default=2.0,
                        help='Focal loss gamma parameter')

    # Output and logging
    parser.add_argument('--output-dir', type=str,
                        default=str(MT_EXPERIMENTS_DIR / "outputs" / "experiments"),
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    # WandB logging
    parser.add_argument('--wandb', action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='landtake-multitemporal',
                        help='WandB project name')
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help='WandB entity (team name)')

    # K-Fold Cross-Validation
    parser.add_argument('--fold', type=int, default=None,
                        help='Fold index for k-fold CV (0 to num-folds-1). If None, uses original split.')
    parser.add_argument('--num-folds', type=int, default=5,
                        help='Number of folds for k-fold CV (default: 5)')

    args = parser.parse_args()

    # Train
    train(args)


if __name__ == "__main__":
    main()
