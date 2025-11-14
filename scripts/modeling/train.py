#!/usr/bin/env python3
"""
Training script for baseline land-take detection models.

Implements:
- Focal Loss for class imbalance
- SGD optimizer with momentum and linear LR decay
- F1-score, IoU, precision, recall metrics
- Checkpointing (save best model by validation IoU)
- Training on VHR 6-channel bi-temporal imagery
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

# Import local modules
from models import create_model, count_parameters
from dataset import create_dataloaders
from logger import WandbLogger, create_run_name, create_tags


class FocalLoss(nn.Module):
    """
    Focal Loss for binary segmentation with class imbalance.

    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Weighting factor for positive class (0-1)
        gamma: Focusing parameter (gamma >= 0)
        reduction: 'mean', 'sum', or 'none'

    Reference: Lin et al. "Focal Loss for Dense Object Detection" (2017)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Model predictions (logits), shape (B, 1, H, W)
            targets: Ground truth labels, shape (B, H, W) or (B, 1, H, W)

        Returns:
            Focal loss value
        """
        # Ensure targets have same shape as inputs
        if targets.ndim == 3:
            targets = targets.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)

        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(inputs)

        # Calculate binary cross entropy
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )

        # Calculate p_t
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Calculate focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Calculate alpha weight
        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Apply focal loss formula
        focal_loss = alpha_weight * focal_weight * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class Metrics:
    """
    Compute segmentation metrics: F1-score, IoU, Precision, Recall.

    All metrics computed at pixel level for binary segmentation.
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        """Reset accumulated metrics."""
        self.tp = 0  # True positives
        self.fp = 0  # False positives
        self.tn = 0  # True negatives
        self.fn = 0  # False negatives

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Update metrics with a batch of predictions and targets.

        Args:
            preds: Model predictions (logits), shape (B, 1, H, W)
            targets: Ground truth labels, shape (B, H, W) or (B, 1, H, W)
        """
        # Convert logits to binary predictions
        preds_binary = (torch.sigmoid(preds) > self.threshold).float()

        # Ensure same shape
        if targets.ndim == 3:
            targets = targets.unsqueeze(1)

        # Flatten tensors
        preds_flat = preds_binary.view(-1)
        targets_flat = targets.view(-1)

        # Compute confusion matrix components
        self.tp += ((preds_flat == 1) & (targets_flat == 1)).sum().item()
        self.fp += ((preds_flat == 1) & (targets_flat == 0)).sum().item()
        self.tn += ((preds_flat == 0) & (targets_flat == 0)).sum().item()
        self.fn += ((preds_flat == 0) & (targets_flat == 1)).sum().item()

    def compute(self) -> dict:
        """
        Compute final metrics from accumulated values.

        Returns:
            Dictionary with precision, recall, f1, iou, accuracy
        """
        # Avoid division by zero
        epsilon = 1e-7

        precision = self.tp / (self.tp + self.fp + epsilon)
        recall = self.tp / (self.tp + self.fn + epsilon)
        f1 = 2 * precision * recall / (precision + recall + epsilon)
        iou = self.tp / (self.tp + self.fp + self.fn + epsilon)
        accuracy = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn + epsilon)

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'iou': iou,
            'accuracy': accuracy,
        }


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, total_epochs):
    """
    Train for one epoch.

    Args:
        model: PyTorch model
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        total_epochs: Total number of epochs

    Returns:
        Dictionary with average loss and metrics
    """
    model.train()
    metrics = Metrics()
    total_loss = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs} [Train]")

    for batch in pbar:
        # Get data
        if 'image' in batch:
            # Early Fusion mode
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)

            # Forward pass
            outputs = model(images)
        else:
            # Siamese mode
            images_2018 = batch['image_2018'].to(device)
            images_2025 = batch['image_2025'].to(device)
            masks = batch['mask'].to(device)

            # Forward pass
            outputs = model(images_2018, images_2025)

        # Compute loss
        loss = criterion(outputs, masks)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update metrics
        total_loss += loss.item()
        metrics.update(outputs.detach(), masks)

        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    # Compute average metrics
    avg_loss = total_loss / len(dataloader)
    epoch_metrics = metrics.compute()
    epoch_metrics['loss'] = avg_loss

    return epoch_metrics


@torch.no_grad()
def validate(model, dataloader, criterion, device, epoch, total_epochs):
    """
    Validate model on validation set.

    Args:
        model: PyTorch model
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device to validate on
        epoch: Current epoch number
        total_epochs: Total number of epochs

    Returns:
        Dictionary with average loss and metrics
    """
    model.eval()
    metrics = Metrics()
    total_loss = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs} [Val]  ")

    for batch in pbar:
        # Get data
        if 'image' in batch:
            # Early Fusion mode
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)

            # Forward pass
            outputs = model(images)
        else:
            # Siamese mode
            images_2018 = batch['image_2018'].to(device)
            images_2025 = batch['image_2025'].to(device)
            masks = batch['mask'].to(device)

            # Forward pass
            outputs = model(images_2018, images_2025)

        # Compute loss
        loss = criterion(outputs, masks)

        # Update metrics
        total_loss += loss.item()
        metrics.update(outputs, masks)

        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    # Compute average metrics
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

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save training configuration
    config = vars(args)
    config['timestamp'] = datetime.now().isoformat()
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print("\n" + "=" * 80)
    print("TRAINING CONFIGURATION")
    print("=" * 80)
    for key, value in config.items():
        if key != 'timestamp':
            print(f"{key:20s}: {value}")
    print("=" * 80 + "\n")

    # Initialize wandb logger
    run_name = create_run_name(args.model_name, args.encoder_name, args.seed)
    tags = create_tags(
        model_name=args.model_name,
        encoder_name=args.encoder_name,
        loss=args.loss,
    )

    logger = WandbLogger(
        project=args.wandb_project,
        name=run_name,
        config=config,
        entity=args.wandb_entity,
        enabled=args.wandb,
        tags=tags,
        notes=f"Training {args.model_name} with {args.encoder_name} encoder (seed={args.seed})",
    )
    print()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    # Create dataloaders
    print("Creating dataloaders...")
    return_separate = args.model_name in ['siam_diff', 'siam_conc']

    dataloaders = create_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        return_separate_images=return_separate,
    )

    train_loader = dataloaders['train']
    val_loader = dataloaders['val']

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print()

    # Create model
    print(f"Creating model: {args.model_name}")
    model = create_model(
        args.model_name,
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
    )
    model = model.to(device)

    # Print model info
    params = count_parameters(model)
    print(f"Model: {model.name}")
    print(f"Parameters: {params['trainable_millions']:.2f}M trainable, "
          f"{params['total_millions']:.2f}M total")
    print()

    # Watch model gradients (optional - can be disabled for faster training)
    if args.wandb and args.wandb_watch:
        logger.watch_model(model, log="gradients", log_freq=100)
        print("✓ Wandb watching model gradients")
        print()

    # Create loss function
    if args.loss == 'focal':
        criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
        print(f"Loss: Focal Loss (alpha={args.focal_alpha}, gamma={args.focal_gamma})")
    elif args.loss == 'bce':
        criterion = nn.BCEWithLogitsLoss()
        print("Loss: Binary Cross Entropy with Logits")
    else:
        raise ValueError(f"Unknown loss: {args.loss}")
    print()

    # Create optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    print(f"Optimizer: SGD (lr={args.lr}, momentum={args.momentum}, "
          f"weight_decay={args.weight_decay})")

    # Create learning rate scheduler (linear decay)
    scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.01,  # Decay to 1% of initial LR
        total_iters=args.epochs,
    )
    print(f"LR Schedule: Linear decay over {args.epochs} epochs")
    print()

    # Load checkpoint if resuming
    start_epoch = 1
    best_val_iou = 0.0
    history = []

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_iou = checkpoint.get('best_val_iou', 0.0)

        # Load history if available
        history_path = output_dir / 'history.json'
        if history_path.exists():
            with open(history_path, 'r') as f:
                history = json.load(f)

        print(f"Resuming from epoch {start_epoch}")
        print(f"Best validation IoU so far: {best_val_iou:.4f}")
        print()

    # Training loop
    print("=" * 80)
    print("STARTING TRAINING")
    print("=" * 80 + "\n")

    for epoch in range(start_epoch, args.epochs + 1):
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.epochs
        )

        # Validate
        val_metrics = validate(
            model, val_loader, criterion, device, epoch, args.epochs
        )

        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Log to wandb
        logger.log_epoch(epoch, train_metrics, val_metrics, current_lr)

        # Log metrics
        print(f"\nEpoch {epoch}/{args.epochs}:")
        print(f"  LR: {current_lr:.6f}")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
              f"F1: {train_metrics['f1']:.4f}, IoU: {train_metrics['iou']:.4f}, "
              f"Prec: {train_metrics['precision']:.4f}, Rec: {train_metrics['recall']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
              f"F1: {val_metrics['f1']:.4f}, IoU: {val_metrics['iou']:.4f}, "
              f"Prec: {val_metrics['precision']:.4f}, Rec: {val_metrics['recall']:.4f}")

        # Save history
        history.append({
            'epoch': epoch,
            'lr': current_lr,
            'train': train_metrics,
            'val': val_metrics,
        })

        # Save checkpoint if best validation IoU
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
            print(f"  ✓ Saved checkpoint (best IoU: {best_val_iou:.4f})")

        print()

    # Save final model
    final_checkpoint = {
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
    }
    torch.save(final_checkpoint, output_dir / 'final_model.pth')

    # Save training history
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best validation IoU: {best_val_iou:.4f}")
    print(f"Checkpoints saved to: {output_dir}")
    print()

    # Finish wandb run
    logger.finish()


def main():
    parser = argparse.ArgumentParser(description='Train baseline land-take detection models')

    # Model configuration
    parser.add_argument('--model-name', type=str, default='early_fusion',
                        choices=['early_fusion', 'siam_diff', 'siam_conc'],
                        help='Model architecture')
    parser.add_argument('--encoder-name', type=str, default='resnet50',
                        help='Encoder architecture')
    parser.add_argument('--encoder-weights', type=str, default='imagenet',
                        help='Pretrained weights for encoder (imagenet or None)')

    # Data configuration
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--image-size', type=int, default=512,
                        help='Input image size (square)')
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

    # Loss configuration
    parser.add_argument('--loss', type=str, default='focal',
                        choices=['focal', 'bce'],
                        help='Loss function')
    parser.add_argument('--focal-alpha', type=float, default=0.25,
                        help='Focal loss alpha parameter')
    parser.add_argument('--focal-gamma', type=float, default=2.0,
                        help='Focal loss gamma parameter')

    # Output configuration
    parser.add_argument('--output-dir', type=str, default='outputs/training',
                        help='Directory to save checkpoints and logs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from (e.g., outputs/training/run/best_model.pth)')

    # Wandb logging
    parser.add_argument('--wandb', action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='landtake-detection',
                        help='Weights & Biases project name')
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help='Weights & Biases team/user name (optional)')
    parser.add_argument('--wandb-watch', action='store_true',
                        help='Watch model gradients in wandb (adds overhead)')

    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()
