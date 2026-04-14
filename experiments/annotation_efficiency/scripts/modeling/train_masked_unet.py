#!/usr/bin/env python3
"""
E4: Train EarlyFusion U-Net on AlphaEarth with masked loss (sparse labels).

Same architecture and training recipe as D2, but loss is computed ONLY at
the 50 sparse-labeled pixel locations per tile. The U-Net still produces
full-tile predictions — it just learns from sparse supervision.

Validation/test evaluation uses the FULL dense mask (same as D2).

This is a standalone script to avoid modifying the shared Part 2 pipeline.
"""

import sys
import argparse
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import importlib.util

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[4]
PART1_DIR = REPO_ROOT / "PART1_multi_temporal_experiments"
PART2_DIR = REPO_ROOT / "PART2_spectral_spatial_resolution_experiments"
AE_DIR = Path(__file__).resolve().parents[2]  # experiments/annotation_efficiency

sys.path.insert(0, str(REPO_ROOT / "scripts" / "modeling"))
from train import FocalLoss, DiceLoss, Metrics
from logger import WandbLogger

sys.path.insert(0, str(PART1_DIR / "scripts" / "modeling"))
from models_multitemporal import create_multitemporal_model, count_parameters

# Part 2 dataset (via importlib to avoid name collision)
_p2_spec = importlib.util.spec_from_file_location(
    "p2_dataset", PART2_DIR / "scripts" / "data_preparation" / "dataset.py"
)
_p2_dataset = importlib.util.module_from_spec(_p2_spec)
_p2_spec.loader.exec_module(_p2_dataset)
EXPERIMENT_CONFIGS = _p2_dataset.EXPERIMENT_CONFIGS
get_dataloaders = _p2_dataset.get_dataloaders

SPLITS_CSV = REPO_ROOT / "preprocessing" / "outputs" / "splits" / "unified" / "split_info.csv"
EPS = 1e-7


# ---------------------------------------------------------------------------
# Masked FocalDice Loss
# ---------------------------------------------------------------------------

class MaskedFocalDiceLoss(nn.Module):
    """FocalDice loss computed only at labeled pixel locations."""

    def __init__(self, focal_alpha=0.75, focal_gamma=2.0,
                 lambda_focal=1.0, lambda_dice=1.0, smooth=1.0):
        super().__init__()
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.lambda_focal = lambda_focal
        self.lambda_dice = lambda_dice
        self.smooth = smooth

    def forward(self, inputs, targets, label_mask):
        """
        Args:
            inputs: (B, 1, H, W) logits
            targets: (B, H, W) dense mask
            label_mask: (B, H, W) binary — 1 at labeled pixels, 0 elsewhere
        """
        if targets.ndim == 3:
            targets = targets.unsqueeze(1)  # (B, 1, H, W)
        if label_mask.ndim == 3:
            label_mask = label_mask.unsqueeze(1)  # (B, 1, H, W)

        n_labeled = label_mask.sum().clamp(min=1.0)

        # Focal loss (per-pixel, then masked mean)
        probs = torch.sigmoid(inputs)
        bce = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.focal_gamma
        alpha_weight = self.focal_alpha * targets + (1 - self.focal_alpha) * (1 - targets)
        focal_per_pixel = alpha_weight * focal_weight * bce
        focal_loss = (focal_per_pixel * label_mask).sum() / n_labeled

        # Dice loss (on masked pixels only)
        probs_masked = probs * label_mask
        targets_masked = targets * label_mask
        intersection = (probs_masked * targets_masked).sum()
        union = probs_masked.sum() + targets_masked.sum()
        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (union + self.smooth)

        return self.lambda_focal * focal_loss + self.lambda_dice * dice_loss


# ---------------------------------------------------------------------------
# Training loop with masked loss
# ---------------------------------------------------------------------------

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, total_epochs,
                    sparse_labels, tile_label_masks, accumulation_steps=1):
    model.train()
    metrics = Metrics()
    total_loss = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs} [Train]")
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        refids = batch['refid']

        # Build label_mask for this batch
        B, _, H, W = images.shape[0], images.shape[1], images.shape[-2], images.shape[-1]
        label_mask = torch.zeros(B, H, W, device=device)
        for i, refid in enumerate(refids):
            if refid in tile_label_masks:
                coords = tile_label_masks[refid]
                for y, x, _ in coords:
                    if 0 <= y < H and 0 <= x < W:
                        label_mask[i, y, x] = 1.0

        outputs = model(images)
        loss = criterion(outputs, masks, label_mask) / accumulation_steps
        loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps
        # Metrics on ALL pixels (not masked) for monitoring
        metrics.update(outputs.detach(), masks)
        pbar.set_postfix({'loss': f'{loss.item() * accumulation_steps:.4f}'})

    avg_loss = total_loss / len(dataloader)
    epoch_metrics = metrics.compute()
    epoch_metrics['loss'] = avg_loss
    return epoch_metrics


def validate(model, dataloader, criterion_dense, device, epoch, total_epochs):
    """Validate on FULL dense mask (not masked)."""
    model.eval()
    metrics = Metrics()
    total_loss = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs} [Val]  ")
    with torch.no_grad():
        for batch in pbar:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)

            outputs = model(images)
            loss = criterion_dense(outputs, masks)

            total_loss += loss.item()
            metrics.update(outputs, masks)
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(dataloader)
    epoch_metrics = metrics.compute()
    epoch_metrics['loss'] = avg_loss
    return epoch_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="E4: Masked-loss U-Net on sparse labels")
    parser.add_argument('--experiment', type=str, default='D2_alphaearth',
                        choices=list(EXPERIMENT_CONFIGS.keys()),
                        help='Experiment config (default: D2_alphaearth)')
    parser.add_argument('--fold', type=int, required=True, help='Fold index (0-4)')
    parser.add_argument('--num-folds', type=int, default=5)
    parser.add_argument('--sparse-labels', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--wandb', action='store_true')
    args = parser.parse_args()

    if args.output_dir is None:
        exp_name = f"E4_{args.experiment}_sparse"
        args.output_dir = str(AE_DIR / "outputs" / exp_name / f"fold{args.fold}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load sparse labels
    with open(args.sparse_labels) as f:
        sparse_data = json.load(f)
    tile_label_masks = sparse_data["tiles"]
    print(f"Sparse labels: {sparse_data['summary']['total_tiles']} tiles, "
          f"{sparse_data['summary']['total_points']} points")

    # Create dataloaders
    print(f"\nCreating dataloaders ({args.experiment}, fold {args.fold})...")
    dataloaders = get_dataloaders(
        experiment=args.experiment,
        batch_size=args.batch_size,
        num_workers=4,
        image_size=64,
        fold=args.fold,
        num_folds=args.num_folds,
        seed=args.seed,
    )
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']

    # Create model (same as D2)
    cfg = EXPERIMENT_CONFIGS[args.experiment]
    T, C, H_exp, W_exp = cfg["expected_shape"]
    print(f"Model: EarlyFusion, in_channels={C}, T={T}")

    model = create_multitemporal_model(
        "early_fusion_unet",
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=C,
        classes=1,
    )
    model = model.to(device)
    params = count_parameters(model)
    print(f"Parameters: {params['total_millions']:.2f}M")

    # Loss: masked for training, dense for validation
    criterion_masked = MaskedFocalDiceLoss(
        focal_alpha=0.75, focal_gamma=2.0,
        lambda_focal=1.0, lambda_dice=1.0,
    )
    from train import FocalDiceLoss
    criterion_dense = FocalDiceLoss(
        focal_alpha=0.75, focal_gamma=2.0,
        lambda_focal=1.0, lambda_dice=1.0,
    )

    # Optimizer + scheduler (same as D2)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Save config
    config = {
        "experiment": f"E4_{args.experiment}_sparse",
        "model_name": "early_fusion_unet",
        "in_channels": C,
        "num_timesteps": T,
        "encoder_name": "resnet50",
        "loss": "masked_focal_dice",
        "focal_alpha": 0.75,
        "focal_gamma": 2.0,
        "sparse_labels": args.sparse_labels,
        "n_sparse_points": sparse_data["summary"]["total_points"],
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "fold": args.fold,
        "seed": args.seed,
        "device": str(device),
        "parameters": params,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Training loop
    best_val_iou = 0.0
    history = {"train": [], "val": []}

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, criterion_masked, optimizer, device,
            epoch, args.epochs, sparse_data, tile_label_masks,
        )
        val_metrics = validate(model, val_loader, criterion_dense, device, epoch, args.epochs)
        scheduler.step()

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        # Save best model
        if val_metrics['iou'] > best_val_iou:
            best_val_iou = val_metrics['iou']
            torch.save(model.state_dict(), output_dir / "best_model.pth")

        if epoch % 50 == 0 or epoch == 1:
            print(f"  Epoch {epoch}: train_iou={train_metrics['iou']:.4f} "
                  f"val_iou={val_metrics['iou']:.4f} (best={best_val_iou:.4f})")

    # Save final
    torch.save(model.state_dict(), output_dir / "final_model.pth")
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest val IoU: {best_val_iou*100:.1f}%")
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
