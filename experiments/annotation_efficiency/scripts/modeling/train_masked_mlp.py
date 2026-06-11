#!/usr/bin/env python3
"""
E5: Train per-pixel MLP on AlphaEarth with masked loss (sparse labels).

Identical training recipe to E4 (masked focal+dice U-Net). The only change is
the model: a stack of 1x1 convolutions, mathematically equivalent to applying
an MLP to each pixel independently. No spatial receptive field. Used to
isolate the contribution of spatial context in the U-Net vs RF comparison.

Output dir convention: outputs/E5_{experiment}_mlp_sparse/fold{N}/
"""

import sys
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import importlib.util

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[4]
AE_DIR = Path(__file__).resolve().parents[2]

from landtake.losses import FocalDiceLoss
from landtake.metrics import Metrics
from landtake.logger import WandbLogger  # noqa: F401

from landtake.models.multitemporal import create_multitemporal_model, count_parameters

# Reuse the masked loss, train_one_epoch, validate from train_masked_unet
sys.path.insert(0, str(AE_DIR / "scripts" / "modeling"))
from train_masked_unet import MaskedFocalDiceLoss, train_one_epoch, validate

# Part 2 dataset
from landtake.data.spectral import EXPERIMENT_CONFIGS, get_dataloaders


def main():
    parser = argparse.ArgumentParser(description="E5: Masked-loss per-pixel MLP on sparse labels")
    parser.add_argument('--experiment', type=str, default='D2_alphaearth',
                        choices=list(EXPERIMENT_CONFIGS.keys()),
                        help='Experiment config (default: D2_alphaearth)')
    parser.add_argument('--fold', type=int, required=True, help='Fold index (0-4)')
    parser.add_argument('--num-folds', type=int, default=5)
    parser.add_argument('--sparse-labels', type=str, required=True)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--num-layers', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        exp_name = f"E5_{args.experiment}_mlp_sparse"
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

    # Dataloaders (same as E4)
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

    # Per-pixel MLP model
    cfg = EXPERIMENT_CONFIGS[args.experiment]
    T, C, H_exp, W_exp = cfg["expected_shape"]
    print(f"Model: PerPixelMLP, in_channels={C} (per timestep), T={T}, hidden_dim={args.hidden_dim}, num_layers={args.num_layers}")

    model = create_multitemporal_model(
        "per_pixel_mlp",
        in_channels=C,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        classes=1,
    )
    model = model.to(device)
    params = count_parameters(model)
    print(f"Parameters: {params['total_millions']:.2f}M")

    # Loss
    criterion_masked = MaskedFocalDiceLoss(
        focal_alpha=0.75, focal_gamma=2.0,
        lambda_focal=1.0, lambda_dice=1.0,
    )
    criterion_dense = FocalDiceLoss(
        focal_alpha=0.75, focal_gamma=2.0,
        lambda_focal=1.0, lambda_dice=1.0,
    )

    # Optimizer + scheduler (same as E4)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Config
    config = {
        "experiment": f"E5_{args.experiment}_mlp_sparse",
        "model_name": "per_pixel_mlp",
        "in_channels": C,
        "num_timesteps": T,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
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

    # Training loop (same as E4)
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

        if val_metrics['iou'] > best_val_iou:
            best_val_iou = val_metrics['iou']
            torch.save(model.state_dict(), output_dir / "best_model.pth")

        if epoch % 50 == 0 or epoch == 1:
            print(f"  Epoch {epoch}: train_iou={train_metrics['iou']:.4f} "
                  f"val_iou={val_metrics['iou']:.4f} (best={best_val_iou:.4f})")

    torch.save(model.state_dict(), output_dir / "final_model.pth")
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest val IoU: {best_val_iou*100:.1f}%")
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
