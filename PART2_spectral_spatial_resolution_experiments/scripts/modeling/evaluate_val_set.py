#!/usr/bin/env python3
"""
Validation-set inference for Part II experiments.

Sister script to ``evaluate_test_set.py`` but operates on the validation tiles
of each fold (not the held-out test set). The purpose is to recompute the
CV column of results tables as per-tile macro IoU, so it matches the metric
used in the Test column and in the statistical tests.

For each fold k of an experiment:
    - Load ``best_model.pth`` from ``{experiment}_fold{k}/``.
    - Build the validation dataloader for fold k via ``get_dataloaders``
      (deterministic StratifiedKFold(n_splits=5, shuffle=True, random_state=42)).
    - Run inference on each validation tile.
    - Compute per-tile IoU at threshold 0.5.

The output is:
    - ``{experiment}_fold{k}/val_per_tile_iou.csv`` — one row per refid with
      iou, f1, precision, recall.
    - ``{experiment}/cv_macro_summary.json`` — aggregated mean/std of per-tile
      macro IoU across the five folds.

Usage:
    python evaluate_val_set.py --experiment A3_s2_9band --all-folds
    python evaluate_val_set.py --experiment A3_s2_9band --fold 0
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
# Path setup (mirrors evaluate_test_set.py)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
PART2_DIR = REPO_ROOT / "PART2_spectral_spatial_resolution_experiments"

from landtake.metrics import Metrics  # noqa: E402

from landtake.models.multitemporal import create_multitemporal_model  # noqa: E402

from landtake.data.spectral import EXPERIMENT_CONFIGS, get_dataloaders

EXPERIMENTS_DIR = PART2_DIR / "outputs" / "experiments"


# ---------------------------------------------------------------------------
# Reused from evaluate_test_set.py (kept inline to avoid cross-script imports)
# ---------------------------------------------------------------------------


def compute_sample_metrics(pred_logits: torch.Tensor, mask: torch.Tensor, threshold: float = 0.5):
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
        "iou": iou.item(),
        "f1": f1.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "tp": int(tp.item()),
        "fp": int(fp.item()),
        "fn": int(fn.item()),
        "tn": int(tn.item()),
    }


def load_model_from_checkpoint(checkpoint_path: Path, config: dict, device: torch.device):
    cfg = EXPERIMENT_CONFIGS[config["experiment"]]
    _, C, _, _ = cfg["expected_shape"]

    model = create_multitemporal_model(
        config["model_name"],
        encoder_name=config["encoder_name"],
        encoder_weights=None,
        in_channels=C,
        classes=1,
        lstm_hidden_dim=config.get("lstm_hidden_dim", 512),
        lstm_num_layers=config.get("lstm_num_layers", 2),
        convlstm_kernel_size=config.get("convlstm_kernel_size", 3),
        skip_aggregation=config.get("skip_aggregation", "max"),
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Validation-set evaluation
# ---------------------------------------------------------------------------


def evaluate_val_fold(model, val_loader, device):
    """Run inference on the validation tiles and return per-tile metrics."""
    model.eval()
    metrics = Metrics()
    per_sample = {}

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating val"):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            refids = batch["refid"]

            outputs = model(images)
            metrics.update(outputs.detach(), masks)

            for i, refid in enumerate(refids):
                sample_metrics = compute_sample_metrics(outputs[i], masks[i])
                per_sample[refid] = sample_metrics

    return metrics.compute(), per_sample


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Re-run validation-set inference for Part II experiments "
        "to compute per-tile macro IoU (CV macro)."
    )
    parser.add_argument("--experiment", type=str, required=True,
                        help="Experiment name (e.g., A3_s2_9band)")
    parser.add_argument("--fold", type=int, default=None,
                        help="Specific fold (0-4). Omit for all folds.")
    parser.add_argument("--all-folds", action="store_true",
                        help="Evaluate all folds 0-4")
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    if args.fold is None and not args.all_folds:
        parser.error("Must specify either --fold or --all-folds")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    folds = list(range(args.num_folds)) if args.all_folds else [args.fold]

    per_fold_per_tile = {}   # fold -> {refid: {iou, ...}}
    per_fold_mean_iou = {}   # fold -> mean per-tile IoU across val tiles
    per_fold_aggregated = {} # fold -> aggregate metrics (sanity check, micro IoU)

    for fold in folds:
        print(f"\n{'='*60}\nValidation inference: {args.experiment} fold {fold}\n{'='*60}")

        exp_dir = EXPERIMENTS_DIR / f"{args.experiment}_fold{fold}"
        if not exp_dir.exists():
            print(f"WARNING: missing experiment dir: {exp_dir}")
            continue

        with open(exp_dir / "config.json") as f:
            config = json.load(f)

        checkpoint_path = exp_dir / "best_model.pth"
        if not checkpoint_path.exists():
            print(f"WARNING: missing checkpoint: {checkpoint_path}")
            continue

        print(f"  Loading: {checkpoint_path}")
        model = load_model_from_checkpoint(checkpoint_path, config, device)

        print(f"  Building val dataloader (fold {fold})...")
        dataloaders = get_dataloaders(
            experiment=config["experiment"],
            batch_size=args.batch_size,
            num_workers=4,
            image_size=config.get("image_size", 64),
            fold=fold,
            num_folds=args.num_folds,
            seed=config.get("seed", 42),
        )
        val_loader = dataloaders["val"]
        print(f"  Val samples in fold {fold}: {len(val_loader.dataset)}")

        results, per_sample = evaluate_val_fold(model, val_loader, device)

        per_fold_per_tile[fold] = per_sample
        per_fold_aggregated[fold] = results

        ious = [m["iou"] for m in per_sample.values()]
        fold_mean_iou = float(np.mean(ious)) if ious else 0.0
        per_fold_mean_iou[fold] = fold_mean_iou

        print(f"\n  Fold {fold}:")
        print(f"    Per-tile macro IoU:  {fold_mean_iou*100:.2f}% (over {len(ious)} tiles)")
        print(f"    Aggregate (micro) IoU (sanity, for comparison): {results['iou']*100:.2f}%")

        # Per-fold per-tile CSV
        csv_path = exp_dir / "val_per_tile_iou.csv"
        with open(csv_path, "w") as f:
            f.write("experiment,fold,refid,iou,f1,precision,recall\n")
            for refid in sorted(per_sample.keys()):
                m = per_sample[refid]
                f.write(
                    f"{args.experiment},{fold},{refid},"
                    f"{m['iou']:.6f},{m['f1']:.6f},"
                    f"{m['precision']:.6f},{m['recall']:.6f}\n"
                )
        print(f"    Per-tile CSV: {csv_path}")

    # Aggregated summary across folds
    if len(per_fold_mean_iou) > 1:
        per_fold_values = [per_fold_mean_iou[f] for f in sorted(per_fold_mean_iou.keys())]
        cv_macro_mean = float(np.mean(per_fold_values))
        cv_macro_std = float(np.std(per_fold_values))

        print(f"\n{'='*60}\nCV (macro) summary for {args.experiment}\n{'='*60}")
        print(f"  Per-fold mean per-tile IoU: {[f'{v*100:.2f}%' for v in per_fold_values]}")
        print(f"  CV (macro) mean ± std: {cv_macro_mean*100:.2f}% ± {cv_macro_std*100:.2f}%")

        summary_dir = EXPERIMENTS_DIR / args.experiment
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "experiment": args.experiment,
            "cv_macro_mean": cv_macro_mean,
            "cv_macro_std": cv_macro_std,
            "per_fold_mean_iou": per_fold_mean_iou,
            "per_fold_aggregated_micro": {
                str(k): {kk: float(vv) for kk, vv in v.items()}
                for k, v in per_fold_aggregated.items()
            },
        }
        summary_path = summary_dir / "cv_macro_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
