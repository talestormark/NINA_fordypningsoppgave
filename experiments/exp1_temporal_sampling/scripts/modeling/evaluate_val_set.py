#!/usr/bin/env python3
"""
Validation-set inference for Part I experiments (CV macro recomputation).

Sister script to ``evaluate_test_set.py`` but operates on the validation tiles
of each fold (not the held-out test set). Used to recompute the CV column of
results tables as per-tile macro IoU, matching the metric used in the Test
column and in the statistical tests.

For each fold k of an experiment:
    - Load ``best_model.pth`` from ``outputs_v3/{experiment}_fold{k}/``.
    - Build the validation dataloader for fold k via ``get_dataloaders``.
    - Run inference on each validation tile.
    - Compute per-tile IoU at threshold 0.5.

Output:
    - ``outputs_v3/{experiment}_fold{k}/val_per_tile_iou.csv`` per fold.
    - ``outputs_v3/{experiment}/cv_macro_summary.json`` aggregated.

Usage:
    python evaluate_val_set.py --experiment exp001 --all-folds
    python evaluate_val_set.py --experiment exp001 --fold 0
"""

import sys
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

script_dir = Path(__file__).resolve().parent
PART1_DIR = script_dir.parent.parent
from landtake.paths import REPO_ROOT


from landtake.metrics import Metrics  # noqa: E402
from landtake.models.multitemporal import create_multitemporal_model  # noqa: E402
from landtake.data.multitemporal import (  # noqa: E402
    get_dataloaders,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # this experiment's scripts/ dir
from experiments_v2 import (  # noqa: E402
    V2_OUTPUTS_DIR,
    V2_SENTINEL_DIR,
    V2_MASK_DIR,
    V2_SPLITS_DIR,
    V2_CHANGE_LEVEL_PATH,
)

V3_OUTPUTS_DIR = V2_OUTPUTS_DIR  # outputs_v3


# ---------------------------------------------------------------------------
# Per-tile metrics
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
    model = create_multitemporal_model(
        config["model_name"],
        encoder_name=config["encoder_name"],
        encoder_weights=None,
        in_channels=9,
        classes=1,
        lstm_hidden_dim=config.get("lstm_hidden_dim", 256),
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


def evaluate_val_fold(model, val_loader, device):
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


def main():
    parser = argparse.ArgumentParser(
        description="Re-run validation-set inference for Part I experiments "
        "to compute per-tile macro IoU (CV macro)."
    )
    parser.add_argument("--experiment", type=str, required=True,
                        help="Experiment name (e.g., exp001)")
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

    per_fold_per_tile = {}
    per_fold_mean_iou = {}
    per_fold_aggregated = {}

    for fold in folds:
        print(f"\n{'='*60}\nValidation inference: {args.experiment} fold {fold}\n{'='*60}")

        exp_dir = V3_OUTPUTS_DIR / f"{args.experiment}_fold{fold}"
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
            temporal_sampling=config["temporal_sampling"],
            batch_size=args.batch_size,
            num_workers=4,
            image_size=config.get("image_size", 64),
            output_format="LSTM",
            fold=fold,
            num_folds=args.num_folds,
            seed=config.get("seed", 42),
            sentinel2_dir=V2_SENTINEL_DIR,
            mask_dir=V2_MASK_DIR,
            splits_dir=V2_SPLITS_DIR,
            change_level_path=V2_CHANGE_LEVEL_PATH,
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

    if len(per_fold_mean_iou) > 1:
        per_fold_values = [per_fold_mean_iou[f] for f in sorted(per_fold_mean_iou.keys())]
        cv_macro_mean = float(np.mean(per_fold_values))
        cv_macro_std = float(np.std(per_fold_values))

        print(f"\n{'='*60}\nCV (macro) summary for {args.experiment}\n{'='*60}")
        print(f"  Per-fold mean per-tile IoU: {[f'{v*100:.2f}%' for v in per_fold_values]}")
        print(f"  CV (macro) mean ± std: {cv_macro_mean*100:.2f}% ± {cv_macro_std*100:.2f}%")

        summary_dir = V3_OUTPUTS_DIR / args.experiment
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
