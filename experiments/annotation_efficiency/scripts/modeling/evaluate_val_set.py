#!/usr/bin/env python3
"""
Validation-set inference for annotation_efficiency masked-loss U-Nets (and MLPs).

Sister script to ``evaluate_test_set.py`` but operates on the validation tiles
of each fold (not the held-out test set). Used to recompute the CV column of
results tables as per-tile macro IoU, matching the metric in the Test column.

For each fold k of an experiment:
    - Load ``best_model.pth`` from ``outputs/<exp>/fold{k}/``.
    - Build the validation dataloader for fold k.
    - Run inference on each validation tile (full mask, not sparse — only
      training is sparse; validation tiles have full annotations).
    - Compute per-tile IoU at threshold 0.5.

Output:
    - ``outputs/<exp>/fold{k}/val_per_tile_iou.csv`` per fold.
    - ``outputs/<exp>/cv_macro_summary.json`` aggregated mean/std across folds.

Usage:
    # One experiment, all folds
    python evaluate_val_set.py --exp E4_D2_alphaearth_sparse_n5

    # Multiple experiments in one job
    python evaluate_val_set.py --exp E4_D2_alphaearth_sparse_n5 \
                                       E4_D2_alphaearth_sparse_n10 \
                                       E4_D2_alphaearth_sparse_n100

    # Override the dataloader experiment (default D2_alphaearth)
    python evaluate_val_set.py --exp E4_A3_s2_9band_sparse \
                               --data-experiment A3_s2_9band
"""

import sys
import argparse
import json
import importlib.util
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[4]
AE_DIR = Path(__file__).resolve().parents[2]

from landtake.metrics import Metrics

from landtake.models.multitemporal import create_multitemporal_model  # noqa: E402

from landtake.data.spectral import EXPERIMENT_CONFIGS, get_dataloaders


def compute_sample_metrics(pred_logits, mask, threshold=0.5):
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


def load_model(checkpoint_path, in_channels, device, config_path=None):
    model_name = "early_fusion_unet"
    extra = {"encoder_name": "resnet50", "encoder_weights": None}
    if config_path is not None and config_path.exists():
        with open(config_path) as f:
            saved = json.load(f)
        model_name = saved.get("model_name", model_name)
        if model_name == "per_pixel_mlp":
            extra = {
                "hidden_dim": saved.get("hidden_dim", 256),
                "num_layers": saved.get("num_layers", 5),
            }

    model = create_multitemporal_model(
        model_name,
        in_channels=in_channels,
        classes=1,
        **extra,
    )
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def evaluate_val_fold(model, val_loader, device):
    metrics = Metrics()
    per_sample = {}

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="  Inference", leave=False):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            refids = batch['refid']

            outputs = model(images)
            metrics.update(outputs.detach(), masks)

            for i, refid in enumerate(refids):
                per_sample[refid] = compute_sample_metrics(outputs[i], masks[i])

    return metrics.compute(), per_sample


def run_experiment(exp_name, data_experiment, num_folds, batch_size, device):
    exp_dir = AE_DIR / "outputs" / exp_name
    if not exp_dir.exists():
        print(f"  SKIP: {exp_dir} not found")
        return None

    cfg = EXPERIMENT_CONFIGS[data_experiment]
    _, in_channels, _, _ = cfg["expected_shape"]
    print(f"  Data config: {data_experiment} (in_channels={in_channels})")

    per_fold_mean_iou = {}
    per_fold_aggregated = {}

    for fold in range(num_folds):
        ckpt = exp_dir / f"fold{fold}" / "best_model.pth"
        config_path = exp_dir / f"fold{fold}" / "config.json"
        if not ckpt.exists():
            print(f"  SKIP fold {fold}: {ckpt} not found")
            continue

        print(f"  --- Fold {fold} ---")
        model = load_model(ckpt, in_channels, device, config_path=config_path)

        loaders = get_dataloaders(
            experiment=data_experiment,
            batch_size=batch_size,
            num_workers=4,
            image_size=64,
            fold=fold,
            num_folds=num_folds,
            seed=42,
        )
        val_loader = loaders['val']

        results, per_sample = evaluate_val_fold(model, val_loader, device)

        per_fold_aggregated[fold] = results

        ious = [m['iou'] for m in per_sample.values()]
        fold_mean = float(np.mean(ious)) if ious else 0.0
        per_fold_mean_iou[fold] = fold_mean

        print(f"    Per-tile macro IoU: {fold_mean*100:.2f}% (n={len(ious)} tiles)")
        print(f"    Aggregate (micro) IoU: {results['iou']*100:.2f}%")

        # Per-fold CSV
        csv_path = exp_dir / f"fold{fold}" / "val_per_tile_iou.csv"
        with open(csv_path, "w") as f:
            f.write("experiment,fold,refid,iou,f1,precision,recall\n")
            for refid in sorted(per_sample.keys()):
                m = per_sample[refid]
                f.write(
                    f"{exp_name},{fold},{refid},"
                    f"{m['iou']:.6f},{m['f1']:.6f},"
                    f"{m['precision']:.6f},{m['recall']:.6f}\n"
                )

        del model
        torch.cuda.empty_cache()

    if len(per_fold_mean_iou) < 1:
        print(f"  No folds evaluated for {exp_name}")
        return None

    per_fold_values = [per_fold_mean_iou[f] for f in sorted(per_fold_mean_iou.keys())]
    cv_macro_mean = float(np.mean(per_fold_values))
    cv_macro_std = float(np.std(per_fold_values))

    print(f"\n  CV (macro) summary for {exp_name}")
    print(f"    Per-fold: {[f'{v*100:.2f}%' for v in per_fold_values]}")
    print(f"    Mean ± std: {cv_macro_mean*100:.2f}% ± {cv_macro_std*100:.2f}%")

    summary = {
        "experiment": exp_name,
        "data_experiment": data_experiment,
        "cv_macro_mean": cv_macro_mean,
        "cv_macro_std": cv_macro_std,
        "per_fold_mean_iou": per_fold_mean_iou,
        "per_fold_aggregated_micro": {
            str(k): {kk: float(vv) for kk, vv in v.items()}
            for k, v in per_fold_aggregated.items()
        },
    }
    summary_path = exp_dir / "cv_macro_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved {summary_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Re-run validation-set inference for annotation_efficiency "
        "experiments to compute per-tile macro IoU."
    )
    parser.add_argument(
        '--exp', nargs='+', required=True,
        help='Experiment output dir name(s) under outputs/'
    )
    parser.add_argument(
        '--data-experiment', default='D2_alphaearth',
        help='EXPERIMENT_CONFIGS key for dataloader (default: D2_alphaearth)'
    )
    parser.add_argument('--num-folds', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=1)
    args = parser.parse_args()

    if args.data_experiment not in EXPERIMENT_CONFIGS:
        raise ValueError(
            f"--data-experiment {args.data_experiment} not in EXPERIMENT_CONFIGS. "
            f"Available: {list(EXPERIMENT_CONFIGS.keys())}"
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    for exp in args.exp:
        print(f"\n{'='*60}\nEvaluating {exp}\n{'='*60}")
        run_experiment(
            exp_name=exp,
            data_experiment=args.data_experiment,
            num_folds=args.num_folds,
            batch_size=args.batch_size,
            device=device,
        )


if __name__ == "__main__":
    main()
