#!/usr/bin/env python3
"""
Test-set evaluation for annotation_efficiency masked-loss U-Nets.

Adapted from PART2/scripts/modeling/evaluate_test_set.py. Loads fold checkpoints
from experiments/exp3_annotation/outputs/<exp>/fold{0-4}/best_model.pth,
runs inference on the 40 held-out test tiles using the AlphaEarth data config
(D2_alphaearth in EXPERIMENT_CONFIGS), and saves per-tile predictions plus an
ensemble (probability-averaged) result.

Output schema matches PART2's test_results.json so the per-tile statistical
analysis script (statistical_analysis_persample.py) can load it via --unet-dir.

Usage:
    # Single experiment, all folds + ensemble
    python evaluate_test_set.py --exp E4_D2_alphaearth_sparse_n5

    # Run all three budget variants in one job
    python evaluate_test_set.py --exp E4_D2_alphaearth_sparse_n5 \
                                       E4_D2_alphaearth_sparse_n10 \
                                       E4_D2_alphaearth_sparse_n100
"""

import sys
import argparse
import json
import importlib.util
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
from landtake.paths import REPO_ROOT
AE_DIR = Path(__file__).resolve().parents[2]

from landtake.metrics import Metrics

from landtake.models.multitemporal import create_multitemporal_model  # noqa: E402

# Part 2 dataset (importlib to avoid name collision)
from landtake.data.spectral import EXPERIMENT_CONFIGS, get_dataloaders


# ---------------------------------------------------------------------------
# Per-sample metrics (same as Part 2)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path, in_channels, device, config_path=None):
    """Load checkpoint. Dispatch on saved config['model_name'] when present."""
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


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_fold(model, test_loader, device):
    """Run inference, return aggregated + per-sample metrics + per-tile probs/masks."""
    metrics = Metrics()
    per_sample = {}
    predictions = {}

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="  Inference", leave=False):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            refids = batch['refid']

            outputs = model(images)
            metrics.update(outputs.detach(), masks)

            probs = torch.sigmoid(outputs)
            for i, refid in enumerate(refids):
                per_sample[refid] = compute_sample_metrics(outputs[i], masks[i])
                predictions[refid] = {
                    'prob': probs[i].cpu().numpy(),
                    'mask': masks[i].cpu().numpy(),
                }

    return metrics.compute(), per_sample, predictions


def evaluate_ensemble(all_predictions):
    """Average probabilities across folds, then compute IoU."""
    refids = list(all_predictions[0].keys())

    metrics = Metrics()
    per_sample = {}
    ensemble_probs = {}

    for refid in refids:
        probs = [all_predictions[fold][refid]['prob'] for fold in all_predictions]
        avg_prob = np.mean(probs, axis=0)
        mask = all_predictions[0][refid]['mask']

        eps = 1e-7
        avg_clipped = np.clip(avg_prob, eps, 1 - eps)
        logits = np.log(avg_clipped / (1 - avg_clipped))
        logit_t = torch.from_numpy(logits).float()
        mask_t = torch.from_numpy(mask)

        metrics.update(logit_t.unsqueeze(0), mask_t.unsqueeze(0))
        per_sample[refid] = compute_sample_metrics(logit_t, mask_t)
        ensemble_probs[refid] = avg_prob

    return metrics.compute(), per_sample, ensemble_probs


# ---------------------------------------------------------------------------
# Driver per experiment
# ---------------------------------------------------------------------------

def run_experiment(exp_name, data_experiment, num_folds, batch_size, save_predictions, device):
    exp_dir = AE_DIR / "outputs" / exp_name
    if not exp_dir.exists():
        print(f"  SKIP: {exp_dir} not found")
        return None

    cfg = EXPERIMENT_CONFIGS[data_experiment]
    _, in_channels, _, _ = cfg["expected_shape"]
    print(f"  Data config: {data_experiment} (in_channels={in_channels})")

    all_results = {}
    all_per_sample = {}
    all_predictions = {}

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
        test_loader = loaders['test']

        results, per_sample, predictions = evaluate_fold(model, test_loader, device)

        all_results[fold] = results
        all_per_sample[fold] = per_sample
        all_predictions[fold] = predictions

        print(f"    IoU: {results['iou']*100:.2f}%  (n={len(per_sample)} tiles)")

        del model
        torch.cuda.empty_cache()

    if not all_results:
        print(f"  No folds evaluated for {exp_name}")
        return None

    print(f"  --- Ensemble ---")
    ens_results, ens_per_sample, ens_probs = evaluate_ensemble(all_predictions)
    print(f"    IoU: {ens_results['iou']*100:.2f}%")

    # Build serialisable result
    out = {
        "experiment": exp_name,
        "data_experiment": data_experiment,
        "folds": {
            str(fold): {
                "aggregated": {k: float(v) for k, v in all_results[fold].items()},
                "per_sample": {
                    refid: {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                            for k, v in m.items()}
                    for refid, m in all_per_sample[fold].items()
                },
            }
            for fold in all_results
        },
        "ensemble": {
            "aggregated": {k: float(v) for k, v in ens_results.items()},
            "per_sample": {
                refid: {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                        for k, v in m.items()}
                for refid, m in ens_per_sample.items()
            },
        },
    }

    metrics_names = ['iou', 'f1', 'precision', 'recall']
    for metric in metrics_names:
        values = [all_results[f][metric] for f in all_results]
        out[f"mean_{metric}"] = float(np.mean(values))
        out[f"std_{metric}"] = float(np.std(values))

    out_path = exp_dir / "test_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved {out_path}")

    if save_predictions:
        ens_dir = exp_dir / "ensemble_predictions"
        ens_dir.mkdir(exist_ok=True)
        for refid, prob in ens_probs.items():
            mask = all_predictions[0][refid]['mask']
            np.savez_compressed(
                ens_dir / f"{refid}.npz",
                prob=prob.astype(np.float16),
                mask=mask.astype(np.uint8),
            )
        print(f"  Saved ensemble predictions to {ens_dir}")

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--exp', nargs='+', required=True,
        help='Experiment output dir name(s) under experiments/exp3_annotation/outputs/'
    )
    parser.add_argument(
        '--data-experiment', default='D2_alphaearth',
        help='EXPERIMENT_CONFIGS key for dataloader (default: D2_alphaearth)'
    )
    parser.add_argument('--num-folds', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--save-predictions', action='store_true')
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
            save_predictions=args.save_predictions,
            device=device,
        )


if __name__ == "__main__":
    main()
