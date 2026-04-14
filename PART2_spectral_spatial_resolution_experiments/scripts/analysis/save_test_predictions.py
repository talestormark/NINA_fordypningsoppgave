#!/usr/bin/env python3
"""
Save test set predictions (ensemble probabilities + ground truth masks) to disk.

For each experiment, loads 5-fold models, runs inference on 40 test tiles,
averages probabilities across folds, and saves per-tile .npz files with:
  - prob: ensemble probability map (H, W), float32
  - mask: ground truth binary mask (H, W), uint8

These .npz files are consumed by:
  - boundary_metrics.py (BF@1, BF@2)
  - qualitative visualization scripts

Usage:
    python save_test_predictions.py --experiment A3_s2_9band
    python save_test_predictions.py --experiment A3_s2_9band E4_ae_unet_sparse D2_alphaearth
    python save_test_predictions.py --all
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
# Path setup
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]
PART1_DIR = REPO_ROOT / "PART1_multi_temporal_experiments"
PART2_DIR = REPO_ROOT / "PART2_spectral_spatial_resolution_experiments"

sys.path.insert(0, str(REPO_ROOT / "scripts" / "modeling"))
from train import Metrics

sys.path.insert(0, str(PART1_DIR / "scripts" / "modeling"))
from models_multitemporal import create_multitemporal_model

_p2_spec = importlib.util.spec_from_file_location(
    "p2_dataset", PART2_DIR / "scripts" / "data_preparation" / "dataset.py"
)
_p2_dataset = importlib.util.module_from_spec(_p2_spec)
_p2_spec.loader.exec_module(_p2_dataset)
EXPERIMENT_CONFIGS = _p2_dataset.EXPERIMENT_CONFIGS
get_dataloaders = _p2_dataset.get_dataloaders

EXPERIMENTS_DIR = PART2_DIR / "outputs" / "experiments"
NUM_FOLDS = 5

ALL_EXPERIMENTS = [
    "A1_s2_rgb", "A2_s2_rgbnir", "A3_s2_9band", "A4_s2_indices",
    "D2_alphaearth", "E4_ae_unet_sparse", "E4_A3_s2_9band_sparse",
]


# ---------------------------------------------------------------------------
# Model loading (same as evaluate_test_set.py)
# ---------------------------------------------------------------------------

def load_model(checkpoint_path, config, device):
    cfg = EXPERIMENT_CONFIGS[config['experiment']]
    _, C, _, _ = cfg["expected_shape"]

    model = create_multitemporal_model(
        config['model_name'],
        encoder_name=config['encoder_name'],
        encoder_weights=None,
        in_channels=C,
        classes=1,
        lstm_hidden_dim=config.get('lstm_hidden_dim', 512),
        lstm_num_layers=config.get('lstm_num_layers', 2),
        convlstm_kernel_size=config.get('convlstm_kernel_size', 3),
        skip_aggregation=config.get('skip_aggregation', 'max'),
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def save_predictions_for_experiment(experiment, device, output_base=None):
    """Run 5-fold ensemble inference and save per-tile predictions."""
    print(f"\n{'='*60}")
    print(f"Saving predictions: {experiment}")
    print(f"{'='*60}")

    if output_base is None:
        output_base = EXPERIMENTS_DIR / experiment / "predictions"
    else:
        output_base = Path(output_base) / experiment
    output_base.mkdir(parents=True, exist_ok=True)

    # Collect predictions from all folds
    all_fold_probs = {}  # refid -> list of prob arrays
    all_masks = {}       # refid -> mask array

    for fold in range(NUM_FOLDS):
        exp_dir = EXPERIMENTS_DIR / f"{experiment}_fold{fold}"
        if not exp_dir.exists():
            print(f"  WARNING: {exp_dir} not found, skipping fold {fold}")
            continue

        config_path = exp_dir / "config.json"
        checkpoint_path = exp_dir / "best_model.pth"

        if not checkpoint_path.exists():
            print(f"  WARNING: {checkpoint_path} not found, skipping fold {fold}")
            continue

        with open(config_path) as f:
            config = json.load(f)

        print(f"  Fold {fold}: loading model...")
        model = load_model(checkpoint_path, config, device)

        print(f"  Fold {fold}: creating dataloader...")
        dataloaders = get_dataloaders(
            experiment=config['experiment'],
            batch_size=1,
            num_workers=4,
            image_size=config.get('image_size', 64),
            fold=fold,
            num_folds=NUM_FOLDS,
            seed=config.get('seed', 42),
        )
        test_loader = dataloaders['test']

        print(f"  Fold {fold}: running inference on {len(test_loader.dataset)} tiles...")
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"  Fold {fold}", leave=False):
                images = batch['image'].to(device)
                masks = batch['mask']
                refids = batch['refid']

                outputs = model(images)
                probs = torch.sigmoid(outputs).cpu().numpy()

                for i, refid in enumerate(refids):
                    prob = probs[i].squeeze(0)  # (H, W)
                    mask = masks[i].squeeze(0).numpy()  # (H, W)

                    if refid not in all_fold_probs:
                        all_fold_probs[refid] = []
                    all_fold_probs[refid].append(prob)
                    all_masks[refid] = mask

        del model
        torch.cuda.empty_cache()

    # Average probabilities across folds (ensemble)
    n_saved = 0
    for refid in sorted(all_fold_probs.keys()):
        probs = np.stack(all_fold_probs[refid], axis=0)
        avg_prob = np.mean(probs, axis=0).astype(np.float32)
        mask = all_masks[refid].astype(np.uint8)

        # Save as .npz
        safe_refid = refid.replace("/", "_")
        out_path = output_base / f"{safe_refid}.npz"
        np.savez_compressed(out_path, prob=avg_prob, mask=mask)
        n_saved += 1

    print(f"\n  Saved {n_saved} predictions to {output_base}")
    return output_base


def main():
    parser = argparse.ArgumentParser(
        description="Save test set ensemble predictions to disk"
    )
    parser.add_argument('--experiment', type=str, nargs='*', default=None,
                        help='Experiment name(s) (e.g., A3_s2_9band)')
    parser.add_argument('--all', action='store_true',
                        help='Process all experiments')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Base output directory (default: per-experiment predictions/ subdir)')
    args = parser.parse_args()

    if args.all:
        experiments = ALL_EXPERIMENTS
    elif args.experiment:
        experiments = args.experiment
    else:
        parser.error("Must specify --experiment or --all")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    prediction_dirs = []
    for exp in experiments:
        if exp not in EXPERIMENT_CONFIGS:
            print(f"WARNING: {exp} not in EXPERIMENT_CONFIGS, skipping")
            continue
        pred_dir = save_predictions_for_experiment(exp, device, args.output_dir)
        prediction_dirs.append((exp, pred_dir))

    print(f"\n{'='*60}")
    print("All predictions saved. To run boundary metrics:")
    for exp, pred_dir in prediction_dirs:
        print(f"  python boundary_metrics.py --predictions-dir {pred_dir}")
    print()


if __name__ == "__main__":
    main()
