#!/usr/bin/env python3
"""
Calibration analysis and Precision-Recall curves for land-take detection.

Combines both analyses in a single GPU pass (shared OOF inference).

Produces:
- Reliability diagram (10-bin, one line per model) -> fig_reliability_diagram.pdf
- ECE per model -> calibration_analysis.json
- PR curves (one line per model) -> fig_pr_curves.pdf
- Average Precision per model -> pr_curves.json

Usage:
    python calibration_pr_analysis.py
    python calibration_pr_analysis.py --experiments annual,bi_temporal,bi_seasonal
"""

import sys
import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# Add paths for imports
script_dir = Path(__file__).resolve().parent
parent_dir = script_dir.parent.parent
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(parent_dir.parent))
sys.path.insert(0, str(parent_dir / "scripts" / "modeling"))

from models_multitemporal import create_multitemporal_model
from PART1_multi_temporal_experiments.scripts.data_preparation.dataset_multitemporal import get_dataloaders
from PART1_multi_temporal_experiments.scripts.experiments_v2 import (
    EXPERIMENTS_V2 as EXPERIMENTS,
    V2_OUTPUTS_DIR, V2_SENTINEL_DIR, V2_MASK_DIR, V2_ANALYSIS_DIR,
    V2_SPLITS_DIR, V2_CHANGE_LEVEL_PATH,
    DISPLAY_NAMES, PLOT_COLORS, TEMPORAL_CONDITIONS,
)


def load_model_from_checkpoint(checkpoint_path: Path, config: dict, device: torch.device):
    """Load a trained model from checkpoint."""
    model = create_multitemporal_model(
        config.get('model_name', 'lstm_unet'),
        encoder_name=config.get('encoder_name', 'resnet50'),
        encoder_weights=None,
        in_channels=9,
        classes=1,
        lstm_hidden_dim=config.get('lstm_hidden_dim', 256),
        lstm_num_layers=config.get('lstm_num_layers', 2),
        convlstm_kernel_size=config.get('convlstm_kernel_size', 3),
        skip_aggregation=config.get('skip_aggregation', 'max'),
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    return model


def collect_oof_predictions(experiment_config: dict, num_folds: int, device: torch.device):
    """
    Collect all OOF pixel-level (predicted_prob, ground_truth) pairs.

    Returns:
        all_probs: np.ndarray of predicted probabilities (flattened)
        all_labels: np.ndarray of ground truth labels (flattened, 0 or 1)
    """
    exp_name = experiment_config['name']
    temporal_sampling = experiment_config['temporal_sampling']

    all_probs = []
    all_labels = []

    for fold in range(num_folds):
        exp_dir = V2_OUTPUTS_DIR / f"{exp_name}_fold{fold}"

        if not exp_dir.exists():
            print(f"  WARNING: {exp_dir} not found, skipping fold {fold}")
            continue

        config_path = exp_dir / "config.json"
        with open(config_path) as f:
            config = json.load(f)

        checkpoint_path = exp_dir / "best_model.pth"
        if not checkpoint_path.exists():
            print(f"  WARNING: {checkpoint_path} not found, skipping fold {fold}")
            continue

        model = load_model_from_checkpoint(checkpoint_path, config, device)

        dataloaders = get_dataloaders(
            temporal_sampling=temporal_sampling,
            batch_size=1,
            num_workers=4,
            image_size=config['image_size'],
            output_format="LSTM",
            fold=fold,
            num_folds=num_folds,
            seed=config.get('seed', 42),
            sentinel2_dir=V2_SENTINEL_DIR,
            mask_dir=V2_MASK_DIR,
            splits_dir=V2_SPLITS_DIR,
            change_level_path=V2_CHANGE_LEVEL_PATH,
        )
        val_loader = dataloaders['val']

        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                masks = batch['mask']

                outputs = model(images)
                probs = torch.sigmoid(outputs).cpu().numpy()

                for i in range(len(masks)):
                    prob_flat = probs[i].squeeze().flatten()
                    mask_flat = masks[i].numpy().flatten()
                    all_probs.append(prob_flat)
                    all_labels.append(mask_flat)

    return np.concatenate(all_probs), np.concatenate(all_labels)


def compute_calibration(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10):
    """
    Compute calibration statistics.

    Returns:
        bin_edges, bin_accs, bin_confs, bin_counts, ece
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_accs = np.zeros(n_bins)
    bin_confs = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (probs >= low) & (probs <= high)
        else:
            mask = (probs >= low) & (probs < high)

        bin_counts[i] = mask.sum()
        if bin_counts[i] > 0:
            bin_accs[i] = labels[mask].mean()
            bin_confs[i] = probs[mask].mean()

    # Expected Calibration Error
    total = len(probs)
    ece = np.sum(bin_counts / total * np.abs(bin_accs - bin_confs))

    return bin_edges, bin_accs, bin_confs, bin_counts, ece


def compute_pr_curve(probs: np.ndarray, labels: np.ndarray, n_thresholds: int = 200):
    """
    Compute precision-recall curve.

    Returns:
        precisions, recalls, thresholds, average_precision
    """
    thresholds = np.linspace(0, 1, n_thresholds + 1)[1:]  # skip 0

    precisions = np.zeros(n_thresholds)
    recalls = np.zeros(n_thresholds)

    for i, thresh in enumerate(thresholds):
        pred = probs >= thresh
        tp = np.sum(pred & (labels == 1))
        fp = np.sum(pred & (labels == 0))
        fn = np.sum((~pred) & (labels == 1))

        precisions[i] = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recalls[i] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Average Precision (trapezoidal approximation, descending recall)
    sorted_idx = np.argsort(recalls)
    sorted_recalls = recalls[sorted_idx]
    sorted_precisions = precisions[sorted_idx]

    ap = np.trapz(sorted_precisions, sorted_recalls)

    return precisions, recalls, thresholds, ap


def plot_reliability_diagram(calibration_data: dict, output_dir: Path, n_bins: int = 10):
    """Plot reliability diagram with one line per model."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect calibration')

    for cond_name, data in calibration_data.items():
        color = PLOT_COLORS.get(cond_name, '#333333')
        display = DISPLAY_NAMES.get(cond_name, cond_name)
        ece = data['ece']

        bin_confs = np.array(data['bin_confs'])
        bin_accs = np.array(data['bin_accs'])
        bin_counts = np.array(data['bin_counts'])

        # Only plot bins with data
        valid = bin_counts > 0
        ax.plot(bin_confs[valid], bin_accs[valid], 'o-',
                color=color, linewidth=2, markersize=6,
                label=f'{display} (ECE={ece:.3f})')

    ax.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax.set_ylabel('Fraction of Positives', fontsize=12)
    ax.set_title('Reliability Diagram', fontsize=13, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "fig_reliability_diagram.pdf"
    plt.savefig(output_path, bbox_inches='tight')
    plt.savefig(output_dir / "fig_reliability_diagram.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_pr_curves(pr_data: dict, output_dir: Path):
    """Plot PR curves with one line per model."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    for cond_name, data in pr_data.items():
        color = PLOT_COLORS.get(cond_name, '#333333')
        display = DISPLAY_NAMES.get(cond_name, cond_name)
        ap = data['average_precision']

        recalls = np.array(data['recalls'])
        precisions = np.array(data['precisions'])

        # Sort by recall for smooth curve
        sorted_idx = np.argsort(recalls)
        ax.plot(recalls[sorted_idx], precisions[sorted_idx],
                color=color, linewidth=2,
                label=f'{display} (AP={ap:.3f})')

    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves', fontsize=13, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "fig_pr_curves.pdf"
    plt.savefig(output_path, bbox_inches='tight')
    plt.savefig(output_dir / "fig_pr_curves.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Calibration and PR curve analysis")
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--experiments', type=str, default=None,
                        help='Comma-separated experiment keys')
    parser.add_argument('--num-folds', type=int, default=5)
    parser.add_argument('--n-bins', type=int, default=10,
                        help='Number of calibration bins')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Filter experiments
    if args.experiments:
        exp_keys = [k.strip() for k in args.experiments.split(',')]
        active_experiments = {k: v for k, v in EXPERIMENTS.items() if k in exp_keys}
    else:
        active_experiments = EXPERIMENTS

    output_dir = Path(args.output_dir) if args.output_dir else V2_ANALYSIS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("CALIBRATION & PR CURVE ANALYSIS")
    print("=" * 70)
    print(f"Experiments: {list(active_experiments.keys())}")

    calibration_data = {}
    pr_data = {}

    for cond_name, config in active_experiments.items():
        print(f"\n--- {cond_name.upper()} ({config['name']}) ---")

        probs, labels = collect_oof_predictions(config, args.num_folds, device)
        n_pixels = len(probs)
        n_positive = labels.sum()
        print(f"  Pixels: {n_pixels:,} (positive: {n_positive:,}, {n_positive/n_pixels*100:.2f}%)")

        # Calibration
        bin_edges, bin_accs, bin_confs, bin_counts, ece = compute_calibration(
            probs, labels, n_bins=args.n_bins
        )
        print(f"  ECE: {ece:.4f}")

        calibration_data[cond_name] = {
            'ece': float(ece),
            'n_bins': args.n_bins,
            'bin_accs': bin_accs.tolist(),
            'bin_confs': bin_confs.tolist(),
            'bin_counts': bin_counts.tolist(),
            'n_pixels': int(n_pixels),
            'n_positive': int(n_positive),
        }

        # PR curve
        precisions, recalls, thresholds, ap = compute_pr_curve(probs, labels)
        print(f"  Average Precision: {ap:.4f}")

        pr_data[cond_name] = {
            'average_precision': float(ap),
            'precisions': precisions.tolist(),
            'recalls': recalls.tolist(),
            'thresholds': thresholds.tolist(),
        }

    # Save JSON results
    cal_file = output_dir / "calibration_analysis.json"
    with open(cal_file, 'w') as f:
        json.dump(calibration_data, f, indent=2)
    print(f"\n  Calibration saved to: {cal_file}")

    pr_file = output_dir / "pr_curves.json"
    with open(pr_file, 'w') as f:
        json.dump(pr_data, f, indent=2)
    print(f"  PR curves saved to: {pr_file}")

    # Generate figures
    print("\nGenerating figures...")
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_reliability_diagram(calibration_data, figures_dir, n_bins=args.n_bins)
    plot_pr_curves(pr_data, figures_dir)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Condition':<20} {'ECE':<10} {'AP':<10}")
    print("-" * 40)
    for cond_name in active_experiments:
        ece = calibration_data[cond_name]['ece']
        ap = pr_data[cond_name]['average_precision']
        display = DISPLAY_NAMES.get(cond_name, cond_name)
        print(f"{display:<20} {ece:<10.4f} {ap:<10.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
