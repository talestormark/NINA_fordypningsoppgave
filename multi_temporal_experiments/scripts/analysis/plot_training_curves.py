#!/usr/bin/env python3
"""
Generate training curves figure for temporal sampling comparison.

Shows validation IoU vs epoch for each condition (T=2, T=7, T=14),
with all 5 folds shown as thin lines and mean as bold line.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
MT_EXPERIMENTS_DIR = SCRIPT_DIR.parent.parent
EXPERIMENTS_DIR = MT_EXPERIMENTS_DIR / "outputs" / "experiments"
OUTPUT_DIR = MT_EXPERIMENTS_DIR / "outputs" / "analysis"

# Experiment configurations
EXPERIMENTS = {
    'annual': {
        'name': 'Annual (T=7)',
        'prefix': 'exp001_v2',
        'color': '#2ecc71',
    },
    'bi_temporal': {
        'name': 'Bi-temporal (T=2)',
        'prefix': 'exp003_v2',
        'color': '#e74c3c',
    },
    'bi_seasonal': {
        'name': 'Bi-seasonal (T=14)',
        'prefix': 'exp002_v2',
        'color': '#3498db',
    },
}


def load_training_history(exp_prefix: str, n_folds: int = 5):
    """Load training history for all folds of an experiment."""
    histories = []

    for fold in range(n_folds):
        history_file = EXPERIMENTS_DIR / f"{exp_prefix}_fold{fold}" / "history.json"
        if not history_file.exists():
            print(f"Warning: {history_file} not found")
            continue

        with open(history_file, 'r') as f:
            data = json.load(f)

        # Extract validation IoU and loss per epoch
        val_ious = [epoch['iou'] for epoch in data['val']]
        val_losses = [epoch['loss'] for epoch in data['val']]
        train_losses = [epoch['loss'] for epoch in data['train']]

        histories.append({
            'val_iou': val_ious,
            'val_loss': val_losses,
            'train_loss': train_losses,
            'n_epochs': len(val_ious),
        })

    return histories


def main():
    print("Loading training histories...")

    # Load all histories
    all_histories = {}
    for key, config in EXPERIMENTS.items():
        histories = load_training_history(config['prefix'])
        all_histories[key] = histories
        print(f"  {config['name']}: {len(histories)} folds, "
              f"epochs: {[h['n_epochs'] for h in histories]}")

    # Create figure with 2 panels
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # =========================================================================
    # Panel A: Validation IoU vs Epoch
    # =========================================================================
    ax1 = axes[0]

    for key, config in EXPERIMENTS.items():
        histories = all_histories[key]
        color = config['color']
        name = config['name']

        # Plot individual folds (thin, transparent)
        max_epochs = max(h['n_epochs'] for h in histories)
        for i, h in enumerate(histories):
            epochs = np.arange(1, h['n_epochs'] + 1)
            ax1.plot(epochs, np.array(h['val_iou']) * 100,
                    color=color, alpha=0.3, linewidth=1)

        # Compute and plot mean (bold)
        # Pad shorter histories with NaN for averaging
        all_ious = []
        for h in histories:
            padded = np.full(max_epochs, np.nan)
            padded[:h['n_epochs']] = h['val_iou']
            all_ious.append(padded)

        mean_iou = np.nanmean(all_ious, axis=0) * 100
        epochs = np.arange(1, max_epochs + 1)
        ax1.plot(epochs, mean_iou, color=color, linewidth=2.5, label=name)

    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Validation IoU (%)', fontsize=12)
    ax1.set_title('(A) Validation IoU During Training', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.set_ylim(0, 80)
    ax1.set_xlim(0, None)
    ax1.grid(True, alpha=0.3)

    # =========================================================================
    # Panel B: Training Loss vs Epoch
    # =========================================================================
    ax2 = axes[1]

    for key, config in EXPERIMENTS.items():
        histories = all_histories[key]
        color = config['color']
        name = config['name']

        # Plot individual folds (thin, transparent)
        max_epochs = max(h['n_epochs'] for h in histories)
        for i, h in enumerate(histories):
            epochs = np.arange(1, h['n_epochs'] + 1)
            ax2.plot(epochs, h['train_loss'],
                    color=color, alpha=0.3, linewidth=1)

        # Compute and plot mean (bold)
        all_losses = []
        for h in histories:
            padded = np.full(max_epochs, np.nan)
            padded[:h['n_epochs']] = h['train_loss']
            all_losses.append(padded)

        mean_loss = np.nanmean(all_losses, axis=0)
        epochs = np.arange(1, max_epochs + 1)
        ax2.plot(epochs, mean_loss, color=color, linewidth=2.5, label=name)

    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Training Loss', fontsize=12)
    ax2.set_title('(B) Training Loss During Training', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.set_xlim(0, None)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_path = OUTPUT_DIR / "training_curves.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Figure saved to: {output_path}")

    # Also save as PDF for LaTeX
    output_pdf = OUTPUT_DIR / "training_curves.pdf"
    plt.savefig(output_pdf, bbox_inches='tight', facecolor='white')
    print(f"✓ PDF saved to: {output_pdf}")

    plt.close()


if __name__ == "__main__":
    main()
