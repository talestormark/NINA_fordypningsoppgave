#!/usr/bin/env python3
"""
Convergence comparison: Dense vs Sparse U-Net on Sentinel-2.

Two-panel figure:
  (a) Validation IoU over epochs — headline comparison
  (b) Training loss over epochs — shows noisier gradient from sparse labels

Both show mean ± 1 std across 5 folds, smoothed with a running average.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
P2_OUTPUTS = REPO_ROOT / "PART2_spectral_spatial_resolution_experiments" / "outputs" / "experiments"
AE_OUTPUTS = Path(__file__).resolve().parents[2] / "outputs"
OUT_DIR = AE_OUTPUTS / "figures"


def load_histories(exp_dir_pattern, n_folds=5):
    """Load per-epoch train/val metrics across folds."""
    all_train_loss, all_val_loss = [], []
    all_train_iou, all_val_iou = [], []

    for fold in range(n_folds):
        hp = exp_dir_pattern(fold)
        if not hp.exists():
            continue
        with open(hp) as f:
            h = json.load(f)

        train_loss = [e['loss'] for e in h['train']]
        val_loss = [e['loss'] for e in h['val']]
        train_iou = [e['iou'] for e in h['train']]
        val_iou = [e['iou'] for e in h['val']]

        all_train_loss.append(train_loss)
        all_val_loss.append(val_loss)
        all_train_iou.append(train_iou)
        all_val_iou.append(val_iou)

    # Truncate to shortest fold (in case of timeout)
    min_len = min(len(x) for x in all_val_iou)
    all_train_loss = np.array([x[:min_len] for x in all_train_loss])
    all_val_loss = np.array([x[:min_len] for x in all_val_loss])
    all_train_iou = np.array([x[:min_len] for x in all_train_iou])
    all_val_iou = np.array([x[:min_len] for x in all_val_iou])

    return {
        'train_loss': all_train_loss,
        'val_loss': all_val_loss,
        'train_iou': all_train_iou,
        'val_iou': all_val_iou,
        'n_epochs': min_len,
        'n_folds': len(all_val_iou),
    }


def smooth(arr, window=10):
    """Running average smoothing per fold."""
    if window <= 1:
        return arr
    kernel = np.ones(window) / window
    smoothed = np.zeros_like(arr)
    for i in range(arr.shape[0]):
        smoothed[i] = np.convolve(arr[i], kernel, mode='same')
    return smoothed


def plot_mean_std(ax, epochs, data, color, label, alpha=0.15, smooth_window=10):
    """Plot mean line with shaded ±1 std band."""
    data_smooth = smooth(data, smooth_window)
    mean = data_smooth.mean(axis=0)
    std = data_smooth.std(axis=0)
    ax.plot(epochs, mean * 100, color=color, label=label, linewidth=1.5)
    ax.fill_between(epochs, (mean - std) * 100, (mean + std) * 100,
                     color=color, alpha=alpha)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load histories
    dense = load_histories(
        lambda fold: P2_OUTPUTS / f"A3_s2_9band_fold{fold}" / "history.json"
    )
    sparse = load_histories(
        lambda fold: AE_OUTPUTS / f"E4_A3_s2_9band_sparse" / f"fold{fold}" / "history.json"
    )

    print(f"Dense (A3): {dense['n_folds']} folds, {dense['n_epochs']} epochs")
    print(f"Sparse (E4-S2): {sparse['n_folds']} folds, {sparse['n_epochs']} epochs")

    n_epochs = min(dense['n_epochs'], sparse['n_epochs'])
    epochs = np.arange(1, n_epochs + 1)

    # Truncate to common length
    for key in ['train_loss', 'val_loss', 'train_iou', 'val_iou']:
        dense[key] = dense[key][:, :n_epochs]
        sparse[key] = sparse[key][:, :n_epochs]

    # Print final values
    dense_final = dense['val_iou'].mean(axis=0)[-1] * 100
    sparse_final = sparse['val_iou'].mean(axis=0)[-1] * 100
    dense_best = dense['val_iou'].mean(axis=0).max() * 100
    sparse_best = sparse['val_iou'].mean(axis=0).max() * 100
    print(f"Dense  — final val IoU: {dense_final:.1f}%, best: {dense_best:.1f}%")
    print(f"Sparse — final val IoU: {sparse_final:.1f}%, best: {sparse_best:.1f}%")

    # Colors
    c_dense = '#2166ac'   # blue
    c_sparse = '#b2182b'  # red

    # =========================================================================
    # Figure: 2-panel (val IoU + train loss)
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Panel (a): Validation IoU
    plot_mean_std(ax1, epochs, dense['val_iou'], c_dense,
                  'Dense supervision (~6,000 px/tile)')
    plot_mean_std(ax1, epochs, sparse['val_iou'], c_sparse,
                  'Sparse supervision (50 pts/tile)')

    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Validation IoU (%)', fontsize=11)
    ax1.set_title('(a) Validation IoU', fontsize=12)
    ax1.legend(loc='lower right', fontsize=9, framealpha=0.9)
    ax1.set_xlim(1, 250)
    ax1.grid(True, alpha=0.3)

    # Panel (b): Training loss
    plot_mean_std(ax2, epochs, dense['train_loss'], c_dense, 'Dense', smooth_window=20)
    plot_mean_std(ax2, epochs, sparse['train_loss'], c_sparse, 'Sparse', smooth_window=20)

    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Training loss', fontsize=11)
    ax2.set_title('(b) Training loss', fontsize=12)
    ax2.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax2.set_xlim(1, 250)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout(pad=1.5)
    fig.savefig(OUT_DIR / "convergence_dense_vs_sparse.pdf", dpi=300, bbox_inches='tight')
    fig.savefig(OUT_DIR / "convergence_dense_vs_sparse.png", dpi=200, bbox_inches='tight')
    print(f"\nSaved: {OUT_DIR / 'convergence_dense_vs_sparse.pdf'}")

    # =========================================================================
    # Figure: val IoU only (single panel, cleaner for slides)
    # =========================================================================
    fig2, ax = plt.subplots(1, 1, figsize=(6, 4))

    plot_mean_std(ax, epochs, dense['val_iou'], c_dense,
                  f'Dense supervision (~6,000 px/tile)')
    plot_mean_std(ax, epochs, sparse['val_iou'], c_sparse,
                  f'Sparse supervision (50 pts/tile)')

    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Validation IoU (%)', fontsize=11)
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax.set_xlim(1, 250)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig2.savefig(OUT_DIR / "convergence_val_iou_only.pdf", dpi=300, bbox_inches='tight')
    fig2.savefig(OUT_DIR / "convergence_val_iou_only.png", dpi=200, bbox_inches='tight')
    print(f"Saved: {OUT_DIR / 'convergence_val_iou_only.pdf'}")


if __name__ == "__main__":
    main()
