#!/usr/bin/env python3
"""
Result plotting script for Part II spectral/spatial resolution experiments.

Generates:
1. Boxplots per condition (IoU distribution across folds)
2. Paired comparison plots (fold-by-fold: condition X vs anchor)
3. Block summary bar charts with error bars
4. Training curve overlays

Usage:
    python plot_results.py
    python plot_results.py --results-summary outputs/analysis/results_summary.json
    python plot_results.py --block A  # only Block A plots
"""

import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

PART2_DIR = Path(__file__).resolve().parent.parent.parent
EXPERIMENTS_DIR = PART2_DIR / "outputs" / "experiments"
OUTPUT_DIR = PART2_DIR / "outputs" / "figures"

NUM_FOLDS = 5

# Colors per block
BLOCK_COLORS = {
    "A": "#2196F3",   # blue
    "B": "#4CAF50",   # green
    "C": "#FF9800",   # orange
    "D": "#9C27B0",   # purple
    "sanity": "#607D8B",
}

# Condition display names
DISPLAY_NAMES = {
    "A1_s2_rgb": "S2 RGB",
    "A2_s2_rgbnir": "S2 RGB+NIR",
    "A3_s2_9band": "S2 9-band",
    "A4_s2_indices": "S2 9-band+idx",
    "A5_indices_only": "Indices only",
    "A6_temporal_diff": "Temporal diff",
    "B2_s2_bandgroup": "Band-group enc.",
    "C2_ps_rgb": "PS RGB",
    "C2hm_ps_rgb_hm": "PS RGB (HM)",
    "C3_s2_ps_fusion": "S2+PS fusion",
    "D2_alphaearth": "AlphaEarth",
    "D3_s2_ae_fusion": "S2+AE fusion",
    "LSTM7lite_sanity": "LSTM-7-lite",
}

BLOCK_ANCHORS = {
    "A": "A3_s2_9band",
    "B": "A3_s2_9band",
    "C": "A1_s2_rgb",
    "D": "A3_s2_9band",
}


def load_fold_ious(experiments_dir: Path, exp_name: str) -> list:
    """Load per-fold best validation IoU."""
    values = []
    for fold in range(NUM_FOLDS):
        hist_path = experiments_dir / f"{exp_name}_fold{fold}" / "history.json"
        if not hist_path.exists():
            return []
        with open(hist_path) as f:
            history = json.load(f)
        val_epochs = history.get('val', [])
        if not val_epochs:
            return []
        best_iou = max(e.get('iou', 0) for e in val_epochs) * 100
        values.append(best_iou)
    return values


def load_training_curves(experiments_dir: Path, exp_name: str) -> dict:
    """Load epoch-by-epoch training curves for all folds."""
    curves = {}
    for fold in range(NUM_FOLDS):
        hist_path = experiments_dir / f"{exp_name}_fold{fold}" / "history.json"
        if not hist_path.exists():
            continue
        with open(hist_path) as f:
            history = json.load(f)
        curves[fold] = {
            'train_loss': [e.get('loss', 0) for e in history.get('train', [])],
            'val_iou': [e.get('iou', 0) * 100 for e in history.get('val', [])],
            'val_loss': [e.get('loss', 0) for e in history.get('val', [])],
        }
    return curves


def plot_block_boxplots(experiments_dir: Path, block: str, experiments: list,
                        output_dir: Path):
    """Create boxplot of IoU distribution per condition in a block."""
    data = {}
    for exp in experiments:
        values = load_fold_ious(experiments_dir, exp)
        if values:
            data[exp] = values

    if not data:
        return

    fig, ax = plt.subplots(figsize=(max(6, len(data) * 1.5), 5))

    positions = range(len(data))
    labels = [DISPLAY_NAMES.get(e, e) for e in data.keys()]
    values_list = list(data.values())

    bp = ax.boxplot(values_list, positions=positions, widths=0.6,
                    patch_artist=True, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='white',
                                   markeredgecolor='black', markersize=6))

    color = BLOCK_COLORS.get(block, '#666666')
    for patch in bp['boxes']:
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Overlay individual fold points
    rng = np.random.RandomState(42)
    for i, vals in enumerate(values_list):
        jitter = rng.uniform(-0.15, 0.15, len(vals))
        ax.scatter([i + j for j in jitter], vals, c='black', s=30, zorder=5, alpha=0.7)

    # Highlight anchor
    anchor = BLOCK_ANCHORS.get(block)
    if anchor in data:
        anchor_idx = list(data.keys()).index(anchor)
        ax.axhline(np.mean(data[anchor]), color='red', linestyle='--', alpha=0.5,
                    label=f'Anchor mean ({np.mean(data[anchor]):.1f}%)')
        ax.legend(loc='lower right')

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_ylabel('Validation IoU (%)')
    ax.set_title(f'Block {block}: IoU Distribution Across Folds')
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / f'block_{block}_boxplots.pdf', dpi=150, bbox_inches='tight')
    fig.savefig(output_dir / f'block_{block}_boxplots.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved block_{block}_boxplots.pdf/png")


def plot_paired_comparison(experiments_dir: Path, block: str, experiments: list,
                           output_dir: Path):
    """Create paired fold-by-fold comparison plot (condition vs anchor)."""
    anchor = BLOCK_ANCHORS.get(block)
    if not anchor:
        return

    anchor_values = load_fold_ious(experiments_dir, anchor)
    if not anchor_values:
        return

    # Filter to non-anchor experiments
    comparisons = [e for e in experiments if e != anchor]
    comp_data = {}
    for exp in comparisons:
        values = load_fold_ious(experiments_dir, exp)
        if values:
            comp_data[exp] = values

    if not comp_data:
        return

    n_comparisons = len(comp_data)
    fig, axes = plt.subplots(1, n_comparisons, figsize=(4 * n_comparisons, 4), squeeze=False)

    for idx, (exp_name, exp_values) in enumerate(comp_data.items()):
        ax = axes[0, idx]

        # Plot paired lines
        for fold in range(NUM_FOLDS):
            ax.plot([0, 1], [anchor_values[fold], exp_values[fold]],
                    'o-', color='gray', alpha=0.5, linewidth=1)

        # Mean
        ax.plot([0, 1], [np.mean(anchor_values), np.mean(exp_values)],
                's-', color='red', linewidth=2, markersize=8, zorder=5)

        ax.set_xticks([0, 1])
        ax.set_xticklabels([DISPLAY_NAMES.get(anchor, anchor),
                            DISPLAY_NAMES.get(exp_name, exp_name)],
                           fontsize=9)
        ax.set_ylabel('Validation IoU (%)')

        delta = np.mean(exp_values) - np.mean(anchor_values)
        ax.set_title(f'{DISPLAY_NAMES.get(exp_name, exp_name)}\n'
                     f'$\\Delta$ = {delta:+.2f} pp', fontsize=10)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle(f'Block {block}: Paired Fold Comparisons', fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / f'block_{block}_paired.pdf', dpi=150, bbox_inches='tight')
    fig.savefig(output_dir / f'block_{block}_paired.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved block_{block}_paired.pdf/png")


def plot_summary_bar_chart(all_data: dict, output_dir: Path):
    """Create combined bar chart across all blocks."""
    if not all_data:
        return

    fig, ax = plt.subplots(figsize=(max(10, len(all_data) * 0.8), 5))

    names = list(all_data.keys())
    means = [np.mean(v) for v in all_data.values()]
    stds = [np.std(v) for v in all_data.values()]
    colors = []
    for name in names:
        for block, anchor in BLOCK_ANCHORS.items():
            if name.startswith(block[0]) or name == anchor:
                colors.append(BLOCK_COLORS.get(block, '#666666'))
                break
        else:
            colors.append('#666666')

    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=stds, capsize=3, color=colors, alpha=0.7,
                  edgecolor='black', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_NAMES.get(n, n) for n in names],
                       rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Validation IoU (%)')
    ax.set_title('Part II: All Conditions Summary')
    ax.grid(axis='y', alpha=0.3)

    # Legend
    patches = [mpatches.Patch(color=c, label=f'Block {b}', alpha=0.7)
               for b, c in BLOCK_COLORS.items() if b != "sanity"]
    ax.legend(handles=patches, loc='upper right')

    fig.tight_layout()
    fig.savefig(output_dir / 'all_conditions_summary.pdf', dpi=150, bbox_inches='tight')
    fig.savefig(output_dir / 'all_conditions_summary.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved all_conditions_summary.pdf/png")


def plot_training_curves_overlay(experiments_dir: Path, experiments: list,
                                  output_dir: Path, block: str = ""):
    """Plot training curves (val IoU + train loss) overlaid for multiple experiments."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for exp_name in experiments:
        curves = load_training_curves(experiments_dir, exp_name)
        if not curves:
            continue

        color = None
        label = DISPLAY_NAMES.get(exp_name, exp_name)

        # Collect all fold curves
        all_val_iou = []
        all_train_loss = []

        for fold, c in curves.items():
            val_iou = c['val_iou']
            train_loss = c['train_loss']
            epochs = range(1, len(val_iou) + 1)

            ax1.plot(epochs, val_iou, alpha=0.15, linewidth=0.5, color=color)
            ax2.plot(epochs, train_loss, alpha=0.15, linewidth=0.5, color=color)

            all_val_iou.append(val_iou)
            all_train_loss.append(train_loss)

        # Mean curve
        max_len = max(len(v) for v in all_val_iou)
        padded_iou = np.full((len(all_val_iou), max_len), np.nan)
        padded_loss = np.full((len(all_train_loss), max_len), np.nan)
        for i, v in enumerate(all_val_iou):
            padded_iou[i, :len(v)] = v
        for i, v in enumerate(all_train_loss):
            padded_loss[i, :len(v)] = v

        mean_iou = np.nanmean(padded_iou, axis=0)
        mean_loss = np.nanmean(padded_loss, axis=0)
        epochs = range(1, max_len + 1)

        line1, = ax1.plot(epochs, mean_iou, linewidth=2, label=label)
        ax2.plot(epochs, mean_loss, linewidth=2, label=label, color=line1.get_color())

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Validation IoU (%)')
    ax1.set_title('(A) Validation IoU')
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Training Loss')
    ax2.set_title('(B) Training Loss')
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    suffix = f'_block_{block}' if block else ''
    fig.suptitle(f'Training Curves{" - Block " + block if block else ""}', fontsize=12)
    fig.tight_layout()
    fig.savefig(output_dir / f'training_curves{suffix}.pdf', dpi=150, bbox_inches='tight')
    fig.savefig(output_dir / f'training_curves{suffix}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved training_curves{suffix}.pdf/png")


def main():
    parser = argparse.ArgumentParser(description="Generate Part II result figures")
    parser.add_argument('--experiments-dir', type=str, default=str(EXPERIMENTS_DIR))
    parser.add_argument('--output-dir', type=str, default=str(OUTPUT_DIR))
    parser.add_argument('--block', type=str, default=None,
                        choices=['A', 'B', 'C', 'D'],
                        help='Only generate plots for this block')
    args = parser.parse_args()

    experiments_dir = Path(args.experiments_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print("Generating Part II Result Figures")
    print(f"{'='*60}")
    print(f"Experiments: {experiments_dir}")
    print(f"Output: {output_dir}")

    # Define block experiment lists
    BLOCK_EXPERIMENTS = {
        "A": ["A1_s2_rgb", "A2_s2_rgbnir", "A3_s2_9band", "A4_s2_indices",
              "A5_indices_only", "A6_temporal_diff"],
        "B": ["A3_s2_9band", "B2_s2_bandgroup"],
        "C": ["A1_s2_rgb", "C2_ps_rgb", "C2hm_ps_rgb_hm", "C3_s2_ps_fusion"],
        "D": ["A3_s2_9band", "D2_alphaearth", "D3_s2_ae_fusion"],
    }

    blocks_to_plot = [args.block] if args.block else list(BLOCK_EXPERIMENTS.keys())

    # Per-block plots
    all_data = {}
    for block in blocks_to_plot:
        experiments = BLOCK_EXPERIMENTS[block]
        print(f"\nBlock {block}:")

        # Collect available data
        block_data = {}
        for exp in experiments:
            values = load_fold_ious(experiments_dir, exp)
            if values:
                block_data[exp] = values
                all_data[exp] = values

        if not block_data:
            print(f"  No data available for Block {block}")
            continue

        print(f"  Available: {list(block_data.keys())}")

        # Boxplots
        plot_block_boxplots(experiments_dir, block, list(block_data.keys()), output_dir)

        # Paired comparisons
        plot_paired_comparison(experiments_dir, block, list(block_data.keys()), output_dir)

        # Training curves
        plot_training_curves_overlay(experiments_dir, list(block_data.keys()),
                                      output_dir, block=block)

    # Overall summary bar chart
    if not args.block and all_data:
        print(f"\nOverall:")
        plot_summary_bar_chart(all_data, output_dir)

    print(f"\nDone! Figures saved to {output_dir}")


if __name__ == "__main__":
    main()
