#!/usr/bin/env python3
"""
Generate statistical distribution figures for temporal sampling comparison.

2-panel figure:
  Panel A (Descriptive): Three violins for IoU by condition (overall spread)
  Panel B (Inferential): Two delta violins for ΔIoU (paired differences)
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Deterministic jitter
RNG = np.random.default_rng(seed=42)

# Paths
SCRIPT_DIR = Path(__file__).parent
MT_EXPERIMENTS_DIR = SCRIPT_DIR.parent.parent
OUTPUT_DIR = MT_EXPERIMENTS_DIR / "outputs" / "analysis"
PER_SAMPLE_FILE = OUTPUT_DIR / "per_sample_iou.json"


def load_per_tile_data():
    """Load per-tile IoU data from per-sample analysis results."""
    with open(PER_SAMPLE_FILE, 'r') as f:
        data = json.load(f)

    # Extract per-tile IoUs for each condition
    # Use lstm7_no_es_iou for annual (unified 400-epoch protocol)
    conditions = {
        'annual': np.array(data['lstm7_no_es_iou']),
        'bi_temporal': np.array(data['bi_temporal_iou']),
        'bi_seasonal': np.array(data['bi_seasonal_iou']),
    }

    return conditions


def main():
    print("Loading per-tile IoU data...")
    conditions = load_per_tile_data()

    # Extract data
    iou_annual = conditions['annual']
    iou_bitemporal = conditions['bi_temporal']
    iou_biseasonal = conditions['bi_seasonal']

    n_tiles = len(iou_annual)
    print(f"Loaded {n_tiles} tiles")

    # Compute paired differences
    delta_vs_bitemporal = iou_annual - iou_bitemporal  # positive = annual better
    delta_vs_biseasonal = iou_annual - iou_biseasonal  # positive = annual better

    # Convert to percentages
    iou_annual_pct = iou_annual * 100
    iou_bitemporal_pct = iou_bitemporal * 100
    iou_biseasonal_pct = iou_biseasonal * 100
    delta_vs_bitemporal_pct = delta_vs_bitemporal * 100
    delta_vs_biseasonal_pct = delta_vs_biseasonal * 100

    # Create figure with Panel B wider (inferential emphasis)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={'width_ratios': [1, 1.2]})

    # Color palette
    colors = {
        'bi_temporal': '#e74c3c',  # red
        'annual': '#2ecc71',        # green
        'bi_seasonal': '#3498db',   # blue
    }

    # =========================================================================
    # Panel A (Descriptive): Three violins for IoU by condition
    # =========================================================================
    ax1 = axes[0]

    # Prepare data for seaborn
    plot_data_a = []
    for iou, label in [(iou_bitemporal_pct, 'Bi-temporal\n(T=2)'),
                       (iou_annual_pct, 'Annual\n(T=7)'),
                       (iou_biseasonal_pct, 'Bi-seasonal\n(T=14)')]:
        for val in iou:
            plot_data_a.append({'Condition': label, 'IoU (%)': val})

    df_a = pd.DataFrame(plot_data_a)

    # Violin plot with cut=0 to prevent extending beyond data range
    palette_a = [colors['bi_temporal'], colors['annual'], colors['bi_seasonal']]
    sns.violinplot(data=df_a, x='Condition', y='IoU (%)', ax=ax1,
                   palette=palette_a, inner='box', linewidth=1, cut=0)

    # Overlay individual points (deterministic jitter)
    for i, iou in enumerate([iou_bitemporal_pct, iou_annual_pct, iou_biseasonal_pct]):
        jitter = RNG.uniform(-0.1, 0.1, len(iou))
        ax1.scatter(i + jitter, iou, color='white', edgecolor='black',
                    s=20, alpha=0.7, zorder=3, linewidth=0.5)

    # Add mean markers (diamond)
    for i, iou in enumerate([iou_bitemporal_pct, iou_annual_pct, iou_biseasonal_pct]):
        ax1.scatter(i, np.mean(iou), marker='D', color='black', s=60, zorder=4,
                    label='Mean' if i == 0 else None)

    ax1.set_ylabel('IoU (%)', fontsize=12)
    ax1.set_xlabel('')
    ax1.set_title('(A) Descriptive: IoU by Condition', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 100)

    # =========================================================================
    # Panel B (Inferential): Two delta violins for paired differences
    # =========================================================================
    ax2 = axes[1]

    # Prepare data
    plot_data_b = []
    for delta, label in [(delta_vs_bitemporal_pct, 'Annual −\nBi-temporal'),
                         (delta_vs_biseasonal_pct, 'Annual −\nBi-seasonal')]:
        for val in delta:
            plot_data_b.append({'Comparison': label, 'ΔIoU (pp)': val})

    df_b = pd.DataFrame(plot_data_b)

    # Violin plot with cut=0 to prevent extending beyond data range
    sns.violinplot(data=df_b, x='Comparison', y='ΔIoU (pp)', ax=ax2,
                   palette=['#9b59b6', '#e67e22'], inner='box', linewidth=1, cut=0)

    # Zero line (bold, emphasized)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=2.5, zorder=1)

    # Overlay individual points (deterministic jitter)
    for i, delta in enumerate([delta_vs_bitemporal_pct, delta_vs_biseasonal_pct]):
        jitter = RNG.uniform(-0.1, 0.1, len(delta))
        ax2.scatter(i + jitter, delta, color='white', edgecolor='black',
                    s=25, alpha=0.8, zorder=3, linewidth=0.5)

    # Add mean (diamond) and median (horizontal line) markers
    for i, delta in enumerate([delta_vs_bitemporal_pct, delta_vs_biseasonal_pct]):
        mean_val = np.mean(delta)
        median_val = np.median(delta)

        # Mean as red diamond
        ax2.scatter(i, mean_val, marker='D', color='red', s=100, zorder=5,
                    edgecolor='black', linewidth=1.5,
                    label='Mean' if i == 0 else None)

        # Median as blue horizontal line
        ax2.hlines(median_val, i - 0.2, i + 0.2, color='blue', linewidth=4, zorder=5,
                   label='Median' if i == 0 else None)

    ax2.set_ylabel('ΔIoU (percentage points)', fontsize=12)
    ax2.set_xlabel('')
    ax2.set_title('(B) Inferential: Paired Differences', fontsize=12, fontweight='bold')

    # Add legend
    ax2.legend(loc='upper right', fontsize=10)

    # Compute and display statistics
    for delta, label in [(delta_vs_bitemporal_pct, 'vs Bi-temporal'),
                         (delta_vs_biseasonal_pct, 'vs Bi-seasonal')]:
        pct_above = (delta > 0).mean() * 100
        mean_val = np.mean(delta)
        median_val = np.median(delta)

        print(f"\n{label}:")
        print(f"  Mean: {mean_val:.1f} pp")
        print(f"  Median: {median_val:.1f} pp")
        print(f"  % tiles where annual wins: {pct_above:.1f}%")

    plt.tight_layout()

    # Save figure
    output_path = OUTPUT_DIR / "iou_distribution_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Figure saved to: {output_path}")

    # Also save as PDF for LaTeX
    output_pdf = OUTPUT_DIR / "iou_distribution_comparison.pdf"
    plt.savefig(output_pdf, bbox_inches='tight', facecolor='white')
    print(f"✓ PDF saved to: {output_pdf}")

    plt.close()


if __name__ == "__main__":
    main()
