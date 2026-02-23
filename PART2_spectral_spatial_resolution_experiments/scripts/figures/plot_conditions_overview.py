#!/usr/bin/env python3
"""
Generate a visual overview of input compositions for all Part II experiment
conditions as a horizontal stacked bar chart with a broken x-axis.

Left axis covers 0–16 channels (most conditions); right axis covers 58–76
channels (D2, D3 with 64 AlphaEarth features). The break is shown with
diagonal hatch marks.

Output: docs/REPORT/Images/Experiment_conditions_overview.pdf

Usage:
    python plot_conditions_overview.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
OUTPUT_DIR = (
    REPO_ROOT
    / "PART2_spectral_spatial_resolution_experiments"
    / "docs"
    / "REPORT"
    / "Images"
)

# ---------------------------------------------------------------------------
# Paul Tol colorblind-safe palette
# ---------------------------------------------------------------------------
COLORS = {
    "S2 Vis (RGB)": "#4477AA",
    "S2 NIR":         "#228833",
    "S2 Red-Edge":    "#EE6677",
    "S2 SWIR":        "#AA3377",
    "Spectral Idx":   "#CCBB44",
    "PS RGB":         "#66CCEE",
    "AlphaEarth":     "#EE8866",
}

# ---------------------------------------------------------------------------
# Condition definitions: list of (group_name, width) tuples per condition
# ---------------------------------------------------------------------------
CONDITIONS = {
    "A1": [("S2 Vis (RGB)", 3)],
    "A2": [("S2 Vis (RGB)", 3), ("S2 NIR", 1)],
    "A3": [("S2 Vis (RGB)", 3), ("S2 NIR", 1), ("S2 Red-Edge", 3), ("S2 SWIR", 2)],
    "A4": [("S2 Vis (RGB)", 3), ("S2 NIR", 1), ("S2 Red-Edge", 3), ("S2 SWIR", 2),
           ("Spectral Idx", 4)],
    "A5": [("Spectral Idx", 4)],
    "C2": [("PS RGB", 3)],
    "C3": [("S2 Vis (RGB)", 3), ("PS RGB", 3)],
    "D2": [("AlphaEarth", 64)],
    "D3": [("S2 Vis (RGB)", 3), ("S2 NIR", 1), ("S2 Red-Edge", 3), ("S2 SWIR", 2),
           ("AlphaEarth", 64)],
}

# Display order top-to-bottom
DISPLAY_ORDER = ["A1", "A2", "A3", "A4", "A5", "C2", "C3", "D2", "D3"]
BLOCKS = {"A": ["A1", "A2", "A3", "A4", "A5"],
          "C": ["C2", "C3"],
          "D": ["D2", "D3"]}

# Legend order
LEGEND_ORDER = [
    "S2 Vis (RGB)", "S2 NIR", "S2 Red-Edge", "S2 SWIR",
    "Spectral Idx", "PS RGB", "AlphaEarth",
]

# ---------------------------------------------------------------------------
# Axis break parameters
# ---------------------------------------------------------------------------
LEFT_XLIM = (0, 16)
RIGHT_XLIM = (58, 78)
BAR_HEIGHT = 0.6


def _get_block(cond):
    """Return block key for a condition."""
    for b, members in BLOCKS.items():
        if cond in members:
            return b
    return None


def _y_positions(order):
    """Compute y positions with extra spacing between blocks."""
    positions = []
    y = 0
    prev_block = None
    for cond in order:
        block = _get_block(cond)
        if prev_block is not None and block != prev_block:
            y -= 0.8  # extra gap between blocks
        positions.append(y)
        prev_block = block
        y -= 1
    return positions


def _draw_break_marks(ax_left, ax_right):
    """Draw diagonal break marks between two subplots."""
    d = 0.015
    kwargs = dict(transform=ax_left.transAxes, color="0.3",
                  clip_on=False, linewidth=0.8)
    ax_left.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    ax_left.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    kwargs["transform"] = ax_right.transAxes
    ax_right.plot((-d, +d), (-d, +d), **kwargs)
    ax_right.plot((-d, +d), (1 - d, 1 + d), **kwargs)


def main():
    y_positions = _y_positions(DISPLAY_ORDER)
    y_map = dict(zip(DISPLAY_ORDER, y_positions))

    # --- Figure setup: two subplots sharing y-axis ---
    fig, (ax_l, ax_r) = plt.subplots(
        1, 2,
        figsize=(7, 3.5),
        sharey=True,
        gridspec_kw={"width_ratios": [3, 1], "wspace": 0.02},
    )

    # Track groups for legend
    drawn_groups = set()
    legend_handles = {}

    for cond in DISPLAY_ORDER:
        y = y_map[cond]
        segments = CONDITIONS[cond]
        total = sum(w for _, w in segments)
        is_anchor = (cond == "A3")

        left = 0
        for group, width in segments:
            color = COLORS[group]
            edge = "black" if is_anchor else "white"
            lw = 1.2 if is_anchor else 0.5

            # Draw every segment on BOTH axes — xlim clipping handles visibility.
            # This ensures bars that span the break (D2, D3) render correctly.
            bar_l = ax_l.barh(y, width, left=left, height=BAR_HEIGHT,
                              color=color, edgecolor=edge, linewidth=lw,
                              zorder=2)
            ax_r.barh(y, width, left=left, height=BAR_HEIGHT,
                      color=color, edgecolor=edge, linewidth=lw,
                      zorder=2)

            if group not in drawn_groups:
                # Create a dedicated handle with neutral edge so the legend
                # swatches are not affected by A3's black anchor border.
                h = ax_l.barh(0, 0, height=0, color=color,
                              edgecolor="white", linewidth=0.5)
                legend_handles[group] = h
                drawn_groups.add(group)

            left += width

        # --- C=N label at end of each bar ---
        label_text = f"$C$={total}"
        if total <= LEFT_XLIM[1]:
            ax_l.text(total + 0.4, y, label_text,
                      va="center", ha="left", fontsize=7, color="0.2")
        else:
            ax_r.text(total + 0.5, y, label_text,
                      va="center", ha="left", fontsize=7, color="0.2")

    # --- Y-axis labels ---
    labels = []
    for cond in DISPLAY_ORDER:
        if cond == "A3":
            labels.append(r"$\bf{A3}$")
        else:
            labels.append(cond)
    ax_l.set_yticks(y_positions)
    ax_l.set_yticklabels(labels, fontsize=8.5)

    # --- Block group headers in separator gaps ---
    # Place a small italic label above each block's first condition,
    # sitting in the gap between blocks.
    block_headers = [
        ("A",  "Block A \u2014 Spectral bands"),
        ("C",  "Block C \u2014 Spatial resolution"),
        ("D",  "Block D \u2014 Alternative features"),
    ]
    trans = ax_l.get_yaxis_transform()
    for block_key, header_text in block_headers:
        members = BLOCKS[block_key]
        y_top = y_map[members[0]]  # first member (top of block)
        # Place label above the top bar of the block, in the gap
        ax_l.text(0.0, y_top + 0.48, header_text,
                  va="bottom", ha="left", fontsize=6, color="0.45",
                  style="italic", transform=trans, clip_on=False)

    # --- Horizontal separator lines between blocks ---
    prev_block = None
    for i, cond in enumerate(DISPLAY_ORDER):
        block = _get_block(cond)
        if prev_block is not None and block != prev_block:
            sep_y = (y_positions[i - 1] + y_positions[i]) / 2
            ax_l.axhline(sep_y, color="0.85", linewidth=0.5, linestyle="-", zorder=0)
            ax_r.axhline(sep_y, color="0.85", linewidth=0.5, linestyle="-", zorder=0)
        prev_block = block

    # --- Axis limits and styling ---
    ax_l.set_xlim(*LEFT_XLIM)
    ax_r.set_xlim(*RIGHT_XLIM)
    ax_l.set_ylim(min(y_positions) - 0.8, max(y_positions) + 1.1)

    # X-axis label spanning both axes
    ax_l.set_xlabel("Channels per timestep", fontsize=9)
    ax_l.xaxis.set_label_coords(0.65, -0.08, transform=ax_l.transAxes)

    # X ticks
    ax_l.set_xticks([0, 3, 4, 6, 9, 13])
    ax_r.set_xticks([64, 73])

    # Hide spines at the break
    ax_l.spines["right"].set_visible(False)
    ax_r.spines["left"].set_visible(False)
    ax_r.tick_params(left=False)

    # Draw break marks
    _draw_break_marks(ax_l, ax_r)

    # --- Legend ---
    handles = [legend_handles[g] for g in LEGEND_ORDER if g in legend_handles]
    labels_leg = [g for g in LEGEND_ORDER if g in legend_handles]
    fig.legend(handles, labels_leg, loc="lower center",
               ncol=len(labels_leg), fontsize=7, frameon=False,
               bbox_to_anchor=(0.45, -0.02))

    # --- Final layout ---
    fig.subplots_adjust(bottom=0.15, left=0.08, right=0.97, top=0.96)

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = OUTPUT_DIR / "Experiment_conditions_overview.pdf"
    out_png = OUTPUT_DIR / "Experiment_conditions_overview.png"
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")
    plt.close(fig)


if __name__ == "__main__":
    main()
