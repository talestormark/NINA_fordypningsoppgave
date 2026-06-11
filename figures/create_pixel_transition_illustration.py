#!/usr/bin/env python3
"""
Pixel-level illustration of the land-take labelling rule.

Two grids (Time 0, Time 1) with a single pixel highlighted to show the
0 -> 1 transition that defines a land-take pixel.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

from landtake.paths import REPO_ROOT
OUT_DIR = REPO_ROOT / "REPORT" / "figures"


def draw_grid(ax, title, value, label_letter):
    """Draw a 5x5 grid with the centre pixel coloured according to value."""
    n = 5
    # Light grey background for non-highlighted cells
    for i in range(n):
        for j in range(n):
            color = "white"
            ax.add_patch(plt.Rectangle((j, n - 1 - i), 1, 1,
                                       facecolor=color, edgecolor="#888888", linewidth=0.7))

    # Highlight the centre pixel (i, j) = (2, 2)
    cy, cx = 2, 2
    if value == 0:
        face = "#cfe5d9"   # soft green = non-artificial
    else:
        face = "#d97c7c"   # soft red = artificial
    ax.add_patch(plt.Rectangle((cx, n - 1 - cy), 1, 1,
                                facecolor=face, edgecolor="black", linewidth=1.6))
    ax.text(cx + 0.5, n - 1 - cy + 0.5, str(value),
            ha="center", va="center", fontsize=18, fontweight="bold")

    # Axes labels
    ax.text(cx + 0.5, -0.5, "$i$", ha="center", va="top", fontsize=12)
    ax.text(-0.5, n - 1 - cy + 0.5, "$j$", ha="right", va="center", fontsize=12)

    # Mark (i, j) coordinate at centre column/row
    ax.plot([cx + 0.5, cx + 0.5], [-0.15, 0], color="black", linewidth=0.6)
    ax.plot([-0.15, 0], [n - 1 - cy + 0.5, n - 1 - cy + 0.5], color="black", linewidth=0.6)

    ax.set_xlim(-1, n + 0.2)
    ax.set_ylim(-1, n + 0.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=12, pad=10)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4.5))
    draw_grid(axes[0], r"Time $t_0$", value=0, label_letter="0")
    draw_grid(axes[1], r"Time $t_1$", value=1, label_letter="1")

    # Arrow connecting the two grids
    fig.patches.append(mpatches.FancyArrowPatch(
        (0.49, 0.5), (0.55, 0.5),
        transform=fig.transFigure,
        arrowstyle="->", mutation_scale=22,
        color="black", linewidth=1.5,
    ))

    # Legend
    handles = [
        mpatches.Patch(facecolor="#cfe5d9", edgecolor="black", label="0 = Non-artificial"),
        mpatches.Patch(facecolor="#d97c7c", edgecolor="black", label="1 = Artificial"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, 0.98))

    plt.subplots_adjust(top=0.85, bottom=0.05, left=0.05, right=0.95)
    fig.savefig(OUT_DIR / "pixel_transition_illustration.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "pixel_transition_illustration.png", dpi=200, bbox_inches="tight")
    print(f"Saved: {OUT_DIR / 'pixel_transition_illustration.pdf'}")
    print(f"Saved: {OUT_DIR / 'pixel_transition_illustration.png'}")


if __name__ == "__main__":
    main()
