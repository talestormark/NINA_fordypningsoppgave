#!/usr/bin/env python3
"""
Receptive-field illustration for Background section 2.3.1.

Produces TWO figures, meant to be placed as subfigures (a) and (b):

  receptive_field_cnn.pdf       (a) Convolutional network (U-Net): the output
                                    unit's receptive field grows with depth,
                                    reaching a 5x5 region of the input.
  receptive_field_perpixel.pdf  (b) Per-pixel classifier (MLP / random forest):
                                    the receptive field stays 1x1 at every
                                    layer, so there is no spatial context.

Each shows the region of the input that influences a single output unit,
traced back through the layers with dashed projection lines. The row titles
live in the subfigure captions, not in the drawing.

Style: feature-map grids in light perspective (vertical side edges, the right
edge raised and slightly shorter so each reads as a square seen at an angle).
Palette is Okabe-Ito (colourblind-safe, no red-green pairing).

Outputs go to REPORT/Figures/2_Background/.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon
from pathlib import Path

from landtake.paths import REPO_ROOT
OUT_DIR = REPO_ROOT / "REPORT" / "Figures" / "2_Background"

# Okabe-Ito colourblind-safe palette
RF_FILL = mcolors.to_rgba("#c99e4b", alpha=0.45)  # tan/gold, semi-transparent;
                                                  # the unused colour from the fusion-figure palette
                                                  # (so the two Background figures share a family)
CELL_BG = "white"        # cells outside the receptive field
GRID_EDGE = "#000000"    # grid lines (black)
PROJ_LINE = "#999999"    # projection lines, grey so they stay subtle under the black grid
TEXT = "#000000"

# Perspective projection. Vertical grid lines stay vertical; columns step right
# and lift slightly (right edge higher); each column is foreshortened a little
# toward the right (right edge shorter than the left -> sense of depth). The
# column step is compressed (it is the receding direction) so each map reads
# as a square seen at an angle.
COLX = 0.50      # +1 column: rightward step (receding dir, compressed)
COLY = 0.14      # +1 column: slight upward lift (raises the right edge)
ROWY = 0.62      # +1 row: upward step
PERSP = 0.13     # per-column vertical foreshortening (right edge ~13% shorter)

N = 7                    # every feature map drawn as N x N for a clean comparison
CEN = N // 2             # central index (3)


def S(c, r, ox, oy):
    """Map grid coordinate (c, r) to screen coordinate.

    Vertical lines (constant c) have constant x, so they stay vertical. Columns
    lift to the right via COLY and their height shrinks via the (1 - PERSP*c/N)
    factor, so the right edge is raised and slightly shorter than the left.
    """
    return (ox + c * COLX,
            oy + c * COLY + r * ROWY * (1.0 - PERSP * c / N))


def region_corners(half, ox, oy):
    """Screen corners (TL, TR, BR, BL) of the centred (2*half+1) region."""
    cmin, cmax = CEN - half, CEN + half + 1
    rmin, rmax = CEN - half, CEN + half + 1
    return [S(cmin, rmax, ox, oy), S(cmax, rmax, ox, oy),
            S(cmax, rmin, ox, oy), S(cmin, rmin, ox, oy)]


def draw_layer(ax, ox, oy, half, label):
    """Draw one N x N feature map at (ox, oy) with the centred receptive field
    (size 2*half+1) filled, plus a layer label below. Returns the region's
    screen corners."""
    for r in range(N):
        for c in range(N):
            in_rf = (abs(c - CEN) <= half) and (abs(r - CEN) <= half)
            poly = [S(c, r, ox, oy), S(c + 1, r, ox, oy),
                    S(c + 1, r + 1, ox, oy), S(c, r + 1, ox, oy)]
            ax.add_patch(Polygon(poly, closed=True,
                                 facecolor=RF_FILL if in_rf else CELL_BG,
                                 edgecolor=GRID_EDGE, linewidth=0.6))
    lx, ly = S(N / 2, -0.6, ox, oy)
    ax.text(lx, ly, label, ha="center", va="top", fontsize=10, color=TEXT)
    return region_corners(half, ox, oy)


def connect(ax, left_corners, right_corners):
    """Dashed projection lines between two regions' matching corners."""
    for (xl, yl), (xr, yr) in zip(left_corners, right_corners):
        ax.plot([xl, xr], [yl, yr], color=PROJ_LINE, linewidth=1.0,
                linestyle=(0, (4, 3)), zorder=5)


def make_row(halves, layer_labels, out_stem):
    """Render one three-layer trace-back row (input -> output) to its own file."""
    fig, ax = plt.subplots(figsize=(8.5, 3.0))
    gap = N * COLX + 2.6
    regions = []
    for k, (half, lab) in enumerate(zip(halves, layer_labels)):
        regions.append(draw_layer(ax, k * gap, 0.0, half, lab))
    connect(ax, regions[0], regions[1])
    connect(ax, regions[1], regions[2])

    ax.set_aspect("equal")
    ax.axis("off")
    ax.autoscale_view()
    fig.savefig(OUT_DIR / f"{out_stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{out_stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_DIR / (out_stem + '.pdf')}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    layer_labels = ["input", "after layer 1", "after layer 2 (output)"]
    # (a) CNN: receptive field grows 5x5 (input) -> 3x3 -> 1x1 (output)
    make_row(halves=[2, 1, 0], layer_labels=layer_labels,
             out_stem="receptive_field_cnn")
    # (b) Per-pixel classifier: receptive field stays 1x1
    make_row(halves=[0, 0, 0], layer_labels=layer_labels,
             out_stem="receptive_field_perpixel")


if __name__ == "__main__":
    main()
