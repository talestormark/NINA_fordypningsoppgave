#!/usr/bin/env python3
"""
Temporal-fusion illustration for Background section 2.4.2 (RQ1b).

Two figures, meant to be placed as subfigures (a) and (b). Both share the same
U-Net encoder and decoder (drawn as black boxes); only the point at which the T
timesteps are merged differs:

  fusion_early.pdf  (a) Early fusion: the timesteps are stacked on the channel
                        axis and merged BEFORE a single encoder.
  fusion_late.pdf   (b) Late fusion: a shared-weight (siamese) encoder processes
                        each timestep, and the features are merged AFTER the
                        encoder (by concatenation, pooling, 3D conv, or ConvLSTM).

Notation follows the Methodology (Section 3.1): the whole input is X, the T
timesteps are t_1..t_T, and the predicted mask is Y-hat.

Input timesteps are drawn as perspective square grids in the same style as the
receptive-field figure (vertical side edges, right edge raised and foreshortened),
with a light grid.

This is a preview for layout/iteration; the thesis version is the TikZ port.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch, Polygon
from pathlib import Path

from landtake.paths import REPO_ROOT
OUT_DIR = REPO_ROOT / "REPORT" / "Figures" / "2_Background"

# Per-role palette (semi-transparent so they read as light tints over white,
# with the black edges remaining crisp).
ALPHA = 0.45
ENC_FC = mcolors.to_rgba("#d28383", alpha=ALPHA)  # encoder : soft pink
FUS_FC = mcolors.to_rgba("#cfd0a0", alpha=ALPHA)  # fusion  : pale olive
DEC_FC = mcolors.to_rgba("#90bd98", alpha=ALPHA)  # decoder : mint
OUT_FC = mcolors.to_rgba("#245e67", alpha=ALPHA)  # output  : dark teal

BOX_FC = "#EDEDED"     # generic block fill (kept as a default fallback)
TILE_FC = "white"      # input timestep tiles
TILE_EDGE = "#999999"  # light grid on the tiles
EDGE = "#000000"

# Perspective for the input tiles (same projection as the receptive-field figure).
_N = 4
# PERSP = 0: no foreshortening, so the tiles are clean parallelograms with
# parallel top/bottom edges (vertical sides, right edge raised by COLY).
_COLX, _COLY, _ROWY, _PERSP = 0.50, 0.14, 0.62, 0.0


def _proj(c, r):
    return (c * _COLX, c * _COLY + r * _ROWY * (1.0 - _PERSP * c / _N))


def persp_tile(ax, cx, cy, size):
    """Draw an _N x _N perspective grid (light lines) centred at (cx, cy)."""
    pts = [_proj(c, r) for c in range(_N + 1) for r in range(_N + 1)]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    mx, my = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    sc = size / max(max(xs) - min(xs), max(ys) - min(ys))

    def T(c, r):
        x, y = _proj(c, r)
        return (cx + (x - mx) * sc, cy + (y - my) * sc)

    for c in range(_N):
        for r in range(_N):
            ax.add_patch(Polygon([T(c, r), T(c + 1, r), T(c + 1, r + 1), T(c, r + 1)],
                                 closed=True, facecolor=TILE_FC,
                                 edgecolor=TILE_EDGE, linewidth=0.6))


def box(ax, cx, cy, w, h, text, fc=BOX_FC, fontsize=11):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=fc, edgecolor=EDGE, linewidth=1.2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize)


def trapezium(ax, cx, cy, w, h_left, h_right, text, fc=BOX_FC, fontsize=10):
    """Trapezium with vertical left/right edges. h_left > h_right narrows
    (encoder, downsamples); h_left < h_right widens (decoder, upsamples)."""
    xl, xr = cx - w / 2, cx + w / 2
    pts = [(xl, cy + h_left / 2), (xr, cy + h_right / 2),
           (xr, cy - h_right / 2), (xl, cy - h_left / 2)]
    ax.add_patch(Polygon(pts, closed=True, facecolor=fc, edgecolor=EDGE, linewidth=1.2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize)


def arrow(ax, x0, y0, x1, y1, lw=1.4):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=EDGE, linewidth=lw,
                                mutation_scale=14))


def elbow_arrow(ax, x0, y0, x_corner, y1, lw=1.4):
    """Orthogonal L-shape: horizontal from (x0,y0) to (x_corner,y0),
    then vertical from (x_corner,y0) to (x_corner,y1). Arrowhead at the end."""
    ax.plot([x0, x_corner], [y0, y0], color=EDGE, linewidth=lw, solid_capstyle="butt")
    ax.annotate("", xy=(x_corner, y1), xytext=(x_corner, y0),
                arrowprops=dict(arrowstyle="-|>", color=EDGE, linewidth=lw,
                                mutation_scale=14))


def finish(fig, ax, stem):
    ax.set_aspect("equal")
    ax.axis("off")
    ax.autoscale_view()
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_DIR / (stem + '.pdf')}")


def early():
    fig, ax = plt.subplots(figsize=(9.5, 3.9))
    # Tiles use the previous uneven spacing: t_1 close to t_2, then a gap to t_T
    # (the gap reads as the omitted timesteps). The whole pipeline is aligned to
    # the middle tile's y so the middle arrow stays straight.
    ys = [3.5, 2.4, 0.9]
    pipe_y = 2.4
    labels = [r"$t_0$", r"$t_1$", r"$t_T$"]
    for y, t in zip(ys, labels):
        persp_tile(ax, 0.9, y, 1.0)
        ax.text(0.05, y, t, ha="right", va="center", fontsize=11)
    # orthogonal arrows into the fusion node
    elbow_arrow(ax, 1.55, 3.5, 3.0, pipe_y + 0.42)   # top: right, then down
    arrow(ax, 1.55, pipe_y, 2.58, pipe_y)            # middle: straight
    elbow_arrow(ax, 1.55, 0.9, 3.0, pipe_y - 0.42)   # bottom: right, then up
    ax.text(0.9, 1.65, r"$\vdots$", ha="center", va="center", fontsize=14)
    ax.text(0.9, -0.05, r"input $\mathbf{X}$", ha="center", va="top", fontsize=10)
    ax.text(10.5, -0.05, r"output $\widehat{\mathbf{Y}}$", ha="center", va="top", fontsize=10)
    # fusion node BEFORE the encoder (channel-stack)
    ax.add_patch(plt.Circle((3.0, pipe_y), 0.42, facecolor=FUS_FC, edgecolor=EDGE, linewidth=1.2))
    ax.text(3.0, pipe_y, r"$f$", ha="center", va="center", fontsize=16)
    ax.text(3.0, -0.05, "fusion", ha="center", va="top", fontsize=9)
    arrow(ax, 3.42, pipe_y, 4.65, pipe_y)
    trapezium(ax, 5.5, pipe_y, 1.7, 1.5, 0.7, "Encoder", fc=ENC_FC)
    arrow(ax, 6.35, pipe_y, 7.15, pipe_y)
    trapezium(ax, 8.0, pipe_y, 1.7, 0.7, 1.5, "Decoder", fc=DEC_FC)
    arrow(ax, 8.85, pipe_y, 9.85, pipe_y)
    box(ax, 10.5, pipe_y, 1.3, 1.3, "Change\nmap", fc=OUT_FC, fontsize=11)
    finish(fig, ax, "fusion_early")


def late():
    fig, ax = plt.subplots(figsize=(9.5, 3.9))
    # Uneven spacing: t_0 close to t_1, then a gap to t_T. Pipeline aligned to t_1.
    ys = [3.5, 2.4, 0.9]
    pipe_y = 2.4
    labels = [r"$t_0$", r"$t_1$", r"$t_T$"]
    enc_x = 3.0
    # parallel encoder branches (one narrowing trapezium per timestep), weights shared
    for y, t in zip(ys, labels):
        persp_tile(ax, 0.9, y, 1.0)
        ax.text(0.05, y, t, ha="right", va="center", fontsize=11)
        arrow(ax, 1.55, y, 2.35, y)                                # tile -> encoder
        trapezium(ax, enc_x, y, 1.3, 0.85, 0.4, "Encoder", fc=ENC_FC, fontsize=9)
    # orthogonal arrows from encoders into the fusion node
    elbow_arrow(ax, enc_x + 0.65, 3.5, 5.5, pipe_y + 0.42)   # top: right, then down
    arrow(ax, enc_x + 0.65, pipe_y, 5.08, pipe_y)            # middle: straight
    elbow_arrow(ax, enc_x + 0.65, 0.9, 5.5, pipe_y - 0.42)   # bottom: right, then up
    ax.text(0.9, 1.65, r"$\vdots$", ha="center", va="center", fontsize=14)
    ax.text(enc_x, 1.65, r"$\vdots$", ha="center", va="center", fontsize=14)
    ax.text(0.9, -0.05, r"input $\mathbf{X}$", ha="center", va="top", fontsize=10)
    ax.text(10.5, -0.05, r"output $\widehat{\mathbf{Y}}$", ha="center", va="top", fontsize=10)
    # shared-weights tie around the encoder branches
    ax.add_patch(FancyBboxPatch((enc_x - 0.85, 0.4), 1.7, 3.6,
                                boxstyle="round,pad=0.02,rounding_size=0.06",
                                facecolor="none", edgecolor="#666666",
                                linewidth=1.0, linestyle=(0, (4, 3))))
    ax.text(enc_x, 4.2, "shared weights (siamese)", ha="center", va="bottom", fontsize=9)
    # fusion node AFTER the encoders
    ax.add_patch(plt.Circle((5.5, pipe_y), 0.42, facecolor=FUS_FC, edgecolor=EDGE, linewidth=1.2))
    ax.text(5.5, pipe_y, r"$f$", ha="center", va="center", fontsize=16)
    ax.text(5.5, -0.05, "fusion", ha="center", va="top", fontsize=9)
    arrow(ax, 5.92, pipe_y, 7.15, pipe_y)
    trapezium(ax, 8.0, pipe_y, 1.7, 0.7, 1.5, "Decoder", fc=DEC_FC)
    arrow(ax, 8.85, pipe_y, 9.85, pipe_y)
    box(ax, 10.5, pipe_y, 1.3, 1.3, "Change\nmap", fc=OUT_FC, fontsize=11)
    finish(fig, ax, "fusion_late")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    early()
    late()


if __name__ == "__main__":
    main()
