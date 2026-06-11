#!/usr/bin/env python3
"""
Pretrained-backbone vs precomputed-embeddings illustration for Background section 2.5 (RQ2b).

Two figures, meant to be placed as subfigures (a) and (b). Both contrast where
the foundation model lives in the workflow:

  backbone_mode.pdf      (a) Pretrained backbone. The foundation model lives
                             inside the user pipeline. The user fine-tunes it
                             (or freezes it) together with a task head.
  embedding_mode.pdf     (b) Precomputed embeddings. The foundation model lives
                             upstream at the provider, who releases per-pixel
                             embeddings. The user fits a small downstream model
                             on those embeddings.

Style mirrors the fusion figure: perspective tiles, trapezium models, orthogonal
arrows, semi-transparent fills with crisp black edges. Same Okabe-Ito-adjacent
palette as the fusion figure plus the receptive-field tan (for embeddings as a
data product, not a model).

Notation follows the Methodology (Section 3.1): X is the input, Y-hat the
predicted mask.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch, Polygon
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parents[2] / "REPORT" / "Figures" / "2_Background"

# Palette (matches the fusion figure family).
# Same role across figures uses the same colour: pink = pretrained encoder
# (backbone / foundation model / fusion-figure encoder), mint = decoder (task
# head, fusion-figure decoder), teal = change map. Embeddings are unique to
# this figure (tan). The downstream model in (b) is NOT a decoder, so it gets
# its own pale-blue colour rather than reusing mint.
ALPHA = 0.45
FM_FC = mcolors.to_rgba("#d28383", alpha=ALPHA)   # foundation model / pretrained backbone : soft pink
HEAD_FC = mcolors.to_rgba("#90bd98", alpha=ALPHA) # task head (= decoder for segmentation) : mint
EMB_FC = mcolors.to_rgba("#c99e4b", alpha=ALPHA)  # embeddings (data product) : tan/gold
OUT_FC = mcolors.to_rgba("#245e67", alpha=ALPHA)  # output : dark teal
DOWN_FC = "#D7E8EA"                               # downstream model on embeddings : pale blue (full opacity)

TILE_FC = "white"
TILE_EDGE = "#999999"
ENCL_EDGE = "#666666"
EDGE = "#000000"

# Perspective tile constants (flattened, same as fusion figure).
_N = 4
_COLX, _COLY, _ROWY, _PERSP = 0.50, 0.14, 0.62, 0.0


def _proj(c, r):
    return (c * _COLX, c * _COLY + r * _ROWY * (1.0 - _PERSP * c / _N))


def persp_tile(ax, cx, cy, size, fc=TILE_FC):
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
                                 closed=True, facecolor=fc,
                                 edgecolor=TILE_EDGE, linewidth=0.6))


def box(ax, cx, cy, w, h, text, fc="#EDEDED", fontsize=10):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=fc, edgecolor=EDGE, linewidth=1.2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize)


def trapezium(ax, cx, cy, w, h_left, h_right, text, fc="#EDEDED", fontsize=10):
    xl, xr = cx - w / 2, cx + w / 2
    pts = [(xl, cy + h_left / 2), (xr, cy + h_right / 2),
           (xr, cy - h_right / 2), (xl, cy - h_left / 2)]
    ax.add_patch(Polygon(pts, closed=True, facecolor=fc, edgecolor=EDGE, linewidth=1.2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize)


def arrow(ax, x0, y0, x1, y1, lw=1.4):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=EDGE, linewidth=lw,
                                mutation_scale=14))


def enclosure(ax, x, y, w, h, label):
    """Rounded dashed grey enclosure with a label above the top edge."""
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor="none", edgecolor=ENCL_EDGE,
                                linewidth=1.0, linestyle=(0, (4, 3))))
    ax.text(x + w / 2, y + h + 0.1, label, ha="center", va="bottom", fontsize=10)


def finish(fig, ax, stem):
    ax.set_aspect("equal")
    ax.axis("off")
    ax.autoscale_view()
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_DIR / (stem + '.pdf')}")


def backbone_mode():
    """(a) Pretrained backbone: the FM is inside the user pipeline. Backbone
    is shown as a narrowing trapezium (encoder), the task head as a widening
    trapezium (decoder for segmentation), so together they read as a U-Net
    hourglass. Annotations show what the user actually trains."""
    fig, ax = plt.subplots(figsize=(9.5, 3.5))
    cy = 1.8

    # User pipeline enclosure (a touch taller to fit the training annotations)
    enclosure(ax, 2.3, 0.55, 4.7, 2.35, "user pipeline")

    # Input X as a 2-sheet stack to signal a multi-channel / multi-temporal
    # stack (e.g., bi-temporal Sentinel-2), not literally one image.
    for back in (1, 0):
        persp_tile(ax, 1.0 - back * 0.06, cy + back * 0.06, 1.0)
    ax.text(1.0, -0.05, r"input $\mathbf{X}$", ha="center", va="top", fontsize=10)

    arrow(ax, 1.55, cy, 2.55, cy)
    trapezium(ax, 3.6, cy, 2.0, 1.5, 0.8, "Pretrained\nbackbone", fc=FM_FC, fontsize=10)
    ax.text(3.6, 0.85, "fine-tuned or frozen", ha="center", va="top",
            fontsize=8, style="italic")

    arrow(ax, 4.6, cy, 5.25, cy)
    trapezium(ax, 6.0, cy, 1.5, 0.8, 1.3, "Task\nhead", fc=HEAD_FC, fontsize=9)
    ax.text(6.0, 0.85, "trained from scratch", ha="center", va="top",
            fontsize=8, style="italic")

    arrow(ax, 6.75, cy, 7.35, cy)
    box(ax, 8.0, cy, 1.3, 1.3, "Change\nmap", fc=OUT_FC, fontsize=11)
    ax.text(8.0, -0.05, r"output $\widehat{\mathbf{Y}}$", ha="center", va="top", fontsize=10)

    finish(fig, ax, "backbone_mode")


def embedding_mode():
    """(b) Precomputed embeddings: the FM is upstream at the provider. The user
    pipeline only holds a downstream model trained on the released embeddings.
    Annotation under the downstream model mirrors (a)'s training annotations."""
    fig, ax = plt.subplots(figsize=(13, 3.5))
    cy = 1.8

    # Provider enclosure (left): imagery + foundation model + embeddings
    enclosure(ax, 0.35, 0.55, 5.6, 2.35, "provider (pretraining)")

    # User pipeline enclosure (right): downstream model only.
    # Width 1.9 keeps the enclosure centred on the downstream model at x=8.5
    # (centre = 7.55 + 1.9/2 = 8.5). The outgoing arrow exits the enclosure
    # before reaching the change map, which is intentional.
    enclosure(ax, 7.55, 0.55, 1.9, 2.35, "user pipeline")

    # Provider side: provider's raw imagery as a 2-sheet stack (multi-source,
    # multi-temporal stack the FM was trained on), not literally one image.
    for back in (1, 0):
        persp_tile(ax, 1.0 - back * 0.06, cy + back * 0.06, 1.0)
    ax.text(1.0, -0.05, "imagery", ha="center", va="top", fontsize=10)

    arrow(ax, 1.55, cy, 2.4, cy)
    trapezium(ax, 3.3, cy, 1.5, 1.4, 0.7, "Foundation\nmodel", fc=FM_FC, fontsize=10)
    arrow(ax, 4.05, cy, 4.7, cy)
    # Embeddings as a 4-sheet stack: deeper than the raw-imagery stacks to
    # convey more feature channels per pixel (64 per year for AlphaEarth,
    # versus 9 spectral bands per Sentinel-2 timestep). The user's X is the
    # embeddings, hence "input X" travels with this tile.
    for back in (3, 2, 1, 0):
        persp_tile(ax, 5.2 - back * 0.06, cy + back * 0.06, 1.0, fc=EMB_FC)
    ax.text(5.2, -0.05, r"embeddings (input $\mathbf{X}$)",
            ha="center", va="top", fontsize=10)

    # Released arrow crossing from provider to user pipeline
    arrow(ax, 5.75, cy, 7.85, cy)
    ax.text(6.8, cy + 0.22, "released", ha="center", va="bottom",
            fontsize=9, style="italic")

    # User side: downstream model with training annotation. Pale blue, not
    # mint, because it is not a decoder (no spatial upsampling).
    box(ax, 8.5, cy, 1.3, 0.8, "Downstream\nmodel", fc=DOWN_FC, fontsize=9)
    ax.text(8.5, 0.85, "trained on embeddings", ha="center", va="top",
            fontsize=8, style="italic")

    arrow(ax, 9.15, cy, 10.0, cy)

    box(ax, 10.65, cy, 1.3, 1.3, "Change\nmap", fc=OUT_FC, fontsize=11)
    ax.text(10.65, -0.05, r"output $\widehat{\mathbf{Y}}$", ha="center", va="top", fontsize=10)

    finish(fig, ax, "embedding_mode")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    backbone_mode()
    embedding_mode()


if __name__ == "__main__":
    main()
