#!/usr/bin/env python3
"""U-Net backbone schematic for the Methodology chapter.

A U-shape diagram showing the ResNet-50 encoder, the U-Net decoder, and the
skip connections between them. Channels and spatial sizes annotated per stage.

Output:
    REPORT/Figures/3_Methodology/unet_backbone.{pdf,png}
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
})

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "REPORT" / "Figures" / "3_Methodology"


def block(ax, x, y, w, h, label, facecolor, edgecolor="black", fontsize=7):
    rect = mpatches.FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=0.8,
    )
    ax.add_patch(rect)
    ax.text(x, y, label, ha="center", va="center", fontsize=fontsize)


def arrow(ax, x1, y1, x2, y2, color="black", linestyle="-", linewidth=1):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="->", color=color, linestyle=linestyle, linewidth=linewidth,
        ),
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.set_aspect("equal")
    ax.axis("off")

    enc_x, dec_x = 2.5, 7.5
    bw, bh = 2.2, 0.6

    # Levels (top → bottom): (y, encoder_label, decoder_label)
    levels = [
        (5.0, "Conv1 (channel adapter)\n64 ch, 32$\\times$32",
              "Decoder 5\n16 ch, 64$\\times$64"),
        (4.0, "Stage 1\n256 ch, 16$\\times$16",
              "Decoder 4\n32 ch, 32$\\times$32"),
        (3.0, "Stage 2\n512 ch, 8$\\times$8",
              "Decoder 3\n64 ch, 16$\\times$16"),
        (2.0, "Stage 3\n1024 ch, 4$\\times$4",
              "Decoder 2\n128 ch, 8$\\times$8"),
        (1.0, "Stage 4\n2048 ch, 2$\\times$2",
              "Decoder 1\n256 ch, 4$\\times$4"),
    ]

    enc_color = "#cce5ff"   # light blue
    dec_color = "#ffe5cc"   # light orange
    io_color  = "#d4f0d4"   # light green

    # Input and output boxes
    block(ax, enc_x, 6.3, bw, bh, "Input\n$C_{\\text{in}}\\times 64\\times 64$",
          io_color, fontsize=8)
    block(ax, dec_x, 6.3, bw, bh,
          "Output (logit)\n$1 \\times 64 \\times 64$",
          io_color, fontsize=8)

    # Encoder + decoder boxes per level
    for y, enc_lbl, dec_lbl in levels:
        block(ax, enc_x, y, bw, bh, enc_lbl, enc_color)
        block(ax, dec_x, y, bw, bh, dec_lbl, dec_color)

    # Skip connections at intermediate levels (grey dashed). The deepest
    # encoder feature (bottleneck) feeds into the first decoder stage as the
    # main data path, not as a skip, so it is drawn as a solid arrow below.
    for y, _, _ in levels[:-1]:
        arrow(ax, enc_x + bw / 2, y, dec_x - bw / 2, y,
              color="gray", linestyle="--", linewidth=1)

    # Encoder vertical down arrows
    for i in range(len(levels) - 1):
        y1 = levels[i][0] - bh / 2
        y2 = levels[i + 1][0] + bh / 2
        arrow(ax, enc_x, y1, enc_x, y2)

    # Decoder vertical up arrows
    for i in range(len(levels) - 1, 0, -1):
        y1 = levels[i][0] + bh / 2
        y2 = levels[i - 1][0] - bh / 2
        arrow(ax, dec_x, y1, dec_x, y2)

    # Bottleneck connector (encoder bottom → decoder bottom)
    arrow(ax, enc_x + bw / 2, levels[-1][0],
          dec_x - bw / 2, levels[-1][0], linewidth=1.5)

    # Input → first encoder
    arrow(ax, enc_x, 6.3 - bh / 2, enc_x, levels[0][0] + bh / 2)
    # Last decoder → output (with seg head label nearby)
    arrow(ax, dec_x, levels[0][0] + bh / 2, dec_x, 6.3 - bh / 2)
    ax.text(dec_x - 0.05, (levels[0][0] + 6.3) / 2, "1$\\times$1 conv",
            ha="right", va="center", fontsize=7, style="italic")

    plt.tight_layout(pad=0.5)
    for ext in ("pdf", "png"):
        out = OUT_DIR / f"unet_backbone.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=300, pad_inches=0.05)
        print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
