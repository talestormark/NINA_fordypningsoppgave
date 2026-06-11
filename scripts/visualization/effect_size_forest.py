#!/usr/bin/env python3
"""
Cross-RQ effect-size forest plot for Discussion 6.2 ("more is not always better").

One row per headline pairwise comparison, showing the mean per-tile IoU
difference (pp) with its 95% bootstrap CI, grouped by research question.
Values are transcribed verbatim from the stats tables in
Sections/5_Experiments_and_Results.tex (tab:rq1a_stats, tab:rq1b_stats,
tab:rq2a_stats, tab:rq2b_dataeff_stats, tab:rq3a/b/c_stats).

Significance is by the Holm-adjusted permutation p-value (the thesis criterion):
filled vermillion = p_adj < 0.05, grey = not significant. Two comparisons
(RGB vs 9-band, LateFusion vs EarlyFusion) have 95% CIs that exclude zero but
adjusted p just above 0.05, so they are shown as not significant.

The Random Forest random-vs-balanced collapse (RQ3d, -16 pp) is intentionally
omitted: it is a robustness result, not a "does more help" comparison, and its
range would compress this plot.

Output: REPORT/Figures/7_Discussion/effect_size_forest.{pdf,png}
"""

from pathlib import Path

import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
})

OUTPUT_DIR = Path("/cluster/home/tmstorma/NINA_fordypningsoppgave/REPORT/Figures/7_Discussion")
OUTPUT_BASENAME = "effect_size_forest"

SIG_COLOR = (0.835, 0.369, 0.0)   # Okabe-Ito vermillion: significant
NS_COLOR = (0.40, 0.40, 0.40)     # grey: not significant

# (group, label, delta_pp, ci_lo, ci_hi, p_adj)  -- top to bottom
ROWS = [
    ("RQ1: temporal sampling and fusion", "Annual vs bi-temporal",        -0.81, -2.49,  0.78, 0.345),
    ("RQ1: temporal sampling and fusion", "Annual vs bi-seasonal",         1.67,  0.28,  3.03, 0.034),
    ("RQ1: temporal sampling and fusion", "LateFusion vs EarlyFusion",    -2.22, -3.86, -0.62, 0.054),
    ("RQ2: input representation",         "RGB vs 9-band",                 -3.5,  -6.4,  -0.9,  0.052),
    ("RQ2: input representation",         "RGB+NIR vs 9-band",              0.7,  -0.5,   1.9,  0.558),
    ("RQ2: input representation",         "9-band + indices vs 9-band",    -0.3,  -1.3,   0.6,  0.558),
    ("RQ2: input representation",         "AlphaEarth vs 9-band (full data)", -2.2, -4.3, -0.1, 0.266),
    ("RQ3: annotation efficiency",        "10 points vs dense (AlphaEarth)",  -2.33, -3.71, -1.00, 0.008),
    ("RQ3: annotation efficiency",        "20 points vs dense (AlphaEarth)",  -2.06, -3.64, -0.54, 0.043),
    ("RQ3: annotation efficiency",        "30 points vs dense (AlphaEarth)",  -2.07, -3.82, -0.46, 0.045),
    ("RQ3: annotation efficiency",        "50 points vs dense (AlphaEarth)",  -0.91, -2.28,  0.40, 0.371),
    ("RQ3: annotation efficiency",        "200 points vs dense (AlphaEarth)", -0.04, -1.18,  1.16, 0.947),
    ("RQ3: annotation efficiency",        "50 points vs dense (Sentinel-2)",  -1.5,  -3.6,   0.5,  0.459),
]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Assign y positions top -> bottom, with a gap + header between groups.
    entries = []   # (y, delta, lo, hi, padj, label)
    headers = []   # (y, group)
    y = 0.0
    prev = None
    for (g, label, d, lo, hi, p) in ROWS:
        if g != prev:
            if prev is not None:
                y -= 1.0          # gap between groups
            headers.append((y + 0.9, g))
            prev = g
        entries.append((y, d, lo, hi, p, label))
        y -= 1.0
    y_bottom = y

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.axvline(0.0, color="0.6", linestyle="--", linewidth=0.8, zorder=0)

    for (yy, d, lo, hi, p, label) in entries:
        c = SIG_COLOR if p < 0.05 else NS_COLOR
        ax.errorbar(d, yy, xerr=[[d - lo], [hi - d]], fmt="o", color=c,
                    ecolor=c, elinewidth=1.3, capsize=3, markersize=5, zorder=3)

    # Comparison labels and bold group headers share the left margin as y-tick
    # labels (right-aligned), so headers never cross into the plot area.
    header_names = {g for (_, g) in headers}
    tick_pos = [e[0] for e in entries] + [yh for (yh, _) in headers]
    tick_lab = [e[5] for e in entries] + [g for (_, g) in headers]
    ax.set_yticks(tick_pos)
    ax.set_yticklabels(tick_lab, fontsize=9)
    for t in ax.get_yticklabels():
        if t.get_text() in header_names:
            t.set_fontweight("bold")
            t.set_fontsize(9.5)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(y_bottom - 0.6, 1.4)

    ax.set_xlabel(r"Per-tile IoU difference $\Delta$ (percentage points)", fontsize=10)
    ax.tick_params(axis="x", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend below the plot, outside the data area.
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color=SIG_COLOR, linestyle="-",
               markersize=5, label=r"Significant ($p_{\mathrm{adj}} < 0.05$)"),
        Line2D([0], [0], marker="o", color=NS_COLOR, linestyle="-",
               markersize=5, label="Not significant"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.10),
              ncol=2, frameon=False, fontsize=9)

    plt.tight_layout()
    pdf = OUTPUT_DIR / f"{OUTPUT_BASENAME}.pdf"
    png = OUTPUT_DIR / f"{OUTPUT_BASENAME}.png"
    plt.savefig(pdf, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.05)
    print(f"Saved {pdf}")
    print(f"Saved {png}")


if __name__ == "__main__":
    main()
