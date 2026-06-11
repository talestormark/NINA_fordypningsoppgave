#!/usr/bin/env python3
"""Three-composite Sentinel-2 visualisation for the Data chapter.

Shows the same tile as fig:annotation_masks rendered as three RGB
composites: true colour, false-colour infrared, and a SWIR composite.
Uses the Q2 2024 timestep.

Output:
    REPORT/figures/sentinel2_bands.pdf   (vector, for LaTeX)
    REPORT/figures/sentinel2_bands.png   (raster, for preview)
"""

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from pathlib import Path

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
})

REPO = Path('/cluster/home/tmstorma/NINA_fordypningsoppgave')
OUT_DIR = REPO / 'REPORT' / 'figures'

# Same tile as fig:annotation_masks (picked by create_annotation_masks_figure.py).
TILE = 'a11-377756363169_45-39326226417903'
S2_PATH = REPO / 'data_v2' / 'Sentinel' / f'{TILE}_RGBNIRRSWIRQ_Mosaic.tif'

# Q2 2024: post-change, mid-growing season. Bands 109-117 (1-based).
# Order within a timestep: blue, green, red, R1, R2, R3, nir, swir1, swir2.
Q2_2024_START = 109
BAND_OFFSETS = {
    'blue': 0, 'green': 1, 'red': 2,
    'R1': 3, 'R2': 4, 'R3': 5,
    'nir': 6, 'swir1': 7, 'swir2': 8,
}

COMPOSITES = [
    ('True colour', ['red', 'green', 'blue']),
    ('False-colour infrared', ['nir', 'red', 'green']),
    ('SWIR composite', ['swir2', 'swir1', 'red']),
]


def stretch(arr, p_lo=2, p_hi=98):
    """Per-channel percentile clip and normalise to [0, 1]."""
    lo, hi = np.percentile(arr, [p_lo, p_hi])
    return np.clip((arr - lo) / max(hi - lo, 1e-9), 0, 1)


def compose(bands_dict, channel_names):
    """Stack 3 named bands into an RGB image with per-channel stretch."""
    return np.stack([stretch(bands_dict[c]) for c in channel_names], axis=-1)


def main():
    with rasterio.open(S2_PATH) as ds:
        indices = list(range(Q2_2024_START, Q2_2024_START + 9))
        raw = ds.read(indices)
        descs = [ds.descriptions[i - 1] for i in indices]
    print(f"Loaded {len(raw)} bands from {S2_PATH.name}")
    for d in descs:
        print(f"  {d}")

    bands = {name: raw[off] for name, off in BAND_OFFSETS.items()}

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(11, 4), dpi=300)
    for ax, (title, channels) in zip(axes, COMPOSITES):
        ax.imshow(compose(bands, channels), interpolation='nearest')
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('0.5')
            spine.set_linewidth(0.5)

    plt.tight_layout(pad=0.5)

    for ext in ('pdf', 'png'):
        out = OUT_DIR / f'sentinel2_bands.{ext}'
        plt.savefig(out, bbox_inches='tight', dpi=300, pad_inches=0.05)
        print(f"Saved: {out}")

    plt.close(fig)


if __name__ == '__main__':
    main()
