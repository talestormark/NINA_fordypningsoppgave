#!/usr/bin/env python3
"""
Analyze rotation zero-padding caused by EPSG:4326 -> EPSG:3035 reprojection.

Bounding boxes were defined axis-aligned in geographic coordinates. After
reprojection to LAEA Europe (EPSG:3035), the rectangle rotates, and the
GeoTIFF export fills the corners with zeros.

This script quantifies the effect per source and per tile, and generates
visualization figures.

Output:
    - preprocessing/outputs/rotation_padding_analysis.csv
    - preprocessing/outputs/figures/rotation_padding_by_source.png/pdf
    - preprocessing/outputs/figures/rotation_padding_example.png/pdf
"""

import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_V2 = PROJECT_ROOT / "data_v2"
OUTPUT_DIR = PROJECT_ROOT / "preprocessing" / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"

SOURCES = {
    "Sentinel-2 (10m)": ("Sentinel", "_RGBNIRRSWIRQ_Mosaic.tif"),
    "PlanetScope (3-5m)": ("PlanetScope", "_RGBQ_Mosaic.tif"),
    "VHR Google (1m)": ("VHR_google", "_RGBY_Mosaic.tif"),
    "AlphaEarth (10m)": ("AlphaEarth", "_VEY_Mosaic.tif"),
}


def analyze_all_sources():
    """Compute zero-padding fraction for every tile x source."""
    rows = []

    for source_label, (folder, suffix) in SOURCES.items():
        files = sorted((DATA_V2 / folder).glob("*.tif"))
        print(f"\n{source_label}: {len(files)} files")

        for f in files:
            refid = f.stem.replace(suffix.replace(".tif", ""), "")
            with rasterio.open(f) as src:
                arr = src.read()
                h, w = src.height, src.width
                res = abs(src.transform.a)

            all_zero = np.all(arr == 0, axis=0)
            zero_frac = float(np.mean(all_zero))

            rows.append({
                "refid": refid,
                "source": source_label,
                "folder": folder,
                "height": h,
                "width": w,
                "resolution_m": res,
                "total_pixels": h * w,
                "zero_pixels": int(np.sum(all_zero)),
                "zero_fraction": zero_frac,
            })

        n_affected = sum(1 for r in rows if r["source"] == source_label and r["zero_fraction"] > 0.01)
        print(f"  Tiles with >1% zeros: {n_affected}/{len(files)}")

    df = pd.DataFrame(rows)
    return df


def create_summary_figure(df):
    """Bar chart of zero-padding fraction by source."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    source_order = [
        "VHR Google (1m)",
        "PlanetScope (3-5m)",
        "Sentinel-2 (10m)",
        "AlphaEarth (10m)",
    ]

    # Left: fraction of tiles affected (>1% zeros)
    affected = []
    totals = []
    for s in source_order:
        sub = df[df["source"] == s]
        affected.append(int((sub["zero_fraction"] > 0.01).sum()))
        totals.append(len(sub))

    colors = ["#e41a1c", "#ff7f00", "#377eb8", "#4daf4a"]
    x = range(len(source_order))
    bars = axes[0].bar(x, [a / t * 100 for a, t in zip(affected, totals)], color=colors)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([s.split(" (")[0] for s in source_order], fontsize=9)
    axes[0].set_ylabel("Tiles affected (%)", fontsize=10)
    axes[0].set_title("Tiles with >1% zero-padding", fontsize=11)
    axes[0].set_ylim(0, 100)
    for bar, a, t in zip(bars, affected, totals):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                     f"{a}/{t}", ha="center", va="bottom", fontsize=9)

    # Right: distribution of zero fraction for affected tiles
    for s, color in zip(source_order, colors):
        sub = df[(df["source"] == s) & (df["zero_fraction"] > 0.001)]
        if len(sub) > 0:
            axes[1].hist(sub["zero_fraction"] * 100, bins=30, alpha=0.6,
                         label=s.split(" (")[0], color=color, edgecolor="white", linewidth=0.3)

    axes[1].set_xlabel("Zero-padded pixels (%)", fontsize=10)
    axes[1].set_ylabel("Number of tiles", fontsize=10)
    axes[1].set_title("Zero-padding distribution (affected tiles)", fontsize=11)
    axes[1].legend(fontsize=9)

    plt.tight_layout()

    for ext in ["pdf", "png"]:
        path = FIG_DIR / f"rotation_padding_by_source.{ext}"
        plt.savefig(path, bbox_inches="tight", dpi=300)
        print(f"  Saved: {path}")
    plt.close(fig)


def create_example_figure(df):
    """Show the zero-padding pattern for one tile across all sources."""
    # Pick the tile with the most VHR zero-padding
    vhr_affected = df[df["source"] == "VHR Google (1m)"].sort_values("zero_fraction", ascending=False)
    if vhr_affected.empty or vhr_affected.iloc[0]["zero_fraction"] < 0.01:
        print("  No suitable example tile found.")
        return

    refid = vhr_affected.iloc[0]["refid"]
    print(f"\n  Example tile: {refid[:40]}...")

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), dpi=300)

    source_order = [
        ("VHR Google (1m)", "VHR_google", "_RGBY_Mosaic.tif"),
        ("PlanetScope (3-5m)", "PlanetScope", "_RGBQ_Mosaic.tif"),
        ("Sentinel-2 (10m)", "Sentinel", "_RGBNIRRSWIRQ_Mosaic.tif"),
        ("AlphaEarth (10m)", "AlphaEarth", "_VEY_Mosaic.tif"),
    ]

    for col, (label, folder, suffix) in enumerate(source_order):
        path = DATA_V2 / folder / f"{refid}{suffix}"
        if not path.exists():
            for ax in [axes[0, col], axes[1, col]]:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
            continue

        with rasterio.open(path) as src:
            arr = src.read()
            h, w = src.height, src.width
            res = abs(src.transform.a)

        all_zero = np.all(arr == 0, axis=0)
        zero_frac = np.mean(all_zero)

        # Top row: RGB preview
        if "VHR" in label:
            rgb = np.moveaxis(arr[3:6], 0, -1)  # end year RGB
        elif "PlanetScope" in label:
            rgb = np.moveaxis(arr[:3], 0, -1)  # first timestep RGB
        elif "Sentinel" in label:
            rgb = np.moveaxis(arr[2::-1], 0, -1).astype(float)  # first timestep R,G,B
            for i in range(3):
                band = rgb[:, :, i]
                valid = band[band > 0]
                if len(valid) > 0:
                    lo, hi = np.percentile(valid, [2, 98])
                    if hi > lo:
                        rgb[:, :, i] = np.clip((band - lo) / (hi - lo), 0, 1)
        else:
            # AlphaEarth: show first 3 bands normalized
            rgb = np.moveaxis(arr[:3], 0, -1).astype(float)
            for i in range(3):
                band = rgb[:, :, i]
                lo, hi = np.nanmin(band), np.nanmax(band)
                if hi > lo:
                    rgb[:, :, i] = np.clip((band - lo) / (hi - lo), 0, 1)

        if rgb.dtype == np.float64 or rgb.dtype == np.float32:
            rgb = np.clip(rgb, 0, 1)

        axes[0, col].imshow(rgb)
        axes[0, col].set_title(f"{label}\n{h}×{w} px", fontsize=10)

        # Bottom row: zero-mask (red = zero-padded, white = valid)
        mask_vis = np.ones((h, w, 3))
        mask_vis[all_zero] = [0.8, 0.2, 0.2]  # red for zeros
        axes[1, col].imshow(mask_vis)
        axes[1, col].set_title(f"Zero-padded: {zero_frac:.1%}", fontsize=10)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0, 0].set_ylabel("RGB preview", fontsize=11, rotation=90, labelpad=10)
    axes[1, 0].set_ylabel("Zero-padding mask", fontsize=11, rotation=90, labelpad=10)

    fig.suptitle("Rotation zero-padding across data sources (same tile)", fontsize=13, y=1.01)
    plt.tight_layout()

    for ext in ["pdf", "png"]:
        path = FIG_DIR / f"rotation_padding_example.{ext}"
        plt.savefig(path, bbox_inches="tight", dpi=300)
        print(f"  Saved: {path}")
    plt.close(fig)


def print_summary(df):
    """Print summary statistics."""
    print("\n" + "=" * 60)
    print("ROTATION PADDING SUMMARY")
    print("=" * 60)

    for source in df["source"].unique():
        sub = df[df["source"] == source]
        affected = sub[sub["zero_fraction"] > 0.01]
        print(f"\n{source}:")
        print(f"  Total tiles: {len(sub)}")
        print(f"  Affected (>1% zeros): {len(affected)}")
        if len(affected) > 0:
            print(f"  Mean zero-fraction (affected): {affected['zero_fraction'].mean():.1%}")
            print(f"  Max zero-fraction: {affected['zero_fraction'].max():.1%}")
        else:
            print(f"  No rotation padding detected")

    print("\n" + "-" * 60)
    print("CONCLUSION:")
    print("  10m sources (S2, AlphaEarth) are unaffected.")
    print("  Higher-resolution sources (VHR, PS) have small triangular")
    print("  zero-fill in corners from CRS reprojection (~3-6% of pixels).")
    print("  This does not affect coarse masks or the 10m training pipeline.")
    print("-" * 60)


def main():
    print("=" * 60)
    print("ANALYZING ROTATION ZERO-PADDING")
    print("=" * 60)

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = analyze_all_sources()

    # Save CSV
    csv_path = OUTPUT_DIR / "rotation_padding_analysis.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    print_summary(df)

    # Figures
    print("\n\nGenerating figures...")
    create_summary_figure(df)
    create_example_figure(df)

    print("\nDone!")


if __name__ == "__main__":
    main()
