#!/usr/bin/env python3
"""
Create a map of Europe showing all data_v2 tile locations.

Adapted from create_europe_map.py for the expanded 260-tile dataset.
Uses the filtered geojson from data_v2/.

Output:
    - REPORT/figures/study_area_map.pdf (vector, for LaTeX)
    - REPORT/figures/study_area_map.png (raster, for preview)
"""

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
})

PROJECT_ROOT = Path(__file__).parent.parent.parent
GEOJSON_PATH = PROJECT_ROOT / "data_v2" / "land_take_bboxes_650m_v1_filtered.geojson"
MASK_ANALYSIS_PATH = PROJECT_ROOT / "preprocessing" / "outputs" / "mask_analysis.csv"
OUTPUT_DIR = PROJECT_ROOT / "REPORT" / "figures"


def load_data():
    """Load tile locations and Europe boundaries."""
    print("Loading data...")

    tiles_gdf = gpd.read_file(GEOJSON_PATH)

    # Remove duplicate tiles (same PLOTID, different change_type)
    print(f"  Raw features: {len(tiles_gdf)}, unique PLOTIDs: {tiles_gdf['PLOTID'].nunique()}")
    tiles_gdf = tiles_gdf.drop_duplicates(subset='PLOTID', keep='first')
    print(f"  After deduplication: {len(tiles_gdf)} unique tiles")

    # Exclude 4 non-trainable tiles (missing S2 or coarse mask)
    excluded = {
        "a-0-77133618972711_46-45684360844514",
        "a3-90419320521538_51-81608889925567",
        "a4-62484266638608_51-98215379896622",
        "a5-37542343603257_51-67195795245962",
    }
    tiles_gdf = tiles_gdf[~tiles_gdf['PLOTID'].isin(excluded)]
    print(f"  After excluding non-trainable: {len(tiles_gdf)} tiles")

    # Load Europe boundaries from Natural Earth (10m = finest resolution, for crisp borders)
    url = "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip"
    print("  Downloading Natural Earth boundaries...")
    world = gpd.read_file(url)

    # Get countries with tiles + all European countries for context
    tile_countries_iso3 = tiles_gdf['country'].unique()
    countries_of_interest = world[world['ADM0_A3'].isin(tile_countries_iso3)]
    europe_continent = world[world["CONTINENT"] == "Europe"]

    europe = gpd.GeoDataFrame(
        pd.concat([europe_continent, countries_of_interest]).drop_duplicates(subset='ADMIN'),
        crs=world.crs
    )

    print(f"  Loaded {len(europe)} country boundaries")
    print(f"  Tile countries: {sorted(tile_countries_iso3)}")

    return tiles_gdf, europe


def create_clean_thesis_map(tiles_gdf, europe):
    """Create a minimal, clean map for thesis inclusion. No title, minimal styling."""
    print("\nCreating clean thesis map...")

    tiles_centroids = tiles_gdf.copy()
    tiles_centroids['geometry'] = tiles_centroids.centroid

    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

    europe.plot(
        ax=ax,
        facecolor="0.95",
        edgecolor="0.5",
        linewidth=0.5,
        zorder=1
    )

    tiles_centroids.plot(
        ax=ax,
        markersize=4,
        color="#d62728",
        edgecolor="white",
        linewidth=0.2,
        alpha=0.85,
        zorder=2
    )

    ax.set_xlim(-12, 45)
    ax.set_ylim(34, 72)

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(True)

    for spine in ax.spines.values():
        spine.set_edgecolor('0.5')
        spine.set_linewidth(0.5)

    plt.tight_layout(pad=0.1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "png"]:
        path = OUTPUT_DIR / f"study_area_map.{ext}"
        plt.savefig(path, bbox_inches='tight', dpi=300, pad_inches=0.05)
        print(f"  Saved: {path}")

    plt.close(fig)


def create_simple_map(tiles_gdf, europe):
    """Create a map with grid lines and axis labels."""
    print("\nCreating simple map...")

    tiles_centroids = tiles_gdf.copy()
    tiles_centroids['geometry'] = tiles_centroids.centroid

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    europe.plot(
        ax=ax,
        facecolor="0.95",
        edgecolor="0.4",
        linewidth=0.6,
        zorder=1
    )

    tiles_centroids.plot(
        ax=ax,
        markersize=8,
        color="tab:red",
        edgecolor="white",
        linewidth=0.3,
        alpha=0.8,
        zorder=2
    )

    ax.set_xlim(-12, 45)
    ax.set_ylim(34, 72)

    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)
    ax.set_xticks(range(-10, 50, 10))
    ax.set_yticks(range(35, 75, 10))
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5, zorder=0)

    plt.tight_layout()

    for ext in ["pdf", "png"]:
        path = OUTPUT_DIR / f"study_area_map_simple.{ext}"
        plt.savefig(path, bbox_inches='tight', dpi=300)
        print(f"  Saved: {path}")

    plt.close(fig)


def create_colored_map(tiles_gdf, europe):
    """Create a map colored by change level (low / moderate / high)."""
    print("\nCreating colored map (by change level)...")

    tiles_centroids = tiles_gdf.copy()
    tiles_centroids['geometry'] = tiles_centroids.centroid

    # Join change level from the coarse mask analysis (matches the change-ratio
    # table in the thesis: low <5%, moderate 5-30%, high >=30%).
    mask_stats = pd.read_csv(MASK_ANALYSIS_PATH)
    mask_stats = (mask_stats[mask_stats['mask_type'] == 'coarse']
                  [['refid', 'change_level']]
                  .drop_duplicates('refid'))
    tiles_centroids = tiles_centroids.merge(
        mask_stats, left_on='PLOTID', right_on='refid', how='left'
    )
    missing = tiles_centroids['change_level'].isna().sum()
    if missing:
        print(f"  WARNING: {missing} tiles have no change level; dropping them from the map")
        tiles_centroids = tiles_centroids[tiles_centroids['change_level'].notna()]

    colors = {
        # Okabe-Ito colourblind-safe palette: distinct under red-green CB and
        # all saturated enough to read on the light-grey basemap.
        'low': '#0072B2',       # blue
        'moderate': '#E69F00',  # orange
        'high': '#D55E00',      # vermillion
    }

    display_labels = {
        'low': r"Low ($<$5%)",
        'moderate': "Moderate (5-30%)",
        'high': r"High ($\geq$30%)",
    }

    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

    europe.plot(
        ax=ax,
        facecolor="0.95",
        edgecolor="0.5",
        linewidth=0.5,
        zorder=1
    )

    # Plot low -> moderate -> high so the rare high-change tiles sit on top
    for level in ['low', 'moderate', 'high']:
        subset = tiles_centroids[tiles_centroids['change_level'] == level]
        if len(subset) == 0:
            continue
        subset.plot(
            ax=ax,
            markersize=8,
            color=colors[level],
            edgecolor="white",
            linewidth=0.2,
            alpha=0.85,
            zorder=2,
            label=display_labels[level],
        )

    ax.set_xlim(-12, 45)
    ax.set_ylim(34, 72)

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(True)

    for spine in ax.spines.values():
        spine.set_edgecolor('0.5')
        spine.set_linewidth(0.5)

    ax.legend(loc='upper left', frameon=False, fontsize=10)

    plt.tight_layout(pad=0.1)

    for ext in ["pdf", "png"]:
        path = OUTPUT_DIR / f"study_area_map_colored.{ext}"
        plt.savefig(path, bbox_inches='tight', dpi=300, pad_inches=0.05)
        print(f"  Saved: {path}")

    plt.close(fig)


def print_statistics(tiles_gdf):
    """Print dataset statistics."""
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)

    print(f"\nTotal tiles: {len(tiles_gdf)}")
    print(f"Countries: {tiles_gdf['country'].nunique()}")

    print("\nTiles per country:")
    for country, count in tiles_gdf['country'].value_counts().items():
        print(f"  {country}: {count}")

    print("\nChange types:")
    for ct, count in tiles_gdf['change_type'].value_counts().items():
        print(f"  {ct}: {count}")

    bounds = tiles_gdf.total_bounds
    print(f"\nGeographic extent:")
    print(f"  Longitude: {bounds[0]:.2f} to {bounds[2]:.2f}")
    print(f"  Latitude: {bounds[1]:.2f} to {bounds[3]:.2f}")


def main():
    print("=" * 60)
    print("CREATING EUROPE MAP — data_v2 (260 tiles)")
    print("=" * 60)

    tiles_gdf, europe = load_data()
    print_statistics(tiles_gdf)

    print("\n" + "=" * 60)
    print("GENERATING MAPS")
    print("=" * 60)

    create_clean_thesis_map(tiles_gdf, europe)
    create_simple_map(tiles_gdf, europe)
    create_colored_map(tiles_gdf, europe)

    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)
    print("\nGenerated 3 versions in REPORT/figures/:")
    print("  1. study_area_map.pdf      — Clean for thesis (RECOMMENDED)")
    print("  2. study_area_map_simple    — With grid and axis labels")
    print("  3. study_area_map_colored   — Colored by change level")


if __name__ == "__main__":
    main()
