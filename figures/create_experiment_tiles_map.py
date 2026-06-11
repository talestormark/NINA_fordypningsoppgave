#!/usr/bin/env python3
"""
Create a map showing only the 53 tiles used in experiments.

This script generates maps showing the train/val/test splits for the thesis.

Output:
    - docs/figures/experiment_tiles_map.pdf (all 53 tiles)
    - docs/figures/experiment_tiles_map_by_split.pdf (colored by split)
"""

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def load_experiment_tiles():
    """Load the 53 tiles used in experiments."""
    print("Loading experiment tiles...")

    # Load split information
    split_info = pd.read_csv("outputs/splits/split_info.csv")

    # Load full tile GeoJSON
    tiles_gdf = gpd.read_file("land_take_bboxes_650m_v1.geojson")

    # Remove duplicates (keep first)
    tiles_gdf = tiles_gdf.drop_duplicates(subset='PLOTID', keep='first')

    # Filter to only experimental tiles
    exp_tiles = tiles_gdf[tiles_gdf['PLOTID'].isin(split_info['refid'])]

    # Merge with split info to get train/val/test labels
    exp_tiles = exp_tiles.merge(
        split_info[['refid', 'split', 'change_ratio', 'change_level']],
        left_on='PLOTID',
        right_on='refid',
        how='left'
    )

    # Load Europe boundaries
    world = gpd.read_file("https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip")

    # Get countries that have experimental tiles
    tile_countries = exp_tiles['country'].unique()
    countries_of_interest = world[world['ADM0_A3'].isin(tile_countries)]
    europe_continent = world[world["CONTINENT"] == "Europe"]
    europe = gpd.GeoDataFrame(
        pd.concat([europe_continent, countries_of_interest]).drop_duplicates(subset='ADMIN'),
        crs=world.crs
    )

    print(f"  Experiment tiles: {len(exp_tiles)}")
    print(f"  Train: {len(exp_tiles[exp_tiles['split']=='train'])}")
    print(f"  Val: {len(exp_tiles[exp_tiles['split']=='val'])}")
    print(f"  Test: {len(exp_tiles[exp_tiles['split']=='test'])}")
    print(f"  Countries: {sorted(exp_tiles['country'].unique())}")

    return exp_tiles, europe


def create_experiment_map_simple(exp_tiles, europe, output_dir):
    """
    Create a simple map with all 53 experimental tiles.
    Similar style to the full dataset map.
    """
    print("\nCreating simple experiment tiles map...")

    # Convert to centroids
    tiles_centroids = exp_tiles.copy()
    tiles_centroids['geometry'] = tiles_centroids.centroid

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    # Plot Europe with light fill
    europe.plot(
        ax=ax,
        facecolor="0.95",
        edgecolor="0.4",
        linewidth=0.6,
        zorder=1
    )

    # Plot tiles as red points
    tiles_centroids.plot(
        ax=ax,
        markersize=20,
        color="tab:red",
        edgecolor="white",
        linewidth=0.5,
        alpha=0.8,
        zorder=2
    )

    # Set extent
    ax.set_xlim(-12, 45)
    ax.set_ylim(34, 72)

    # Add grid and labels
    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)
    ax.set_xticks(range(-10, 45, 10))
    ax.set_yticks(range(35, 75, 10))
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5, zorder=0)


    plt.tight_layout()

    # Save
    pdf_path = output_dir / "experiment_tiles_map.pdf"
    png_path = output_dir / "experiment_tiles_map.png"

    plt.savefig(pdf_path, bbox_inches='tight', dpi=300)
    plt.savefig(png_path, bbox_inches='tight', dpi=300)

    print(f"  Saved: {pdf_path}")
    print(f"  Saved: {png_path}")

    plt.close(fig)


def create_experiment_map_by_split(exp_tiles, europe, output_dir):
    """
    Create a map with tiles colored by split (train/val/test).
    """
    print("\nCreating map colored by split...")

    # Convert to centroids
    tiles_centroids = exp_tiles.copy()
    tiles_centroids['geometry'] = tiles_centroids.centroid

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    # Plot Europe with light fill
    europe.plot(
        ax=ax,
        facecolor="0.95",
        edgecolor="0.4",
        linewidth=0.6,
        zorder=1
    )

    # Define colors for splits
    colors = {
        'train': '#1f77b4',    # Blue
        'val': '#ff7f0e',      # Orange
        'test': '#2ca02c'      # Green
    }

    # Plot each split separately for legend
    for split, color in colors.items():
        subset = tiles_centroids[tiles_centroids['split'] == split]
        subset.plot(
            ax=ax,
            markersize=20,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            alpha=0.8,
            label=f"{split.capitalize()} (n={len(subset)})",
            zorder=2
        )

    # Set extent
    ax.set_xlim(-12, 45)
    ax.set_ylim(34, 72)

    # Add grid and labels
    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)
    ax.set_xticks(range(-10, 45, 10))
    ax.set_yticks(range(35, 75, 10))
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5, zorder=0)

    # Add legend
    ax.legend(
        loc='upper left',
        fontsize=10,
        framealpha=0.9,
        edgecolor='0.5'
    )


    plt.tight_layout()

    # Save
    pdf_path = output_dir / "experiment_tiles_map_by_split.pdf"
    png_path = output_dir / "experiment_tiles_map_by_split.png"

    plt.savefig(pdf_path, bbox_inches='tight', dpi=300)
    plt.savefig(png_path, bbox_inches='tight', dpi=300)

    print(f"  Saved: {pdf_path}")
    print(f"  Saved: {png_path}")

    plt.close(fig)


def create_experiment_map_clean(exp_tiles, europe, output_dir):
    """
    Create a minimal, clean map for thesis.
    """
    print("\nCreating clean experiment tiles map (thesis version)...")

    # Convert to centroids
    tiles_centroids = exp_tiles.copy()
    tiles_centroids['geometry'] = tiles_centroids.centroid

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

    # Plot Europe with light fill
    europe.plot(
        ax=ax,
        facecolor="0.95",
        edgecolor="0.5",
        linewidth=0.5,
        zorder=1
    )

    # Plot tiles as dark red points
    tiles_centroids.plot(
        ax=ax,
        markersize=12,
        color="#d62728",
        edgecolor="white",
        linewidth=0.4,
        alpha=0.85,
        zorder=2
    )

    # Set extent
    ax.set_xlim(-12, 45)
    ax.set_ylim(34, 72)

    # Minimal styling
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(True)

    for spine in ax.spines.values():
        spine.set_edgecolor('0.5')
        spine.set_linewidth(0.5)

    plt.tight_layout(pad=0.1)

    # Save
    pdf_path = output_dir / "experiment_tiles_map_clean.pdf"
    png_path = output_dir / "experiment_tiles_map_clean.png"

    plt.savefig(pdf_path, bbox_inches='tight', dpi=300, pad_inches=0.05)
    plt.savefig(png_path, bbox_inches='tight', dpi=300, pad_inches=0.05)

    print(f"  Saved: {pdf_path} (THESIS VERSION)")
    print(f"  Saved: {png_path}")

    plt.close(fig)


def print_statistics(exp_tiles):
    """Print statistics about experimental tiles."""
    print("\n" + "="*60)
    print("EXPERIMENTAL DATASET STATISTICS")
    print("="*60)

    print(f"\nTotal tiles: {len(exp_tiles)}")

    # By split
    print(f"\nBy split:")
    for split in ['train', 'val', 'test']:
        count = len(exp_tiles[exp_tiles['split'] == split])
        print(f"  {split.capitalize()}: {count}")

    # By country
    print(f"\nBy country:")
    country_counts = exp_tiles['country'].value_counts()
    for country, count in country_counts.items():
        print(f"  {country}: {count}")

    # By change level
    print(f"\nBy change level:")
    level_counts = exp_tiles['change_level'].value_counts()
    for level, count in level_counts.items():
        print(f"  {level.capitalize()}: {count}")

    # Geographic extent
    bounds = exp_tiles.total_bounds
    print(f"\nGeographic extent:")
    print(f"  Longitude: {bounds[0]:.2f}° to {bounds[2]:.2f}°")
    print(f"  Latitude: {bounds[1]:.2f}° to {bounds[3]:.2f}°")


def main():
    """Main execution function."""
    print("="*60)
    print("CREATING EXPERIMENT TILES MAP (53 TILES)")
    print("="*60)

    # Setup output directory
    output_dir = Path("docs/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    exp_tiles, europe = load_experiment_tiles()

    # Print statistics
    print_statistics(exp_tiles)

    # Create maps
    print("\n" + "="*60)
    print("GENERATING MAPS")
    print("="*60)

    create_experiment_map_clean(exp_tiles, europe, output_dir)
    create_experiment_map_simple(exp_tiles, europe, output_dir)
    create_experiment_map_by_split(exp_tiles, europe, output_dir)

    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)
    print("\nGenerated 3 versions:")
    print("  1. experiment_tiles_map_clean.pdf - Clean for thesis (RECOMMENDED)")
    print("  2. experiment_tiles_map.pdf - Simple with title")
    print("  3. experiment_tiles_map_by_split.pdf - Colored by train/val/test")
    print("\nLaTeX caption suggestion:")
    print('  "Geographic distribution of the 53 tiles used in model training')
    print('   and evaluation (train: 37, validation: 8, test: 8)."')


if __name__ == "__main__":
    main()
