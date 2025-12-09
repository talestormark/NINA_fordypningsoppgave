#!/usr/bin/env python3
"""
Create a map of Europe showing all land-take detection tile locations.

This script generates a publication-quality figure showing the geographic
distribution of study tiles across Europe for the thesis.

Output:
    - docs/figures/study_area_map.pdf (vector, for LaTeX)
    - docs/figures/study_area_map.png (raster, for preview)
"""

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_data():
    """Load tile locations and Europe boundaries."""
    print("Loading data...")

    # Load tile bounding boxes from GeoJSON
    tiles_gdf = gpd.read_file("land_take_bboxes_650m_v1.geojson")

    # Remove duplicate tiles (same PLOTID, different change_type)
    print(f"  Raw tiles: {len(tiles_gdf)} rows, {tiles_gdf['PLOTID'].nunique()} unique tiles")
    tiles_gdf = tiles_gdf.drop_duplicates(subset='PLOTID', keep='first')
    print(f"  After deduplication: {len(tiles_gdf)} unique tiles")

    # Load Europe + relevant countries from Natural Earth
    try:
        # Try to load from Natural Earth URL directly
        url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
        print("  Downloading Natural Earth boundaries...")
        world = gpd.read_file(url)

        # Get all countries that have tiles in our dataset
        tile_countries_iso3 = tiles_gdf['country'].unique()

        # Map 3-letter codes to country names (use ADM0_A3, not ISO_A3 which has -99 for some countries)
        countries_of_interest = world[world['ADM0_A3'].isin(tile_countries_iso3)]

        # Also include traditional Europe for context (but Turkey is in Asia)
        europe_continent = world[world["CONTINENT"] == "Europe"]

        # Combine both (union)
        europe = gpd.GeoDataFrame(
            pd.concat([europe_continent, countries_of_interest]).drop_duplicates(subset='ADMIN'),
            crs=world.crs
        )

        print(f"  Loaded {len(europe)} country boundaries")

    except Exception as e:
        print(f"  Warning: Could not load Natural Earth data: {e}")
        print("  Creating simple Europe boundary from tile extent...")
        # Create a simple rectangle boundary from tile extent
        bounds = tiles_gdf.total_bounds
        # Add buffer for visualization
        minx, miny, maxx, maxy = bounds[0]-5, bounds[1]-5, bounds[2]+5, bounds[3]+5
        from shapely.geometry import box
        europe_box = gpd.GeoDataFrame(
            {"geometry": [box(minx, miny, maxx, maxy)]},
            crs=tiles_gdf.crs
        )
        europe = europe_box

    print(f"  Countries: {sorted(tiles_gdf['country'].unique())}")
    print(f"  CRS: {tiles_gdf.crs}")

    return tiles_gdf, europe


def create_simple_map(tiles_gdf, europe, output_dir):
    """
    Create a simple, clean map with all tiles as red points.

    Best for showing overall geographic distribution without clutter.
    """
    print("\nCreating simple map (red points)...")

    # Convert polygons to centroids for cleaner visualization
    tiles_centroids = tiles_gdf.copy()
    tiles_centroids['geometry'] = tiles_centroids.centroid

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    # Plot Europe with light fill for land/ocean distinction
    europe.plot(
        ax=ax,
        facecolor="0.95",      # Light gray fill
        edgecolor="0.4",       # Darker gray borders
        linewidth=0.6,
        zorder=1
    )

    # Plot tiles as points
    tiles_centroids.plot(
        ax=ax,
        markersize=8,
        color="tab:red",
        edgecolor="white",
        linewidth=0.3,
        alpha=0.8,
        zorder=2
    )

    # Set Europe extent (extended to include Turkey and all tiles)
    ax.set_xlim(-12, 45)  # Extended east to include Turkey
    ax.set_ylim(34, 72)

    # Add grid and labels
    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)
    ax.set_xticks(range(-10, 45, 10))
    ax.set_yticks(range(35, 75, 10))
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5, zorder=0)


    plt.tight_layout()

    # Save outputs
    pdf_path = output_dir / "study_area_map_simple.pdf"
    png_path = output_dir / "study_area_map_simple.png"

    plt.savefig(pdf_path, bbox_inches='tight', dpi=300)
    plt.savefig(png_path, bbox_inches='tight', dpi=300)

    print(f"  Saved: {pdf_path}")
    print(f"  Saved: {png_path}")

    plt.close(fig)


def create_colored_map(tiles_gdf, europe, output_dir):
    """
    Create a map with tiles colored by change type (nature loss vs cropland loss).

    Shows the distribution of different land-take categories.
    """
    print("\nCreating colored map (by change category)...")

    # Convert polygons to centroids
    tiles_centroids = tiles_gdf.copy()
    tiles_centroids['geometry'] = tiles_centroids.centroid

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    # Plot Europe with light fill for land/ocean distinction
    europe.plot(
        ax=ax,
        facecolor="0.95",      # Light gray fill
        edgecolor="0.4",       # Darker gray borders
        linewidth=0.6,
        zorder=1
    )

    # Define colors for change categories
    colors = {
        'nature loss': '#2ca02c',     # Green (ironic but clear)
        'cropland loss': '#ff7f0e'    # Orange
    }

    # Plot each category separately for legend control
    for category, color in colors.items():
        subset = tiles_centroids[tiles_centroids['r'] == category]
        subset.plot(
            ax=ax,
            markersize=8,
            color=color,
            edgecolor="white",
            linewidth=0.3,
            alpha=0.8,
            label=category.title(),
            zorder=2
        )

    # Set Europe extent
    ax.set_xlim(-12, 40)
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

    # Add title
    n_tiles = len(tiles_gdf)
    n_countries = tiles_gdf['country'].nunique()
    ax.set_title(
        f"Land-Take Detection Study Areas by Category\n{n_tiles} tiles across {n_countries} countries",
        fontsize=13,
        pad=15
    )

    plt.tight_layout()

    # Save outputs
    pdf_path = output_dir / "study_area_map_colored.pdf"
    png_path = output_dir / "study_area_map_colored.png"

    plt.savefig(pdf_path, bbox_inches='tight', dpi=300)
    plt.savefig(png_path, bbox_inches='tight', dpi=300)

    print(f"  Saved: {pdf_path}")
    print(f"  Saved: {png_path}")

    plt.close(fig)


def create_clean_thesis_map(tiles_gdf, europe, output_dir):
    """
    Create a minimal, clean map optimized for thesis inclusion.

    No title, minimal styling, just the essential information.
    """
    print("\nCreating clean thesis map (minimal styling)...")

    # Convert polygons to centroids
    tiles_centroids = tiles_gdf.copy()
    tiles_centroids['geometry'] = tiles_centroids.centroid

    # Create figure with specific size for thesis column
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

    # Plot Europe with light fill for land/ocean distinction
    europe.plot(
        ax=ax,
        facecolor="0.95",      # Light gray fill
        edgecolor="0.5",       # Medium gray borders
        linewidth=0.5,
        zorder=1
    )

    # Plot tiles as points (dark red, small)
    tiles_centroids.plot(
        ax=ax,
        markersize=4,
        color="#d62728",
        edgecolor="white",
        linewidth=0.2,
        alpha=0.85,
        zorder=2
    )

    # Set Europe extent
    ax.set_xlim(-12, 40)
    ax.set_ylim(34, 72)

    # Minimal axis styling
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(True)

    # Just a thin border
    for spine in ax.spines.values():
        spine.set_edgecolor('0.5')
        spine.set_linewidth(0.5)

    plt.tight_layout(pad=0.1)

    # Save outputs - this is the main thesis figure
    pdf_path = output_dir / "study_area_map.pdf"
    png_path = output_dir / "study_area_map.png"

    plt.savefig(pdf_path, bbox_inches='tight', dpi=300, pad_inches=0.05)
    plt.savefig(png_path, bbox_inches='tight', dpi=300, pad_inches=0.05)

    print(f"  Saved: {pdf_path} (MAIN THESIS FIGURE)")
    print(f"  Saved: {png_path}")

    plt.close(fig)


def print_statistics(tiles_gdf):
    """Print useful statistics about the dataset."""
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)

    print(f"\nTotal tiles: {len(tiles_gdf)}")
    print(f"Countries: {tiles_gdf['country'].nunique()}")

    print("\nTiles per country:")
    country_counts = tiles_gdf['country'].value_counts()
    for country, count in country_counts.items():
        print(f"  {country}: {count}")

    print("\nChange categories:")
    category_counts = tiles_gdf['r'].value_counts()
    for category, count in category_counts.items():
        print(f"  {category}: {count}")

    print("\nChange types:")
    type_counts = tiles_gdf['change_type'].value_counts()
    for change_type, count in type_counts.head(10).items():
        print(f"  {change_type}: {count}")

    # Geographic extent
    bounds = tiles_gdf.total_bounds  # minx, miny, maxx, maxy
    print(f"\nGeographic extent:")
    print(f"  Longitude: {bounds[0]:.2f}° to {bounds[2]:.2f}°")
    print(f"  Latitude: {bounds[1]:.2f}° to {bounds[3]:.2f}°")


def main():
    """Main execution function."""
    print("="*60)
    print("CREATING EUROPE MAP WITH TILE LOCATIONS")
    print("="*60)

    # Setup output directory
    output_dir = Path("docs/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    tiles_gdf, europe = load_data()

    # Print statistics
    print_statistics(tiles_gdf)

    # Create all three map versions
    print("\n" + "="*60)
    print("GENERATING MAPS")
    print("="*60)

    create_clean_thesis_map(tiles_gdf, europe, output_dir)
    create_simple_map(tiles_gdf, europe, output_dir)
    create_colored_map(tiles_gdf, europe, output_dir)

    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)
    print("\nGenerated 3 versions:")
    print("  1. study_area_map.pdf - Clean version for thesis (RECOMMENDED)")
    print("  2. study_area_map_simple.pdf - Simple with title and stats")
    print("  3. study_area_map_colored.pdf - Colored by change category")
    print("\nUse the clean version in your thesis with a caption like:")
    print('  "Geographic distribution of land-take detection study areas')
    print(f'   across Europe. Each point represents one 650m × 650m tile with')
    print(f'   bi-temporal VHR imagery (2018-2025). Total: {len(tiles_gdf)} unique tiles')
    print(f'   across {tiles_gdf["country"].nunique()} countries."')
    print("\nNote: Some coastal tiles may appear over water due to centroid calculation.")


if __name__ == "__main__":
    main()
