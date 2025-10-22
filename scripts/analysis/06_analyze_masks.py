"""
Script to comprehensively analyze masks and compute class balance statistics
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import rasterio
from scipy import ndimage
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import DATA_ROOT, FOLDERS, REPORTS_DIR, FIGURES_DIR


def analyze_mask(mask_path):
    """
    Analyze a single mask file

    Args:
        mask_path: Path to mask file

    Returns:
        dict: Analysis results including patch_sizes list
    """
    try:
        with rasterio.open(mask_path) as src:
            mask = src.read(1)

        # Basic statistics
        total_pixels = mask.size
        change_pixels = np.sum(mask == 1)
        no_change_pixels = np.sum(mask == 0)
        change_ratio = (change_pixels / total_pixels) * 100

        # Find connected change regions using scipy.ndimage.label
        labeled_mask, num_patches = ndimage.label(mask == 1)

        # Get patch sizes
        patch_sizes = []
        if num_patches > 0:
            for patch_id in range(1, num_patches + 1):
                patch_size = np.sum(labeled_mask == patch_id)
                patch_sizes.append(patch_size)

            max_patch_size = max(patch_sizes)
            mean_patch_size = np.mean(patch_sizes)
        else:
            max_patch_size = 0
            mean_patch_size = 0

        return {
            'total_pixels': total_pixels,
            'change_pixels': change_pixels,
            'no_change_pixels': no_change_pixels,
            'change_ratio': change_ratio,
            'num_change_patches': num_patches,
            'max_patch_size': max_patch_size,
            'mean_patch_size': mean_patch_size,
            'patch_sizes': patch_sizes,  # NEW: return actual patch sizes list
            'success': True
        }

    except Exception as e:
        return {
            'total_pixels': None,
            'change_pixels': None,
            'no_change_pixels': None,
            'change_ratio': None,
            'num_change_patches': None,
            'max_patch_size': None,
            'mean_patch_size': None,
            'patch_sizes': [],  # NEW: empty list on error
            'success': False,
            'error': str(e)
        }


def load_refids(refid_file):
    """
    Load REFIDs from enhanced refid_list.txt format (with metadata table)

    Args:
        refid_file: Path to refid_list.txt

    Returns:
        list: REFIDs
    """
    refids = []
    in_data_section = False

    with open(refid_file, 'r') as f:
        for line in f:
            line_stripped = line.strip()

            # Skip empty lines
            if not line_stripped:
                continue

            # Skip header lines
            if line_stripped.startswith('=') or line_stripped.startswith('#') or \
               'Land-Take Detection Dataset' in line_stripped or \
               'European countries' in line_stripped or \
               'Generated:' in line_stripped or \
               'Total REFIDs:' in line_stripped or \
               'These REFIDs have complete data' in line_stripped or \
               'REFID List with Metadata' in line_stripped:
                continue

            # Detect start of data section (header row with "REFID Country Loss Type...")
            if line_stripped.startswith('REFID') and 'Country' in line_stripped:
                in_data_section = True
                continue

            # Skip separator lines
            if line_stripped.startswith('-'):
                continue

            # Parse data lines: REFID is first whitespace-separated token
            if in_data_section:
                # Split on whitespace and take first token
                parts = line_stripped.split()
                if parts:
                    refid = parts[0]
                    # Validate REFID format: starts with 'a', 'R', or 'r' and contains '_'
                    if refid and len(refid) > 10 and refid[0] in 'aRr' and '_' in refid:
                        refids.append(refid)

    return refids


if __name__ == "__main__":
    try:
        # Load REFID list using proper parser
        refid_list_file = Path(REPORTS_DIR) / "refid_list.txt"
        refids = load_refids(refid_list_file)

        print(f"\nLoaded {len(refids)} REFIDs")
        print(f"Analyzing ALL {len(refids)} masks...\n")
        print("=" * 80)

        # Process ALL REFIDs
        results = []
        data_root = Path(DATA_ROOT)

        for refid in tqdm(refids, desc="Analyzing masks"):
            mask_path = data_root / FOLDERS['masks'] / f"{refid}_mask.tif"

            row = {'refid': refid}

            if mask_path.exists():
                analysis = analyze_mask(mask_path)
                row.update(analysis)
            else:
                row.update({
                    'total_pixels': None,
                    'change_pixels': None,
                    'no_change_pixels': None,
                    'change_ratio': None,
                    'num_change_patches': None,
                    'max_patch_size': None,
                    'mean_patch_size': None,
                    'success': False,
                    'error': 'File not found'
                })

            results.append(row)

        # Create DataFrame
        df = pd.DataFrame(results)

        # Filter successful analyses
        df_success = df[df['success'] == True].copy()

        # Compute global statistics
        print("\n" + "=" * 80)
        print("\n📊 Global Statistics:\n")

        if len(df_success) > 0:
            # Overall change ratio (all tiles combined)
            total_all_pixels = df_success['total_pixels'].sum()
            total_all_change = df_success['change_pixels'].sum()
            overall_change_ratio = (total_all_change / total_all_pixels) * 100

            print(f"Overall change ratio (all tiles combined): {overall_change_ratio:.2f}%")
            print(f"Median change ratio per tile: {df_success['change_ratio'].median():.2f}%")
            print(f"Mean change ratio per tile: {df_success['change_ratio'].mean():.2f}%")

            # Count tiles with specific change levels
            zero_change = (df_success['change_ratio'] == 0).sum()
            high_change = (df_success['change_ratio'] > 50).sum()

            print(f"\nTiles with 0% change: {zero_change}/{len(df_success)}")
            print(f"Tiles with >50% change: {high_change}/{len(df_success)}")

            print(f"\nPatch statistics:")
            print(f"  Average patches per tile: {df_success['num_change_patches'].mean():.1f}")
            print(f"  Average mean patch size: {df_success['mean_patch_size'].mean():.1f} pixels")
            print(f"  Largest single patch: {df_success['max_patch_size'].max():.0f} pixels")

        # Save to CSV
        output_file = Path(REPORTS_DIR) / "mask_analysis.csv"
        df.to_csv(output_file, index=False)
        print(f"\n✓ Mask analysis saved to: {output_file}")

        # Create visualization
        print("\nCreating visualizations...")

        # Collect all patch sizes from all tiles
        all_patch_sizes = []
        for idx, row in df_success.iterrows():
            if 'patch_sizes' in row and row['patch_sizes']:
                all_patch_sizes.extend(row['patch_sizes'])

        print(f"Total patches across all tiles: {len(all_patch_sizes)}")

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Subplot 1: Histogram of change_ratio
        axes[0].hist(df_success['change_ratio'], bins=30, color='steelblue', edgecolor='black')
        axes[0].set_xlabel('Change Ratio (%)', fontsize=12)
        axes[0].set_ylabel('Number of Tiles', fontsize=12)
        axes[0].set_title('Distribution of Change Ratio', fontsize=14, fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)

        # Subplot 2: ENHANCED - Histogram of patch sizes (log scale)
        if len(all_patch_sizes) > 0:
            # Use log scale for better visualization of wide range
            axes[1].hist(all_patch_sizes, bins=50, color='coral', edgecolor='black')
            axes[1].set_xlabel('Patch Size (pixels)', fontsize=12)
            axes[1].set_ylabel('Number of Patches', fontsize=12)
            axes[1].set_title('Distribution of Patch Sizes', fontsize=14, fontweight='bold')
            axes[1].set_yscale('log')  # Log scale for y-axis
            axes[1].grid(axis='y', alpha=0.3, which='both')

            # Add statistics text
            median_size = np.median(all_patch_sizes)
            mean_size = np.mean(all_patch_sizes)
            axes[1].axvline(median_size, color='red', linestyle='--', linewidth=2,
                           label=f'Median: {median_size:.0f}px')
            axes[1].legend(fontsize=10)
        else:
            axes[1].text(0.5, 0.5, 'No patch data available',
                        ha='center', va='center', transform=axes[1].transAxes)

        # Subplot 3: Boxplot of change_ratio
        axes[2].boxplot(df_success['change_ratio'], vert=True, patch_artist=True,
                       boxprops=dict(facecolor='lightgreen', edgecolor='black'),
                       medianprops=dict(color='red', linewidth=2),
                       whiskerprops=dict(color='black'),
                       capprops=dict(color='black'))
        axes[2].set_ylabel('Change Ratio (%)', fontsize=12)
        axes[2].set_title('Change Ratio Distribution', fontsize=14, fontweight='bold')
        axes[2].grid(axis='y', alpha=0.3)
        axes[2].set_xticklabels(['All Tiles'])

        plt.tight_layout()

        # Save figure
        fig_output = Path(FIGURES_DIR) / "mask_statistics.png"
        fig_output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_output, dpi=150, bbox_inches='tight')
        print(f"✓ Visualization saved to: {fig_output}")

        plt.close()

        # Summary
        print("\n" + "=" * 80)
        print(f"\n✅ Mask analysis complete!")
        print(f"   Processed: {len(df_success)}/{len(df)} tiles")
        if len(df_success) > 0:
            print(f"   Overall change: {overall_change_ratio:.2f}%")
        else:
            print(f"   No successful analyses to report")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
