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
        dict: Analysis results
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
        if num_patches > 0:
            patch_sizes = []
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
            'success': False,
            'error': str(e)
        }


if __name__ == "__main__":
    try:
        # Load REFID list
        refid_list_file = Path(REPORTS_DIR) / "refid_list.txt"
        with open(refid_list_file, 'r') as f:
            refids = [line.strip() for line in f if line.strip()]

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

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Subplot 1: Histogram of change_ratio
        axes[0].hist(df_success['change_ratio'], bins=30, color='steelblue', edgecolor='black')
        axes[0].set_xlabel('Change Ratio (%)', fontsize=12)
        axes[0].set_ylabel('Number of Tiles', fontsize=12)
        axes[0].set_title('Distribution of Change Ratio', fontsize=14, fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)

        # Subplot 2: Histogram of num_change_patches
        axes[1].hist(df_success['num_change_patches'], bins=30, color='coral', edgecolor='black')
        axes[1].set_xlabel('Number of Change Patches', fontsize=12)
        axes[1].set_ylabel('Number of Tiles', fontsize=12)
        axes[1].set_title('Distribution of Change Patches', fontsize=14, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)

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
        print(f"   Overall change: {overall_change_ratio:.2f}%")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
