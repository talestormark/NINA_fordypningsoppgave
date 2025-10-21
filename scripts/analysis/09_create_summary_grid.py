"""
Script to create a summary grid showing diverse tile examples using VHR Google imagery
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for SLURM

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import DATA_ROOT, FOLDERS, REPORTS_DIR, FIGURES_DIR


def load_and_prepare_images(data_root, refid):
    """
    Load VHR RGB images and mask for a single tile

    Returns:
        tuple: (rgb_2018, rgb_2025, mask_resampled)
    """
    data_root = Path(data_root)

    vhr_path = data_root / FOLDERS['vhr'] / f"{refid}_RGBY_Mosaic.tif"
    mask_path = data_root / FOLDERS['masks'] / f"{refid}_mask.tif"

    # Load VHR RGB (6 bands: 2018 RGB + 2025 RGB)
    with rasterio.open(vhr_path) as src:
        # 2018 RGB (bands 1, 2, 3 = R, G, B)
        rgb_2018 = src.read([1, 2, 3])
        rgb_2018 = np.transpose(rgb_2018, (1, 2, 0))
        rgb_2018 = rgb_2018 / 255.0

        # 2025 RGB (bands 4, 5, 6 = R, G, B)
        rgb_2025 = src.read([4, 5, 6])
        rgb_2025 = np.transpose(rgb_2025, (1, 2, 0))
        rgb_2025 = rgb_2025 / 255.0

        # Get VHR dimensions for mask resampling
        vhr_height, vhr_width = rgb_2018.shape[:2]

    # Load and resample mask to match VHR resolution
    with rasterio.open(mask_path) as src:
        mask = src.read(
            1,
            out_shape=(vhr_height, vhr_width),
            resampling=Resampling.nearest
        )

    return rgb_2018, rgb_2025, mask


if __name__ == "__main__":
    try:
        # Load edge case files
        reports_dir = Path(REPORTS_DIR)

        # Read category files
        categories = {}
        for category in ['zero_change', 'low_change', 'moderate_change', 'high_change']:
            file_path = reports_dir / f"refids_{category}.txt"
            if file_path.exists():
                with open(file_path, 'r') as f:
                    refids = [line.strip() for line in f if line.strip()]
                categories[category] = refids
            else:
                categories[category] = []

        # Load mask analysis to get change percentages
        mask_analysis = pd.read_csv(reports_dir / "mask_analysis.csv")

        print("\nSelecting diverse REFIDs for VHR summary grid...\n")
        print("=" * 80)

        # Select 6 diverse REFIDs
        selected_refids = []
        selected_labels = []

        # 2 from zero_change (if available)
        if len(categories['zero_change']) >= 2:
            selected_refids.extend(categories['zero_change'][:2])
            selected_labels.extend(['Zero Change'] * 2)
        elif len(categories['zero_change']) == 1:
            selected_refids.append(categories['zero_change'][0])
            selected_labels.append('Zero Change')

        # Fill remaining slots with moderate and high
        # 2 from moderate_change
        if len(categories['moderate_change']) >= 2:
            selected_refids.extend(categories['moderate_change'][:2])
            selected_labels.extend(['Moderate Change'] * 2)
        elif len(categories['moderate_change']) == 1:
            selected_refids.append(categories['moderate_change'][0])
            selected_labels.append('Moderate Change')

        # 2 from high_change
        if len(categories['high_change']) >= 2:
            selected_refids.extend(categories['high_change'][:2])
            selected_labels.extend(['High Change'] * 2)
        elif len(categories['high_change']) == 1:
            selected_refids.append(categories['high_change'][0])
            selected_labels.append('High Change')

        # If we don't have 6 yet, fill with low_change
        while len(selected_refids) < 6 and len(categories['low_change']) > 0:
            idx = len(selected_refids) - (2 if len(categories['zero_change']) >= 2 else len(categories['zero_change']))
            if idx < len(categories['low_change']):
                selected_refids.append(categories['low_change'][idx])
                selected_labels.append('Low Change')
            else:
                break

        print(f"Selected {len(selected_refids)} tiles:")
        for refid, label in zip(selected_refids, selected_labels):
            change_pct = mask_analysis[mask_analysis['refid'] == refid]['change_ratio'].values
            if len(change_pct) > 0:
                print(f"  - {refid[:40]}... [{label}: {change_pct[0]:.1f}%]")

        # Create summary grid
        print("\nCreating VHR summary grid visualization...")

        n_tiles = len(selected_refids)
        fig, axes = plt.subplots(n_tiles, 3, figsize=(15, 5 * n_tiles))

        # Handle case where we have only one tile
        if n_tiles == 1:
            axes = axes.reshape(1, -1)

        for i, (refid, label) in enumerate(zip(selected_refids, selected_labels)):
            print(f"  Loading tile {i+1}/{n_tiles}: {refid[:40]}...")

            # Get change percentage
            change_pct = mask_analysis[mask_analysis['refid'] == refid]['change_ratio'].values[0]

            # Load images
            rgb_2018, rgb_2025, mask = load_and_prepare_images(DATA_ROOT, refid)

            # Plot 2018 RGB
            axes[i, 0].imshow(rgb_2018)
            axes[i, 0].axis('off')
            if i == 0:
                axes[i, 0].set_title('2018 RGB (VHR - 1m)', fontsize=13, fontweight='bold')

            # Plot 2025 RGB
            axes[i, 1].imshow(rgb_2025)
            axes[i, 1].axis('off')
            if i == 0:
                axes[i, 1].set_title('2025 RGB (VHR - 1m)', fontsize=13, fontweight='bold')

            # Plot 2025 RGB + Mask overlay
            axes[i, 2].imshow(rgb_2025)
            masked_overlay = np.ma.masked_where(mask == 0, mask)
            axes[i, 2].imshow(masked_overlay, cmap='Reds', alpha=0.6, vmin=0, vmax=1)
            axes[i, 2].axis('off')
            if i == 0:
                axes[i, 2].set_title('2025 + Land-Take Mask', fontsize=13, fontweight='bold')

            # Add row label on the left
            fig.text(0.02, 1 - (i + 0.5) / n_tiles, f'{label}\n{change_pct:.1f}% change',
                    fontsize=11, ha='left', va='center', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        fig.suptitle('Land-Take Detection Dataset: High-Resolution Sample Tiles (VHR Google)',
                     fontsize=17, fontweight='bold')
        plt.tight_layout(rect=[0.08, 0, 1, 0.97])

        # Save
        output_path = Path(FIGURES_DIR) / "summary_grid.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\n✓ VHR summary grid saved to: {output_path}")

        print("\n" + "=" * 80)
        print("\n✅ VHR summary grid creation complete!")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
