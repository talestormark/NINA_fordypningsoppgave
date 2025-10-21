"""
Script to visualize individual tiles with VHR Google RGB composites and mask overlays
"""

import sys
from pathlib import Path
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for SLURM

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import DATA_ROOT, FOLDERS, REPORTS_DIR, FIGURES_DIR


def visualize_tile(data_root, refid, save_path):
    """
    Create 2x3 subplot visualization for a single tile using VHR Google imagery

    Args:
        data_root: Root data directory
        refid: REFID string
        save_path: Path to save figure
    """
    data_root = Path(data_root)

    # Construct file paths
    vhr_path = data_root / FOLDERS['vhr'] / f"{refid}_RGBY_Mosaic.tif"
    mask_path = data_root / FOLDERS['masks'] / f"{refid}_mask.tif"

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Load VHR data (6 bands: 2018 RGB + 2025 RGB)
    with rasterio.open(vhr_path) as src:
        # Band order: [2018_R, 2018_G, 2018_B, 2025_R, 2025_G, 2025_B]
        # Read RGB in correct order for matplotlib (RGB not BGR)

        # 2018 RGB (bands 1, 2, 3 = R, G, B)
        rgb_2018 = src.read([1, 2, 3])  # R, G, B
        rgb_2018 = np.transpose(rgb_2018, (1, 2, 0))  # (C, H, W) -> (H, W, C)
        rgb_2018 = rgb_2018 / 255.0  # Normalize uint8 to 0-1

        # 2025 RGB (bands 4, 5, 6 = R, G, B)
        rgb_2025 = src.read([4, 5, 6])  # R, G, B
        rgb_2025 = np.transpose(rgb_2025, (1, 2, 0))
        rgb_2025 = rgb_2025 / 255.0

    # Load mask (need to resample to match VHR resolution)
    with rasterio.open(mask_path) as src:
        # Mask is at 10m resolution, VHR is at 1m resolution
        # We need to resample mask to match VHR dimensions
        from rasterio.enums import Resampling

        # Read mask with resampling to match VHR dimensions
        mask = src.read(
            1,
            out_shape=(rgb_2018.shape[0], rgb_2018.shape[1]),
            resampling=Resampling.nearest
        )

    # Row 1 - RGB composites
    axes[0, 0].imshow(rgb_2018)
    axes[0, 0].set_title('2018 RGB (VHR Google - 1m)', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(rgb_2025)
    axes[0, 1].set_title('2025 RGB (VHR Google - 1m)', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')

    # RGB difference visualization
    diff_rgb = rgb_2025 - rgb_2018
    # Enhance difference for better visualization
    diff_enhanced = np.clip(diff_rgb * 2 + 0.5, 0, 1)
    axes[0, 2].imshow(diff_enhanced)
    axes[0, 2].set_title('RGB Change (Enhanced)', fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')

    # Row 2 - Masks & change
    # Mask alone
    axes[1, 0].imshow(mask, cmap='RdYlGn_r', vmin=0, vmax=1)
    axes[1, 0].set_title('Land-Take Mask (Resampled to 1m)', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')

    # 2025 RGB with mask overlay
    axes[1, 1].imshow(rgb_2025)
    masked_overlay = np.ma.masked_where(mask == 0, mask)
    axes[1, 1].imshow(masked_overlay, cmap='Reds', alpha=0.5, vmin=0, vmax=1)
    axes[1, 1].set_title('2025 RGB + Mask Overlay', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')

    # Grayscale difference
    gray_2018 = np.mean(rgb_2018, axis=2)
    gray_2025 = np.mean(rgb_2025, axis=2)
    diff = gray_2025 - gray_2018
    im = axes[1, 2].imshow(diff, cmap='RdBu', vmin=-0.3, vmax=0.3)
    axes[1, 2].set_title('Grayscale Difference (2025 - 2018)', fontsize=14, fontweight='bold')
    axes[1, 2].axis('off')
    plt.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04)

    # Set overall title
    refid_short = refid[:60] + "..." if len(refid) > 60 else refid
    fig.suptitle(f'REFID: {refid_short}',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout()

    # Save figure
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    try:
        # Load REFID list
        refid_list_file = Path(REPORTS_DIR) / "refid_list.txt"
        with open(refid_list_file, 'r') as f:
            refids = [line.strip() for line in f if line.strip()]

        print(f"\nLoaded {len(refids)} REFIDs")
        print(f"Creating VHR visualizations for first 3 REFIDs...\n")
        print("=" * 80)

        # Visualize first 3 REFIDs
        figures_dir = Path(FIGURES_DIR)
        figures_dir.mkdir(parents=True, exist_ok=True)

        for i, refid in enumerate(refids[:3], 1):
            print(f"\n[{i}/3] Processing: {refid}")

            output_path = figures_dir / f"tile_viz_{refid}.png"

            try:
                visualize_tile(DATA_ROOT, refid, output_path)
                print(f"  ✓ Saved to: {output_path}")
            except Exception as e:
                print(f"  ✗ Error: {e}")
                import traceback
                traceback.print_exc()

        print("\n" + "=" * 80)
        print("\n✅ VHR tile visualizations complete!")
        print(f"   Created 3 high-resolution visualizations in: {figures_dir}")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
