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
    Create 2x2 subplot visualization for a single tile using VHR Google imagery

    Layout:
    [0,0]: VHR 2018 RGB
    [0,1]: VHR 2025 RGB
    [1,0]: Binary mask (land-take areas)
    [1,1]: VHR 2025 with mask overlay

    Args:
        data_root: Root data directory
        refid: REFID string
        save_path: Path to save figure
    """
    data_root = Path(data_root)

    # Construct file paths
    vhr_path = data_root / FOLDERS['vhr'] / f"{refid}_RGBY_Mosaic.tif"
    mask_path = data_root / FOLDERS['masks'] / f"{refid}_mask.tif"

    # Create figure with 2x2 layout
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))

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

    # [0,0]: VHR 2018 RGB
    axes[0, 0].imshow(rgb_2018)
    axes[0, 0].set_title('VHR 2018 RGB (1m resolution)', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    # [0,1]: VHR 2025 RGB
    axes[0, 1].imshow(rgb_2025)
    axes[0, 1].set_title('VHR 2025 RGB (1m resolution)', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')

    # [1,0]: Binary mask (land-take areas)
    axes[1, 0].imshow(mask, cmap='RdYlGn_r', vmin=0, vmax=1)
    axes[1, 0].set_title('Land-Take Mask (Binary)', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')

    # [1,1]: VHR 2025 with mask overlay
    axes[1, 1].imshow(rgb_2025)
    masked_overlay = np.ma.masked_where(mask == 0, mask)
    axes[1, 1].imshow(masked_overlay, cmap='Reds', alpha=0.5, vmin=0, vmax=1)
    axes[1, 1].set_title('VHR 2025 + Mask Overlay', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')

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
        # Load REFID list (enhanced format with header and metadata table)
        refid_list_file = Path(REPORTS_DIR) / "refid_list.txt"
        refids = []
        with open(refid_list_file, 'r') as f:
            lines = f.readlines()
            # Skip header lines, find the metadata table section
            in_table = False
            for line in lines:
                line = line.strip()
                # Start reading after the column headers line
                if line.startswith('REFID') and 'Country' in line:
                    in_table = True
                    continue
                # Skip separator lines
                if line.startswith('-') or line.startswith('='):
                    continue
                # Parse REFID lines (they start with 'a')
                if in_table and line and line.startswith('a'):
                    # Extract just the REFID (first column)
                    refid = line.split()[0]
                    refids.append(refid)

        print(f"\nLoaded {len(refids)} REFIDs")
        print(f"Creating streamlined 2×2 VHR visualizations for first 3 REFIDs...\n")
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
        print("\n✅ Streamlined VHR tile visualizations complete!")
        print(f"   Created 3 visualizations (2×2 layout) in: {figures_dir}")
        print(f"\n   Layout: [VHR 2018] [VHR 2025]")
        print(f"           [Mask]     [Overlay]")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
