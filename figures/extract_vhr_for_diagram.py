"""
Extract 2018 and 2025 RGB images from VHR tile for LaTeX diagram
"""
import rasterio
import numpy as np
from PIL import Image
import sys
import os

def extract_vhr_images(vhr_path, output_dir, refid):
    """
    Extract 2018 and 2025 RGB images from VHR 6-band GeoTIFF

    Bands:
    0-2: 2018 RGB
    3-5: 2025 RGB
    """
    print(f"Reading VHR tile: {vhr_path}")

    with rasterio.open(vhr_path) as src:
        # Read all bands
        data = src.read()  # Shape: (6, H, W)

        print(f"VHR data shape: {data.shape}")
        print(f"VHR dtype: {data.dtype}")
        print(f"VHR value range: {data.min()} - {data.max()}")

        # Extract 2018 RGB (bands 0-2)
        img_2018 = data[0:3, :, :]  # Shape: (3, H, W)
        img_2018 = np.transpose(img_2018, (1, 2, 0))  # Shape: (H, W, 3)

        # Extract 2025 RGB (bands 3-5)
        img_2025 = data[3:6, :, :]  # Shape: (3, H, W)
        img_2025 = np.transpose(img_2025, (1, 2, 0))  # Shape: (H, W, 3)

        # Ensure uint8 range
        if img_2018.dtype != np.uint8:
            img_2018 = np.clip(img_2018, 0, 255).astype(np.uint8)
            img_2025 = np.clip(img_2025, 0, 255).astype(np.uint8)

        # Create small thumbnails for diagram (300x300 pixels)
        h, w = img_2018.shape[:2]
        target_size = 300

        # Calculate center crop
        if h > w:
            crop_size = w
            y_start = (h - crop_size) // 2
            x_start = 0
        else:
            crop_size = h
            y_start = 0
            x_start = (w - crop_size) // 2

        # Crop center square
        img_2018_crop = img_2018[y_start:y_start+crop_size, x_start:x_start+crop_size]
        img_2025_crop = img_2025[y_start:y_start+crop_size, x_start:x_start+crop_size]

        # Convert to PIL and resize
        pil_2018 = Image.fromarray(img_2018_crop)
        pil_2025 = Image.fromarray(img_2025_crop)

        pil_2018_thumb = pil_2018.resize((target_size, target_size), Image.Resampling.LANCZOS)
        pil_2025_thumb = pil_2025.resize((target_size, target_size), Image.Resampling.LANCZOS)

        # Save images
        os.makedirs(output_dir, exist_ok=True)

        path_2018 = os.path.join(output_dir, f"{refid}_2018.jpg")
        path_2025 = os.path.join(output_dir, f"{refid}_2025.jpg")

        pil_2018_thumb.save(path_2018, quality=95)
        pil_2025_thumb.save(path_2025, quality=95)

        print(f"Saved 2018 image: {path_2018}")
        print(f"Saved 2025 image: {path_2025}")

        return path_2018, path_2025


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python extract_vhr_for_diagram.py <refid>")
        sys.exit(1)

    refid = sys.argv[1]

    # Paths
    vhr_dir = "data/raw/VHR_google"
    vhr_path = os.path.join(vhr_dir, f"{refid}_RGBY_Mosaic.tif")
    output_dir = "docs/figures/input_images"

    if not os.path.exists(vhr_path):
        print(f"ERROR: VHR file not found: {vhr_path}")
        sys.exit(1)

    extract_vhr_images(vhr_path, output_dir, refid)
    print("\nDone! Images ready for LaTeX diagram.")
