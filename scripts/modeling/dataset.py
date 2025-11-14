#!/usr/bin/env python3
"""
PyTorch Dataset for land-take detection using bi-temporal VHR imagery.

The dataset handles:
- 6-channel GeoTIFF files (bands 0-2: 2018 RGB, bands 3-5: 2025 RGB)
- Binary mask labels (0: no change, 1: land-take/change)
- Data augmentation using albumentations
- Train/val/test splits
"""

import torch
from torch.utils.data import Dataset
import rasterio
import numpy as np
from pathlib import Path
from typing import Optional, Callable, List, Tuple
import albumentations as A
from albumentations.pytorch import ToTensorV2


class LandTakeDataset(Dataset):
    """
    Dataset for bi-temporal land-take detection.

    Args:
        refids: List of reference IDs for tiles to include
        vhr_dir: Directory containing VHR GeoTIFF files
        mask_dir: Directory containing mask GeoTIFF files
        transform: Albumentations transform to apply
        return_separate_images: If True, returns (img_2018, img_2025, mask)
                                If False, returns (img_concat, mask) where img_concat has 6 channels
    """

    def __init__(
        self,
        refids: List[str],
        vhr_dir: str = "data/raw/VHR_google",
        mask_dir: str = "data/raw/Land_take_masks",
        transform: Optional[Callable] = None,
        return_separate_images: bool = False,
    ):
        self.refids = refids
        self.vhr_dir = Path(vhr_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.return_separate_images = return_separate_images

        # Verify files exist
        self._verify_files()

    def _verify_files(self):
        """Verify that VHR and mask files exist for all refids."""
        missing_vhr = []
        missing_mask = []

        for refid in self.refids:
            vhr_path = self.vhr_dir / f"{refid}_RGBY_Mosaic.tif"
            mask_path = self.mask_dir / f"{refid}_mask.tif"

            if not vhr_path.exists():
                missing_vhr.append(refid)
            if not mask_path.exists():
                missing_mask.append(refid)

        if missing_vhr:
            raise FileNotFoundError(
                f"Missing VHR files for {len(missing_vhr)} refids: {missing_vhr[:5]}..."
            )
        if missing_mask:
            raise FileNotFoundError(
                f"Missing mask files for {len(missing_mask)} refids: {missing_mask[:5]}..."
            )

    def __len__(self) -> int:
        return len(self.refids)

    def _load_vhr(self, refid: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load VHR imagery and split into 2018 and 2025 images.

        Args:
            refid: Reference ID

        Returns:
            img_2018: (H, W, 3) array with 2018 RGB
            img_2025: (H, W, 3) array with 2025 RGB
        """
        vhr_path = self.vhr_dir / f"{refid}_RGBY_Mosaic.tif"

        with rasterio.open(vhr_path) as src:
            # Read all 6 bands: [2018_R, 2018_G, 2018_B, 2025_R, 2025_G, 2025_B]
            data = src.read()  # Shape: (6, H, W)

        # Split into 2018 and 2025
        img_2018 = data[0:3, :, :].transpose(1, 2, 0)  # (H, W, 3)
        img_2025 = data[3:6, :, :].transpose(1, 2, 0)  # (H, W, 3)

        # Convert to float32 and normalize to [0, 1]
        img_2018 = img_2018.astype(np.float32) / 255.0
        img_2025 = img_2025.astype(np.float32) / 255.0

        return img_2018, img_2025

    def _load_mask(self, refid: str, target_shape: tuple = None) -> np.ndarray:
        """
        Load binary mask and optionally resample to match VHR resolution.

        Args:
            refid: Reference ID
            target_shape: (H, W) to resample mask to VHR resolution

        Returns:
            mask: (H, W) array with binary labels (0 or 1)
        """
        from scipy.ndimage import zoom

        mask_path = self.mask_dir / f"{refid}_mask.tif"

        with rasterio.open(mask_path) as src:
            mask = src.read(1)  # Read first band, shape: (H, W)

        # Ensure binary (0 or 1)
        mask = (mask > 0).astype(np.float32)

        # Resample mask to match VHR resolution if target shape provided
        if target_shape is not None:
            zoom_factors = (target_shape[0] / mask.shape[0], target_shape[1] / mask.shape[1])
            mask = zoom(mask, zoom_factors, order=0)  # order=0 for nearest neighbor (preserve binary)

        return mask

    def __getitem__(self, idx: int) -> dict:
        """
        Get a single sample.

        Returns:
            dict with keys:
                - 'image': Either (C=6, H, W) concatenated or (C=3, H, W) per image
                - 'image_2018': (C=3, H, W) if return_separate_images=True
                - 'image_2025': (C=3, H, W) if return_separate_images=True
                - 'mask': (H, W) binary mask
                - 'refid': Reference ID string
        """
        refid = self.refids[idx]

        # Load data
        img_2018, img_2025 = self._load_vhr(refid)
        # Resample mask to match VHR resolution (1m vs 10m)
        target_shape = img_2018.shape[:2]  # (H, W)
        mask = self._load_mask(refid, target_shape=target_shape)

        # Apply transformations
        if self.transform:
            # Albumentations requires a single image input
            # For bi-temporal, we concatenate along channel dimension
            img_concat = np.concatenate([img_2018, img_2025], axis=2)  # (H, W, 6)

            transformed = self.transform(image=img_concat, mask=mask)
            img_concat = transformed['image']
            mask = transformed['mask']

            # If image is a tensor, split back into separate images if needed
            if isinstance(img_concat, torch.Tensor):
                # img_concat is (6, H, W) after ToTensorV2
                if self.return_separate_images:
                    img_2018 = img_concat[0:3, :, :]  # (3, H, W)
                    img_2025 = img_concat[3:6, :, :]  # (3, H, W)
        else:
            # No transform, manually convert to tensors
            img_concat = np.concatenate([img_2018, img_2025], axis=2)  # (H, W, 6)
            img_concat = torch.from_numpy(img_concat.transpose(2, 0, 1))  # (6, H, W)
            mask = torch.from_numpy(mask)  # (H, W)

            if self.return_separate_images:
                img_2018 = img_concat[0:3, :, :]
                img_2025 = img_concat[3:6, :, :]

        # Build output dictionary
        output = {
            'mask': mask,
            'refid': refid,
        }

        if self.return_separate_images:
            output['image_2018'] = img_2018
            output['image_2025'] = img_2025
        else:
            output['image'] = img_concat

        return output


def get_training_augmentation(image_size: int = 512) -> A.Compose:
    """
    Get augmentation pipeline for training.

    Args:
        image_size: Target image size (assumes square images)

    Returns:
        Albumentations Compose object
    """
    return A.Compose([
        # Resize to fixed size first (required for batching)
        A.Resize(height=image_size, width=image_size, interpolation=1, p=1.0),

        # Geometric transforms
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=45,
            border_mode=0,
            p=0.5
        ),

        # Color transforms (applied to all 6 channels)
        # Note: HueSaturationValue only works with 3-channel images, so excluded
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
            A.RandomGamma(gamma_limit=(80, 120), p=1),
        ], p=0.5),

        # Noise and blur
        A.OneOf([
            A.GaussNoise(p=1),
            A.GaussianBlur(blur_limit=(3, 7), p=1),
        ], p=0.3),

        # Normalize and convert to tensor
        ToTensorV2(),
    ])


def get_validation_augmentation(image_size: int = 512) -> A.Compose:
    """
    Get augmentation pipeline for validation/test (no augmentation, just resize and normalization).

    Args:
        image_size: Target image size (assumes square images)

    Returns:
        Albumentations Compose object
    """
    return A.Compose([
        # Resize to fixed size (required for batching)
        A.Resize(height=image_size, width=image_size, interpolation=1, p=1.0),
        ToTensorV2(),
    ])


def load_refids_from_file(filepath: str) -> List[str]:
    """
    Load reference IDs from a text file (one per line).

    Args:
        filepath: Path to text file with refids

    Returns:
        List of reference ID strings
    """
    with open(filepath, 'r') as f:
        refids = [line.strip() for line in f if line.strip()]
    return refids


def create_dataloaders(
    batch_size: int = 8,
    num_workers: int = 4,
    image_size: int = 512,
    return_separate_images: bool = False,
    train_refids_path: str = "outputs/splits/train_refids.txt",
    val_refids_path: str = "outputs/splits/val_refids.txt",
    test_refids_path: str = "outputs/splits/test_refids.txt",
) -> dict:
    """
    Create train, validation, and test dataloaders.

    Args:
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        image_size: Target image size
        return_separate_images: Whether to return separate 2018/2025 images
        train_refids_path: Path to train refids file
        val_refids_path: Path to validation refids file
        test_refids_path: Path to test refids file

    Returns:
        Dictionary with keys 'train', 'val', 'test' containing DataLoader objects
    """
    from torch.utils.data import DataLoader

    # Load refids
    train_refids = load_refids_from_file(train_refids_path)
    val_refids = load_refids_from_file(val_refids_path)
    test_refids = load_refids_from_file(test_refids_path)

    # Create datasets
    train_dataset = LandTakeDataset(
        refids=train_refids,
        transform=get_training_augmentation(image_size),
        return_separate_images=return_separate_images,
    )

    val_dataset = LandTakeDataset(
        refids=val_refids,
        transform=get_validation_augmentation(image_size),
        return_separate_images=return_separate_images,
    )

    test_dataset = LandTakeDataset(
        refids=test_refids,
        transform=get_validation_augmentation(image_size),
        return_separate_images=return_separate_images,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
    }


if __name__ == "__main__":
    """Test the dataset implementation."""
    print("Testing LandTakeDataset...")

    # Load a few refids
    train_refids = load_refids_from_file("outputs/splits/train_refids.txt")[:5]

    # Create dataset with augmentation
    dataset = LandTakeDataset(
        refids=train_refids,
        transform=get_training_augmentation(),
        return_separate_images=False,
    )

    print(f"Dataset size: {len(dataset)}")

    # Test loading a sample
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"Image shape: {sample['image'].shape}")
    print(f"Mask shape: {sample['mask'].shape}")
    print(f"Refid: {sample['refid']}")
    print(f"Image dtype: {sample['image'].dtype}")
    print(f"Mask dtype: {sample['mask'].dtype}")
    print(f"Image range: [{sample['image'].min():.3f}, {sample['image'].max():.3f}]")
    print(f"Mask unique values: {torch.unique(sample['mask'])}")

    # Test with separate images
    dataset_sep = LandTakeDataset(
        refids=train_refids,
        transform=get_training_augmentation(),
        return_separate_images=True,
    )

    sample_sep = dataset_sep[0]
    print(f"\nWith separate images:")
    print(f"Sample keys: {sample_sep.keys()}")
    print(f"Image 2018 shape: {sample_sep['image_2018'].shape}")
    print(f"Image 2025 shape: {sample_sep['image_2025'].shape}")

    print("\nDataset test completed successfully!")
