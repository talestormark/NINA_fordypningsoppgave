#!/usr/bin/env python3
"""
PyTorch Dataset for multi-temporal Sentinel-2 land-take detection.

Handles:
- Multi-temporal Sentinel-2 sequences (quarterly/annual/bi-temporal sampling)
- Z-score normalization using pre-computed statistics
- Data augmentation consistent across time steps
- Flexible output format for different architectures (LSTM vs 3D CNN)
"""

import torch
from torch.utils.data import Dataset
import rasterio
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import albumentations as A
from albumentations.pytorch import ToTensorV2
import sys

# Add parent directories to path
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(parent_dir.parent))

from multi_temporal_experiments.config import (
    DATA_DIR,
    SENTINEL2_BANDS,
    YEARS,
    QUARTERS,
    TEMPORAL_SAMPLING_MODES,
    MT_REPORTS_DIR,
)
from tqdm import tqdm


def compute_normalization_stats(
    refids: List[str],
    sentinel2_dir: Path = None,
    sample_pixels: int = 10000,
) -> Dict[str, np.ndarray]:
    """
    Compute z-score normalization statistics (mean, std) from a list of refids.

    This should be called with ONLY the training refids for the current fold
    to avoid data leakage during cross-validation.

    Args:
        refids: List of reference IDs to compute statistics from
        sentinel2_dir: Directory containing Sentinel-2 GeoTIFF files
        sample_pixels: Number of random pixels to sample per tile (for speed)

    Returns:
        Dict with 'mean' and 'std' arrays of shape (num_bands,)
    """
    if sentinel2_dir is None:
        sentinel2_dir = DATA_DIR / "Sentinel"
    sentinel2_dir = Path(sentinel2_dir)

    num_bands = len(SENTINEL2_BANDS)
    num_time_steps = len(YEARS) * len(QUARTERS)  # 14

    # Accumulate pixel values per band
    all_values = [[] for _ in range(num_bands)]

    for refid in tqdm(refids, desc="Computing normalization stats", leave=False):
        s2_path = sentinel2_dir / f"{refid}_RGBNIRRSWIRQ_Mosaic.tif"

        if not s2_path.exists():
            print(f"Warning: Missing Sentinel-2 file for {refid}, skipping")
            continue

        try:
            with rasterio.open(s2_path) as src:
                data = src.read()  # (126, H, W)

            if data.shape[0] != num_time_steps * num_bands:
                print(f"Warning: {refid} has {data.shape[0]} bands, expected {num_time_steps * num_bands}")
                continue

            # Reshape: (126, H, W) -> (14, 9, H, W)
            data = data.reshape(num_time_steps, num_bands, data.shape[1], data.shape[2])

            # Sample random pixels for efficiency
            H, W = data.shape[2], data.shape[3]
            total_pixels = H * W

            if total_pixels > sample_pixels:
                indices = np.random.choice(total_pixels, size=sample_pixels, replace=False)
                h_idx = indices // W
                w_idx = indices % W
                samples = data[:, :, h_idx, w_idx]  # (14, 9, sample_pixels)
            else:
                samples = data.reshape(num_time_steps, num_bands, -1)

            # Accumulate values per band (pooling over time and space)
            for band_idx in range(num_bands):
                band_data = samples[:, band_idx, :].flatten()
                all_values[band_idx].append(band_data)

        except Exception as e:
            print(f"Error processing {refid}: {e}")
            continue

    # Compute statistics
    means = np.zeros(num_bands, dtype=np.float64)
    stds = np.zeros(num_bands, dtype=np.float64)

    for band_idx in range(num_bands):
        if all_values[band_idx]:
            band_values = np.concatenate(all_values[band_idx])
            means[band_idx] = np.nanmean(band_values)
            stds[band_idx] = np.nanstd(band_values)

    # Prevent division by zero
    stds[stds == 0] = 1.0

    return {
        "mean": means,
        "std": stds,
    }


class MultiTemporalSentinel2Dataset(Dataset):
    """
    Multi-temporal Sentinel-2 dataset for land-take detection.

    Args:
        refids: List of reference IDs for tiles
        sentinel2_dir: Directory containing Sentinel-2 GeoTIFF files
        mask_dir: Directory containing mask GeoTIFF files
        normalization_stats: Dict with 'mean' and 'std' arrays, or path to CSV file.
                            Should be computed from training samples only to avoid leakage.
        temporal_sampling: Sampling mode ('quarterly', 'annual', 'bi_temporal')
        transform: Albumentations transform (applied consistently across time)
        output_format: 'LSTM' (B,T,C,H,W) or '3D' (B,C,T,H,W)
    """

    def __init__(
        self,
        refids: List[str],
        sentinel2_dir: str = None,
        mask_dir: str = None,
        normalization_stats: Dict[str, np.ndarray] = None,
        temporal_sampling: str = "annual",
        transform: Optional[A.Compose] = None,
        output_format: str = "LSTM",
    ):
        self.refids = refids
        self.sentinel2_dir = Path(sentinel2_dir or DATA_DIR / "Sentinel")
        self.mask_dir = Path(mask_dir or DATA_DIR / "Land_take_masks")
        self.temporal_sampling = temporal_sampling
        self.output_format = output_format
        self.transform = transform

        # Load or use provided normalization statistics
        if normalization_stats is None:
            # Fallback to CSV file (for backwards compatibility / testing)
            normalization_stats_path = MT_REPORTS_DIR / "sentinel2_normalization_stats.csv"
            self.norm_stats = self._load_normalization_stats(normalization_stats_path)
        elif isinstance(normalization_stats, (str, Path)):
            # Path to CSV file provided
            self.norm_stats = self._load_normalization_stats(normalization_stats)
        else:
            # Dict with mean/std arrays provided directly
            self.norm_stats = normalization_stats

        # Get temporal sampling configuration
        if temporal_sampling not in TEMPORAL_SAMPLING_MODES:
            raise ValueError(f"Unknown temporal_sampling: {temporal_sampling}")

        self.sampling_config = TEMPORAL_SAMPLING_MODES[temporal_sampling]
        self.num_time_steps = self.sampling_config["num_steps"]
        self.num_bands = len(SENTINEL2_BANDS)

        # Verify files exist
        self._verify_files()

    def _load_normalization_stats(self, stats_path: Path) -> Dict[str, np.ndarray]:
        """Load normalization statistics from CSV."""
        df = pd.read_csv(stats_path)

        stats = {
            "mean": df["mean"].values,
            "std": df["std"].values,
            "min": df["min"].values,
            "max": df["max"].values,
        }

        return stats

    def _verify_files(self):
        """Verify that Sentinel-2 and mask files exist for all refids."""
        missing_s2 = []
        missing_mask = []

        for refid in self.refids:
            s2_path = self.sentinel2_dir / f"{refid}_RGBNIRRSWIRQ_Mosaic.tif"
            mask_path = self.mask_dir / f"{refid}_mask.tif"

            if not s2_path.exists():
                missing_s2.append(refid)
            if not mask_path.exists():
                missing_mask.append(refid)

        if missing_s2:
            raise FileNotFoundError(
                f"Missing Sentinel-2 files for {len(missing_s2)} refids: {missing_s2[:5]}..."
            )
        if missing_mask:
            raise FileNotFoundError(
                f"Missing mask files for {len(missing_mask)} refids: {missing_mask[:5]}..."
            )

    def __len__(self) -> int:
        return len(self.refids)

    def _load_sentinel2(self, refid: str) -> np.ndarray:
        """
        Load Sentinel-2 time series with specified temporal sampling.

        Returns:
            Array of shape (T, C, H, W) where:
                T = num_time_steps (2, 7, or 14)
                C = num_bands (9)
                H, W = spatial dimensions
        """
        s2_path = self.sentinel2_dir / f"{refid}_RGBNIRRSWIRQ_Mosaic.tif"

        with rasterio.open(s2_path) as src:
            # Read all 126 bands: 7 years × 2 quarters × 9 bands
            data = src.read()  # (126, H, W)

        # Reshape to (num_full_time_steps, num_bands, H, W)
        num_full_time_steps = len(YEARS) * len(QUARTERS)  # 14
        data = data.reshape(num_full_time_steps, self.num_bands, data.shape[1], data.shape[2])

        # Apply temporal sampling
        if self.temporal_sampling == "quarterly":
            # Use all 14 time steps
            selected_data = data  # (14, 9, H, W)

        elif self.temporal_sampling == "annual":
            # Average Q2+Q3 for each year: 7 time steps
            # But if one quarter has significantly more NaN, use the better one
            selected_data = []
            for year_idx in range(len(YEARS)):
                q2_idx = year_idx * 2
                q3_idx = year_idx * 2 + 1

                q2_data = data[q2_idx]  # (9, H, W)
                q3_data = data[q3_idx]  # (9, H, W)

                # Calculate NaN percentage for each quarter
                q2_nan_pct = np.isnan(q2_data).sum() / q2_data.size * 100
                q3_nan_pct = np.isnan(q3_data).sum() / q3_data.size * 100

                # Threshold-based selection:
                # If one quarter has >50% NaN and the other has <20% NaN, use the better one
                # Otherwise, average them (standard approach)
                if q2_nan_pct > 50 and q3_nan_pct < 20:
                    # Q2 is bad, use Q3
                    year_data = q3_data
                elif q3_nan_pct > 50 and q2_nan_pct < 20:
                    # Q3 is bad, use Q2
                    year_data = q2_data
                else:
                    # Both similar quality, average them
                    year_data = (q2_data + q3_data) / 2.0

                selected_data.append(year_data)
            selected_data = np.stack(selected_data, axis=0)  # (7, 9, H, W)

        elif self.temporal_sampling == "bi_temporal":
            # Use annual composites for first (2018) and last (2024) years only
            # This ensures season-consistency with the annual sampling mode
            selected_data = []
            for year_idx in [0, 6]:  # 2018 and 2024
                q2_idx = year_idx * 2
                q3_idx = year_idx * 2 + 1

                q2_data = data[q2_idx]  # (9, H, W)
                q3_data = data[q3_idx]  # (9, H, W)

                # Same Q2/Q3 averaging logic as annual mode
                q2_nan_pct = np.isnan(q2_data).sum() / q2_data.size * 100
                q3_nan_pct = np.isnan(q3_data).sum() / q3_data.size * 100

                if q2_nan_pct > 50 and q3_nan_pct < 20:
                    year_data = q3_data
                elif q3_nan_pct > 50 and q2_nan_pct < 20:
                    year_data = q2_data
                else:
                    year_data = (q2_data + q3_data) / 2.0

                selected_data.append(year_data)
            selected_data = np.stack(selected_data, axis=0)  # (2, 9, H, W)

        else:
            raise ValueError(f"Unknown temporal_sampling: {self.temporal_sampling}")

        return selected_data

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """
        Apply z-score normalization per band.

        Args:
            data: Array of shape (T, C, H, W)

        Returns:
            Normalized array of shape (T, C, H, W)
        """
        mean = self.norm_stats["mean"].reshape(1, -1, 1, 1)  # (1, C, 1, 1)
        std = self.norm_stats["std"].reshape(1, -1, 1, 1)    # (1, C, 1, 1)

        # Z-score normalization
        normalized = (data - mean) / std

        return normalized

    def _load_mask(self, refid: str, target_shape: Tuple[int, int]) -> np.ndarray:
        """
        Load binary mask and resample to target shape if needed.

        Args:
            refid: Reference ID
            target_shape: (H, W) to match Sentinel-2 resolution

        Returns:
            mask: (H, W) array with binary labels
        """
        from scipy.ndimage import zoom

        mask_path = self.mask_dir / f"{refid}_mask.tif"

        with rasterio.open(mask_path) as src:
            mask = src.read(1)  # (H, W)

        # Ensure binary
        mask = (mask > 0).astype(np.float32)

        # Resample if needed (mask is 10m, Sentinel-2 is 10m, so should match)
        if mask.shape != target_shape:
            zoom_factors = (target_shape[0] / mask.shape[0], target_shape[1] / mask.shape[1])
            mask = zoom(mask, zoom_factors, order=0)  # Nearest neighbor

        return mask

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            dict with keys:
                'image': Tensor of shape (T, C, H, W) or (C, T, H, W) depending on output_format
                'mask': Tensor of shape (H, W)
                'refid': Reference ID string
        """
        refid = self.refids[idx]

        # Load Sentinel-2 time series: (T, C, H, W)
        s2_data = self._load_sentinel2(refid)
        T, C, H, W = s2_data.shape

        # Load mask: (H, W)
        mask = self._load_mask(refid, target_shape=(H, W))

        # Apply augmentation (consistent across time steps)
        if self.transform is not None:
            # Reshape to (H, W, T*C) for albumentations
            s2_reshaped = s2_data.transpose(2, 3, 0, 1).reshape(H, W, -1)  # (H, W, T*C)

            augmented = self.transform(image=s2_reshaped, mask=mask)
            s2_augmented = augmented["image"]  # (H', W', T*C) after crop/resize
            mask = augmented["mask"]  # (H', W')

            # Get new dimensions after augmentation
            H_new, W_new = s2_augmented.shape[:2]

            # Reshape back to (T, C, H', W')
            s2_data = s2_augmented.reshape(H_new, W_new, T, C).transpose(2, 3, 0, 1)
        else:
            # Convert to float32
            s2_data = s2_data.astype(np.float32)
            mask = mask.astype(np.float32)

        # Normalize
        s2_data = self._normalize(s2_data)

        # Safety check: Replace NaN with 0 (nodata handling)
        if np.isnan(s2_data).any():
            print(f"WARNING: NaN detected in sample {refid} after normalization. Replacing with 0.")
            s2_data = np.nan_to_num(s2_data, nan=0.0)

        # Convert to torch tensors
        if self.output_format == "LSTM":
            # For LSTM: (T, C, H, W)
            image = torch.from_numpy(s2_data).float()
        elif self.output_format == "3D":
            # For 3D CNN: (C, T, H, W)
            image = torch.from_numpy(s2_data).float().permute(1, 0, 2, 3)
        else:
            raise ValueError(f"Unknown output_format: {self.output_format}")

        mask = torch.from_numpy(mask).float()

        return {
            "image": image,
            "mask": mask,
            "refid": refid,
        }


def get_transform(is_train: bool = True, image_size: int = 512) -> A.Compose:
    """
    Get albumentations transform for multi-temporal data.

    Args:
        is_train: Whether this is for training (enables augmentation)
        image_size: Target image size

    Returns:
        Albumentations Compose transform
    """
    if is_train:
        return A.Compose([
            A.RandomCrop(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            # Note: We don't use photometric augmentations for Sentinel-2
            # as they would distort the normalized spectral values
        ])
    else:
        return A.Compose([
            A.CenterCrop(image_size, image_size),
        ])


def get_dataloaders(
    temporal_sampling: str = "annual",
    batch_size: int = 4,
    num_workers: int = 4,
    image_size: int = 512,
    output_format: str = "LSTM",
    fold: int = None,
    num_folds: int = 5,
    seed: int = 42,
) -> Dict[str, torch.utils.data.DataLoader]:
    """
    Create train/val/test dataloaders for multi-temporal Sentinel-2.

    Args:
        temporal_sampling: 'quarterly', 'annual', or 'bi_temporal'
        batch_size: Batch size
        num_workers: Number of data loading workers
        image_size: Image size after cropping
        output_format: 'LSTM' or '3D'
        fold: Fold index for k-fold CV (0 to num_folds-1). If None, uses original split.
        num_folds: Number of folds for k-fold CV (default: 5)
        seed: Random seed for fold generation (default: 42)

    Returns:
        Dictionary with 'train', 'val', 'test' dataloaders
    """
    # Load splits
    base_dir = Path(__file__).resolve().parent.parent.parent.parent
    splits_dir = base_dir / "outputs/splits"

    train_refids_orig = [line.strip() for line in open(splits_dir / "train_refids.txt")]
    val_refids_orig = [line.strip() for line in open(splits_dir / "val_refids.txt")]
    test_refids = [line.strip() for line in open(splits_dir / "test_refids.txt")]

    # K-fold cross-validation: combine train+val, split into folds
    if fold is not None:
        from sklearn.model_selection import StratifiedKFold

        # Combine train and val for k-fold CV
        trainval_refids = train_refids_orig + val_refids_orig

        # Load change level information for stratification
        change_level_path = base_dir / "multi_temporal_experiments" / "sample_change_levels.csv"
        change_level_df = pd.read_csv(change_level_path)
        refid_to_level = dict(zip(change_level_df['refid'], change_level_df['change_level']))

        # Get change levels for trainval samples (for stratification)
        change_levels = [refid_to_level[refid] for refid in trainval_refids]

        # Create stratified k-fold splits
        skfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        splits = list(skfold.split(trainval_refids, change_levels))

        if fold < 0 or fold >= num_folds:
            raise ValueError(f"fold must be in range [0, {num_folds-1}], got {fold}")

        train_indices, val_indices = splits[fold]
        train_refids = [trainval_refids[i] for i in train_indices]
        val_refids = [trainval_refids[i] for i in val_indices]

        # Print fold info with stratification statistics
        train_levels = [change_levels[i] for i in train_indices]
        val_levels = [change_levels[i] for i in val_indices]

        print(f"\nStratified K-Fold CV: Using fold {fold}/{num_folds-1}")
        print(f"  Train samples: {len(train_refids)} (low: {train_levels.count('low')}, "
              f"mod: {train_levels.count('moderate')}, high: {train_levels.count('high')})")
        print(f"  Val samples: {len(val_refids)} (low: {val_levels.count('low')}, "
              f"mod: {val_levels.count('moderate')}, high: {val_levels.count('high')})")
    else:
        # Original single split
        train_refids = train_refids_orig
        val_refids = val_refids_orig
        print(f"\nUsing original train/val split (no k-fold CV)")
        print(f"  Train samples: {len(train_refids)}")
        print(f"  Val samples: {len(val_refids)}")

    print(f"  Test samples: {len(test_refids)} (held out)")

    # Compute normalization statistics from TRAINING samples only (per-fold)
    # This avoids data leakage during cross-validation
    print(f"\n  Computing normalization stats from {len(train_refids)} training samples...")
    norm_stats = compute_normalization_stats(train_refids)
    print(f"  Stats computed: mean range [{norm_stats['mean'].min():.1f}, {norm_stats['mean'].max():.1f}], "
          f"std range [{norm_stats['std'].min():.2f}, {norm_stats['std'].max():.2f}]")

    # Create datasets (all use same normalization stats computed from training set)
    train_dataset = MultiTemporalSentinel2Dataset(
        refids=train_refids,
        temporal_sampling=temporal_sampling,
        normalization_stats=norm_stats,
        transform=get_transform(is_train=True, image_size=image_size),
        output_format=output_format,
    )

    val_dataset = MultiTemporalSentinel2Dataset(
        refids=val_refids,
        temporal_sampling=temporal_sampling,
        normalization_stats=norm_stats,
        transform=get_transform(is_train=False, image_size=image_size),
        output_format=output_format,
    )

    test_dataset = MultiTemporalSentinel2Dataset(
        refids=test_refids,
        temporal_sampling=temporal_sampling,
        normalization_stats=norm_stats,
        transform=get_transform(is_train=False, image_size=image_size),
        output_format=output_format,
    )

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }


if __name__ == "__main__":
    """Test the dataset."""
    print("Testing MultiTemporalSentinel2Dataset...")

    # Test with annual sampling (use small crop since Sentinel-2 tiles are only ~65x65)
    dataloaders = get_dataloaders(
        temporal_sampling="annual",
        batch_size=2,
        num_workers=0,
        image_size=64,  # Smaller size to fit Sentinel-2 tiles
        output_format="LSTM",
    )

    # Get one batch
    batch = next(iter(dataloaders["train"]))

    print(f"\nBatch contents:")
    print(f"  Image shape: {batch['image'].shape}")  # Expected: (B, T, C, H, W) = (2, 7, 9, 64, 64)
    print(f"  Mask shape: {batch['mask'].shape}")    # Expected: (B, H, W) = (2, 64, 64)
    print(f"  Refids: {batch['refid']}")

    print(f"\nImage statistics:")
    print(f"  Min: {batch['image'].min():.2f}")
    print(f"  Max: {batch['image'].max():.2f}")
    print(f"  Mean: {batch['image'].mean():.2f}")
    print(f"  Std: {batch['image'].std():.2f}")

    print("\n✓ Dataset test passed!")
