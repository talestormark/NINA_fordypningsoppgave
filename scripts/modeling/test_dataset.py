
#!/usr/bin/env python3
"""
Comprehensive dataset testing script.

Tests:
1. File existence and accessibility
2. Data loading and shapes
3. Value ranges and data types
4. CRS and spatial properties
5. DataLoader functionality
6. Augmentation pipeline
"""

import sys
import torch
import rasterio
import numpy as np
from pathlib import Path
from dataset import (
    LandTakeDataset,
    load_refids_from_file,
    get_training_augmentation,
    get_validation_augmentation,
    create_dataloaders,
)


def test_file_structure():
    """Test that data directories and split files exist."""
    print("=" * 60)
    print("TEST 1: File Structure")
    print("=" * 60)

    # Check data directories
    vhr_dir = Path("data/raw/VHR_google")
    mask_dir = Path("data/raw/Land_take_masks")
    splits_dir = Path("outputs/splits")

    assert vhr_dir.exists(), f"VHR directory not found: {vhr_dir}"
    assert mask_dir.exists(), f"Mask directory not found: {mask_dir}"
    assert splits_dir.exists(), f"Splits directory not found: {splits_dir}"

    print(f"✓ VHR directory exists: {vhr_dir}")
    print(f"✓ Mask directory exists: {mask_dir}")
    print(f"✓ Splits directory exists: {splits_dir}")

    # Check split files
    train_file = splits_dir / "train_refids.txt"
    val_file = splits_dir / "val_refids.txt"
    test_file = splits_dir / "test_refids.txt"

    assert train_file.exists(), f"Train split file not found: {train_file}"
    assert val_file.exists(), f"Val split file not found: {val_file}"
    assert test_file.exists(), f"Test split file not found: {test_file}"

    # Load and count refids
    train_refids = load_refids_from_file(str(train_file))
    val_refids = load_refids_from_file(str(val_file))
    test_refids = load_refids_from_file(str(test_file))

    print(f"✓ Train split: {len(train_refids)} tiles")
    print(f"✓ Val split: {len(val_refids)} tiles")
    print(f"✓ Test split: {len(test_refids)} tiles")
    print(f"✓ Total: {len(train_refids) + len(val_refids) + len(test_refids)} tiles")

    return train_refids, val_refids, test_refids


def test_raw_data_loading(refid):
    """Test loading raw GeoTIFF files with rasterio."""
    print("\n" + "=" * 60)
    print("TEST 2: Raw Data Loading")
    print("=" * 60)

    vhr_path = Path(f"data/raw/VHR_google/{refid}_RGBY_Mosaic.tif")
    mask_path = Path(f"data/raw/Land_take_masks/{refid}_mask.tif")

    # Load VHR
    with rasterio.open(vhr_path) as src:
        vhr_data = src.read()
        vhr_crs = src.crs
        vhr_transform = src.transform
        vhr_bounds = src.bounds
        vhr_nodata = src.nodata

    print(f"\nVHR Image ({refid}):")
    print(f"  Path: {vhr_path}")
    print(f"  Shape: {vhr_data.shape} (bands, height, width)")
    print(f"  Dtype: {vhr_data.dtype}")
    print(f"  Value range: [{vhr_data.min()}, {vhr_data.max()}]")
    print(f"  CRS: {vhr_crs}")
    print(f"  Transform: {vhr_transform}")
    print(f"  Bounds: {vhr_bounds}")
    print(f"  NoData value: {vhr_nodata}")

    # Check expected bands
    assert vhr_data.shape[0] == 6, f"Expected 6 bands, got {vhr_data.shape[0]}"
    print(f"✓ VHR has 6 bands (2018 RGB + 2025 RGB)")

    # Load Mask
    with rasterio.open(mask_path) as src:
        mask_data = src.read(1)  # Read first band
        mask_crs = src.crs
        mask_transform = src.transform
        mask_bounds = src.bounds

    print(f"\nMask ({refid}):")
    print(f"  Path: {mask_path}")
    print(f"  Shape: {mask_data.shape}")
    print(f"  Dtype: {mask_data.dtype}")
    print(f"  Value range: [{mask_data.min()}, {mask_data.max()}]")
    print(f"  Unique values: {np.unique(mask_data)}")
    print(f"  CRS: {mask_crs}")
    print(f"  Transform: {mask_transform}")
    print(f"  Bounds: {mask_bounds}")

    # Calculate change statistics
    change_pixels = (mask_data > 0).sum()
    total_pixels = mask_data.size
    change_ratio = change_pixels / total_pixels * 100
    print(f"  Change ratio: {change_ratio:.2f}% ({change_pixels}/{total_pixels} pixels)")

    # Check CRS match
    assert mask_crs == vhr_crs, f"CRS mismatch: VHR={vhr_crs}, Mask={mask_crs}"
    print(f"✓ CRS matches between VHR and mask")

    return vhr_data, mask_data


def test_dataset_class(refids):
    """Test the LandTakeDataset class."""
    print("\n" + "=" * 60)
    print("TEST 3: Dataset Class")
    print("=" * 60)

    # Test without augmentation
    print("\nTesting without augmentation:")
    dataset = LandTakeDataset(
        refids=refids[:3],  # Test with 3 samples
        transform=None,
        return_separate_images=False,
    )

    print(f"✓ Dataset created with {len(dataset)} samples")

    # Load first sample
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  Keys: {list(sample.keys())}")
    print(f"  Image shape: {sample['image'].shape}")
    print(f"  Mask shape: {sample['mask'].shape}")
    print(f"  Image dtype: {sample['image'].dtype}")
    print(f"  Mask dtype: {sample['mask'].dtype}")
    print(f"  Image range: [{sample['image'].min():.3f}, {sample['image'].max():.3f}]")
    print(f"  Mask unique: {torch.unique(sample['mask']).tolist()}")
    print(f"  RefID: {sample['refid']}")

    # Check shapes
    assert sample['image'].ndim == 3, "Image should be 3D (C, H, W)"
    assert sample['image'].shape[0] == 6, "Image should have 6 channels"
    assert sample['mask'].ndim == 2, "Mask should be 2D (H, W)"
    print(f"✓ Shapes are correct (image: {sample['image'].shape}, mask: {sample['mask'].shape})")

    # Check value ranges
    assert 0 <= sample['image'].min() <= 1, "Image values should be normalized [0, 1]"
    assert 0 <= sample['image'].max() <= 1, "Image values should be normalized [0, 1]"
    assert set(torch.unique(sample['mask']).tolist()).issubset({0.0, 1.0}), "Mask should be binary"
    print(f"✓ Value ranges are correct")

    # Test with augmentation
    print("\nTesting with augmentation:")
    dataset_aug = LandTakeDataset(
        refids=refids[:3],
        transform=get_training_augmentation(image_size=512),
        return_separate_images=False,
    )

    sample_aug = dataset_aug[0]
    print(f"  Augmented image shape: {sample_aug['image'].shape}")
    print(f"  Augmented mask shape: {sample_aug['mask'].shape}")
    print(f"✓ Augmentation pipeline works")

    # Test with separate images mode
    print("\nTesting separate images mode:")
    dataset_sep = LandTakeDataset(
        refids=refids[:1],
        transform=get_validation_augmentation(),
        return_separate_images=True,
    )

    sample_sep = dataset_sep[0]
    print(f"  Keys: {list(sample_sep.keys())}")
    print(f"  Image 2018 shape: {sample_sep['image_2018'].shape}")
    print(f"  Image 2025 shape: {sample_sep['image_2025'].shape}")
    print(f"  Mask shape: {sample_sep['mask'].shape}")

    assert sample_sep['image_2018'].shape[0] == 3, "2018 image should have 3 channels"
    assert sample_sep['image_2025'].shape[0] == 3, "2025 image should have 3 channels"
    print(f"✓ Separate images mode works")


def test_dataloader(refids):
    """Test DataLoader functionality."""
    print("\n" + "=" * 60)
    print("TEST 4: DataLoader")
    print("=" * 60)

    from torch.utils.data import DataLoader

    # Create small dataset
    dataset = LandTakeDataset(
        refids=refids[:8],  # 8 samples for testing
        transform=get_training_augmentation(),
        return_separate_images=False,
    )

    # Create dataloader
    batch_size = 4
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
        pin_memory=False,
    )

    print(f"✓ DataLoader created (batch_size={batch_size})")

    # Test loading a batch
    batch = next(iter(dataloader))
    print(f"\nBatch contents:")
    print(f"  Image shape: {batch['image'].shape}")
    print(f"  Mask shape: {batch['mask'].shape}")
    print(f"  Batch size: {len(batch['refid'])}")
    print(f"  RefIDs: {batch['refid']}")

    # Check batch dimensions
    assert batch['image'].shape[0] == batch_size, f"Expected batch size {batch_size}"
    assert batch['image'].shape[1] == 6, "Expected 6 channels"
    assert batch['mask'].shape[0] == batch_size, "Mask batch size mismatch"
    print(f"✓ Batch shapes are correct")

    # Test iterating through dataloader
    num_batches = 0
    for batch in dataloader:
        num_batches += 1

    expected_batches = (len(dataset) + batch_size - 1) // batch_size
    assert num_batches == expected_batches, f"Expected {expected_batches} batches, got {num_batches}"
    print(f"✓ DataLoader iteration works ({num_batches} batches)")


def test_full_pipeline():
    """Test the complete data loading pipeline."""
    print("\n" + "=" * 60)
    print("TEST 5: Full Pipeline")
    print("=" * 60)

    print("\nCreating train/val/test dataloaders...")

    try:
        dataloaders = create_dataloaders(
            batch_size=4,
            num_workers=0,  # Use 0 for testing
            image_size=512,
            return_separate_images=False,
        )

        print(f"✓ Dataloaders created successfully")

        # Test each split
        for split_name, loader in dataloaders.items():
            print(f"\n{split_name.upper()} split:")
            print(f"  Dataset size: {len(loader.dataset)}")
            print(f"  Num batches: {len(loader)}")

            # Load one batch
            batch = next(iter(loader))
            print(f"  Batch image shape: {batch['image'].shape}")
            print(f"  Batch mask shape: {batch['mask'].shape}")

        print(f"\n✓ All dataloaders work correctly")

    except Exception as e:
        print(f"✗ Error creating dataloaders: {e}")
        raise


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("LAND-TAKE DATASET TESTING")
    print("=" * 60)

    try:
        # Test 1: File structure
        train_refids, val_refids, test_refids = test_file_structure()

        # Test 2: Raw data loading (use first train refid)
        test_refid = train_refids[0]
        vhr_data, mask_data = test_raw_data_loading(test_refid)

        # Test 3: Dataset class
        test_dataset_class(train_refids)

        # Test 4: DataLoader
        test_dataloader(train_refids)

        # Test 5: Full pipeline
        test_full_pipeline()

        # Summary
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        print("\nDataset is ready for training!")
        print("\nNext steps:")
        print("  1. Implement training script")
        print("  2. Test forward pass with models")
        print("  3. Start baseline training")

        return 0

    except Exception as e:
        print("\n" + "=" * 60)
        print("TEST FAILED ✗")
        print("=" * 60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
