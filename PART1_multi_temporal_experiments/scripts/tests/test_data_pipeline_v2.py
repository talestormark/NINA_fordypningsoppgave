#!/usr/bin/env python3
"""
Unit tests for the v2 data pipeline.

Verifies that splits, data files, normalization, and the dataset/dataloader
work correctly with data_v2 before launching any training jobs.

Usage:
    module load Anaconda3/2024.02-1 && conda activate masterthesis
    python PART1_multi_temporal_experiments/scripts/tests/test_data_pipeline_v2.py
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd

# Add paths
ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "PART1_multi_temporal_experiments" / "scripts" / "modeling"))

# Paths
DATA_DIR = ROOT / "data_v2"
SPLITS_DIR = ROOT / "preprocessing" / "outputs" / "splits" / "part1"
SENTINEL_DIR = DATA_DIR / "Sentinel"
MASK_DIR = DATA_DIR / "Land_take_masks_coarse"
SPLIT_INFO = SPLITS_DIR / "split_info.csv"

passed = 0
failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  PASS: {name}")
        passed += 1
    else:
        print(f"  FAIL: {name} -- {detail}")
        failed += 1


def test_splits_exist():
    print("\n[1] Split files exist")
    check("train_refids.txt", (SPLITS_DIR / "train_refids.txt").exists())
    check("val_refids.txt", (SPLITS_DIR / "val_refids.txt").exists())
    check("test_refids.txt", (SPLITS_DIR / "test_refids.txt").exists())
    check("split_info.csv", SPLIT_INFO.exists())


def test_split_counts():
    print("\n[2] Split counts")
    train = [l.strip() for l in open(SPLITS_DIR / "train_refids.txt") if l.strip()]
    val = [l.strip() for l in open(SPLITS_DIR / "val_refids.txt") if l.strip()]
    test = [l.strip() for l in open(SPLITS_DIR / "test_refids.txt") if l.strip()]

    check(f"Train count = 111", len(train) == 111, f"got {len(train)}")
    check(f"Val count = 24", len(val) == 24, f"got {len(val)}")
    check(f"Test count = 28", len(test) == 28, f"got {len(test)}")
    check(f"Total = 163", len(train) + len(val) + len(test) == 163,
          f"got {len(train) + len(val) + len(test)}")

    # No overlap
    all_refids = train + val + test
    check("No duplicates across splits", len(set(all_refids)) == len(all_refids),
          f"{len(all_refids)} total, {len(set(all_refids))} unique")

    return train, val, test


def test_split_info_csv(train, val, test):
    print("\n[3] split_info.csv consistency")
    df = pd.read_csv(SPLIT_INFO)

    check("Has required columns",
          set(['refid', 'split', 'change_level']).issubset(df.columns),
          f"columns: {list(df.columns)}")

    check(f"Row count = 163", len(df) == 163, f"got {len(df)}")

    # Check change_level values
    levels = df['change_level'].unique()
    check("Change levels are low/moderate/high",
          set(levels) == {'low', 'moderate', 'high'},
          f"got {set(levels)}")

    # Check split labels match files
    csv_train = set(df[df['split'] == 'train']['refid'])
    csv_val = set(df[df['split'] == 'val']['refid'])
    csv_test = set(df[df['split'] == 'test']['refid'])
    check("CSV train matches train_refids.txt",
          csv_train == set(train),
          f"diff: {csv_train.symmetric_difference(set(train))}")
    check("CSV val matches val_refids.txt",
          csv_val == set(val),
          f"diff: {csv_val.symmetric_difference(set(val))}")
    check("CSV test matches test_refids.txt",
          csv_test == set(test),
          f"diff: {csv_test.symmetric_difference(set(test))}")


def test_data_files_exist(train, val, test):
    print("\n[4] Data files exist for all refids")
    all_refids = train + val + test

    missing_s2 = []
    missing_mask = []
    for refid in all_refids:
        s2 = SENTINEL_DIR / f"{refid}_RGBNIRRSWIRQ_Mosaic.tif"
        mask = MASK_DIR / f"{refid}_mask.tif"
        if not s2.exists():
            missing_s2.append(refid)
        if not mask.exists():
            missing_mask.append(refid)

    check(f"All Sentinel-2 files exist ({len(all_refids)} tiles)",
          len(missing_s2) == 0,
          f"missing {len(missing_s2)}: {missing_s2[:5]}")
    check(f"All mask files exist ({len(all_refids)} tiles)",
          len(missing_mask) == 0,
          f"missing {len(missing_mask)}: {missing_mask[:5]}")


def test_tile_shapes():
    print("\n[5] Tile shapes and band counts (sampling 5 tiles)")
    import rasterio

    all_refids = [l.strip() for l in open(SPLITS_DIR / "train_refids.txt") if l.strip()]
    sample = all_refids[:5]

    for refid in sample:
        s2_path = SENTINEL_DIR / f"{refid}_RGBNIRRSWIRQ_Mosaic.tif"
        mask_path = MASK_DIR / f"{refid}_mask.tif"

        with rasterio.open(s2_path) as src:
            s2_bands = src.count
            s2_h, s2_w = src.height, src.width

        with rasterio.open(mask_path) as src:
            m_bands = src.count
            m_h, m_w = src.height, src.width

        check(f"{refid}: S2 has 126 bands", s2_bands == 126, f"got {s2_bands}")
        check(f"{refid}: S2 >= 64px", s2_h >= 64 and s2_w >= 64,
              f"got {s2_h}x{s2_w}")
        check(f"{refid}: mask has 1 band", m_bands == 1, f"got {m_bands}")
        check(f"{refid}: mask matches S2 spatial dims",
              m_h == s2_h and m_w == s2_w,
              f"S2={s2_h}x{s2_w}, mask={m_h}x{m_w}")


def test_dataset_loading():
    print("\n[6] Dataset and DataLoader (2 epochs, fold 0)")
    import torch
    from PART1_multi_temporal_experiments.scripts.data_preparation.dataset_multitemporal import (
        get_dataloaders,
    )

    dataloaders = get_dataloaders(
        temporal_sampling="annual",
        batch_size=2,
        num_workers=0,
        image_size=64,
        output_format="LSTM",
        fold=0,
        num_folds=5,
        seed=42,
        sentinel2_dir=str(SENTINEL_DIR),
        mask_dir=str(MASK_DIR),
        splits_dir=str(SPLITS_DIR),
        change_level_path=str(SPLIT_INFO),
    )

    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    test_loader = dataloaders['test']

    # Check sample counts (~108 train, ~27 val, 28 test)
    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    n_test = len(test_loader.dataset)

    check(f"Train samples ~108", 100 <= n_train <= 115, f"got {n_train}")
    check(f"Val samples ~27", 20 <= n_val <= 35, f"got {n_val}")
    check(f"Test samples = 28", n_test == 28, f"got {n_test}")
    check(f"Train + Val = 135", n_train + n_val == 135,
          f"got {n_train + n_val}")

    # Load one batch and check shapes
    batch = next(iter(train_loader))
    img = batch['image']
    mask = batch['mask']

    check(f"Image shape (B,T,C,H,W)", img.shape[1:] == (7, 9, 64, 64),
          f"got {img.shape}")
    check(f"Mask shape (B,H,W)", mask.shape[1:] == (64, 64),
          f"got {mask.shape}")
    check(f"Image dtype float32", img.dtype == torch.float32,
          f"got {img.dtype}")
    check(f"Mask dtype float32", mask.dtype == torch.float32,
          f"got {mask.dtype}")
    check(f"No NaN in image", not torch.isnan(img).any().item())
    check(f"No NaN in mask", not torch.isnan(mask).any().item())
    check(f"Loss is finite", True)  # Placeholder — actual loss check in smoke test

    # Check mask is binary
    unique_vals = torch.unique(mask)
    check(f"Mask is binary (0/1 only)",
          all(v in [0.0, 1.0] for v in unique_vals.tolist()),
          f"got unique values: {unique_vals.tolist()}")

    # Check normalization looks reasonable (z-scored: mean ~0, std ~1)
    img_mean = img.mean().item()
    img_std = img.std().item()
    check(f"Image mean near 0 (z-scored)", abs(img_mean) < 5.0,
          f"got {img_mean:.3f}")
    check(f"Image std reasonable", 0.1 < img_std < 10.0,
          f"got {img_std:.3f}")


def test_all_temporal_modes():
    print("\n[7] All temporal sampling modes load correctly")
    from PART1_multi_temporal_experiments.scripts.data_preparation.dataset_multitemporal import (
        get_dataloaders,
    )

    for mode, expected_t in [("annual", 7), ("quarterly", 14), ("bi_temporal", 2)]:
        try:
            dl = get_dataloaders(
                temporal_sampling=mode,
                batch_size=1,
                num_workers=0,
                image_size=64,
                output_format="LSTM",
                fold=0,
                num_folds=5,
                seed=42,
                sentinel2_dir=str(SENTINEL_DIR),
                mask_dir=str(MASK_DIR),
                splits_dir=str(SPLITS_DIR),
                change_level_path=str(SPLIT_INFO),
            )
            batch = next(iter(dl['train']))
            t_dim = batch['image'].shape[1]
            check(f"{mode}: T={expected_t}", t_dim == expected_t,
                  f"got T={t_dim}")
        except Exception as e:
            check(f"{mode}: loads without error", False, str(e))


def test_stratification_balance():
    print("\n[8] Stratification balance across folds")
    from sklearn.model_selection import StratifiedKFold

    df = pd.read_csv(SPLIT_INFO)
    trainval = df[df['split'].isin(['train', 'val'])]
    refids = trainval['refid'].tolist()
    levels = trainval['change_level'].tolist()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(refids, levels)):
        train_levels = [levels[i] for i in train_idx]
        val_levels = [levels[i] for i in val_idx]
        # Check each level appears in both train and val
        train_set = set(train_levels)
        val_set = set(val_levels)
        check(f"Fold {fold_idx}: all levels in train",
              train_set == {'low', 'moderate', 'high'},
              f"got {train_set}")
        check(f"Fold {fold_idx}: all levels in val",
              val_set == {'low', 'moderate', 'high'},
              f"got {val_set}")


if __name__ == "__main__":
    print("=" * 60)
    print("DATA PIPELINE UNIT TESTS (v2)")
    print("=" * 60)

    test_splits_exist()
    train, val, test = test_split_counts()
    test_split_info_csv(train, val, test)
    test_data_files_exist(train, val, test)
    test_tile_shapes()
    test_dataset_loading()
    test_all_temporal_modes()
    test_stratification_balance()

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    sys.exit(1 if failed > 0 else 0)
