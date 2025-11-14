#!/usr/bin/env python3
"""
Create stratified train/val/test splits for land-take detection.

Splits 53 tiles into 38 train / 8 val / 7 test, stratified by change_ratio
to ensure balanced representation across low/moderate/high change levels.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
MASK_ANALYSIS_PATH = "outputs/reports/mask_analysis.csv"
OUTPUT_DIR = "outputs/splits"
RANDOM_SEED = 42

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Change level thresholds (from data analysis)
LOW_CHANGE_THRESHOLD = 5.0  # < 5%
HIGH_CHANGE_THRESHOLD = 30.0  # >= 30%


def categorize_change_level(change_ratio):
    """Categorize tile by change level."""
    if change_ratio < LOW_CHANGE_THRESHOLD:
        return "low"
    elif change_ratio < HIGH_CHANGE_THRESHOLD:
        return "moderate"
    else:
        return "high"


def create_stratified_splits(df, train_ratio, val_ratio, test_ratio, seed):
    """
    Create stratified splits based on change level.

    Args:
        df: DataFrame with refid and change_ratio columns
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        seed: Random seed for reproducibility

    Returns:
        train_df, val_df, test_df
    """
    np.random.seed(seed)

    # Categorize tiles by change level
    df["change_level"] = df["change_ratio"].apply(categorize_change_level)

    train_list = []
    val_list = []
    test_list = []

    # Stratify by change level
    for level in ["low", "moderate", "high"]:
        level_df = df[df["change_level"] == level].copy()
        n = len(level_df)

        # Calculate split sizes
        n_train = int(np.round(n * train_ratio))
        n_val = int(np.round(n * val_ratio))
        n_test = n - n_train - n_val

        # Randomly shuffle
        level_df = level_df.sample(frac=1, random_state=seed).reset_index(drop=True)

        # Split
        train_list.append(level_df.iloc[:n_train])
        val_list.append(level_df.iloc[n_train:n_train+n_val])
        test_list.append(level_df.iloc[n_train+n_val:])

    # Concatenate all levels
    train_df = pd.concat(train_list, ignore_index=True)
    val_df = pd.concat(val_list, ignore_index=True)
    test_df = pd.concat(test_list, ignore_index=True)

    return train_df, val_df, test_df


def print_split_statistics(train_df, val_df, test_df):
    """Print statistics about the splits."""
    print("\n" + "="*60)
    print("SPLIT STATISTICS")
    print("="*60)

    for split_name, split_df in [("TRAIN", train_df), ("VAL", val_df), ("TEST", test_df)]:
        print(f"\n{split_name} SET:")
        print(f"  Total tiles: {len(split_df)}")
        print(f"  Change ratio: mean={split_df['change_ratio'].mean():.2f}%, "
              f"median={split_df['change_ratio'].median():.2f}%")
        print(f"  Change level distribution:")
        for level in ["low", "moderate", "high"]:
            count = (split_df["change_level"] == level).sum()
            print(f"    {level}: {count} tiles")

    print("\n" + "="*60)
    print(f"Total tiles: {len(train_df) + len(val_df) + len(test_df)}")
    print("="*60 + "\n")


def main():
    # Load mask analysis
    print(f"Loading mask analysis from {MASK_ANALYSIS_PATH}...")
    df = pd.read_csv(MASK_ANALYSIS_PATH)

    # Filter successful tiles only
    df = df[df["success"] == True].copy()
    print(f"Found {len(df)} tiles with successful mask analysis")

    # Create splits
    print(f"\nCreating splits (seed={RANDOM_SEED})...")
    print(f"  Train: {TRAIN_RATIO*100:.0f}%")
    print(f"  Val:   {VAL_RATIO*100:.0f}%")
    print(f"  Test:  {TEST_RATIO*100:.0f}%")

    train_df, val_df, test_df = create_stratified_splits(
        df, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED
    )

    # Print statistics
    print_split_statistics(train_df, val_df, test_df)

    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save refid lists
    train_path = output_dir / "train_refids.txt"
    val_path = output_dir / "val_refids.txt"
    test_path = output_dir / "test_refids.txt"

    train_df["refid"].to_csv(train_path, index=False, header=False)
    val_df["refid"].to_csv(val_path, index=False, header=False)
    test_df["refid"].to_csv(test_path, index=False, header=False)

    print(f"Saved splits to {OUTPUT_DIR}/")
    print(f"  {train_path.name}: {len(train_df)} tiles")
    print(f"  {val_path.name}: {len(val_df)} tiles")
    print(f"  {test_path.name}: {len(test_df)} tiles")

    # Save detailed split info
    split_info_path = output_dir / "split_info.csv"
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    all_splits = pd.concat([train_df, val_df, test_df], ignore_index=True)
    all_splits[["refid", "change_ratio", "change_level", "split"]].to_csv(
        split_info_path, index=False
    )

    print(f"\nDetailed split info saved to {split_info_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
