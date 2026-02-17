#!/usr/bin/env python3
"""
Compute normalization statistics for Sentinel-2 bands.

Calculates mean and std for each of the 9 spectral bands across all time steps
from the training set. These statistics are used for z-score normalization in
the multi-temporal dataset class.

Usage:
    python 03_compute_normalization_stats.py [--output-dir DIR] [--sample-size N]
"""

import sys
from pathlib import Path

# Add parent directory to path to import config
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(parent_dir.parent))

import argparse
import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm

try:
    from PART1_multi_temporal_experiments.config import (
        DATA_DIR,
        SENTINEL2_BANDS,
        SENTINEL2_NUM_BANDS,
        YEARS,
        QUARTERS,
        MT_REPORTS_DIR,
    )
except ImportError:
    # Fallback if config import fails
    DATA_DIR = Path("data/raw")
    SENTINEL2_BANDS = ["blue", "green", "red", "R1", "R2", "R3", "nir", "swir1", "swir2"]
    SENTINEL2_NUM_BANDS = 9
    YEARS = list(range(2018, 2025))
    QUARTERS = [2, 3]
    MT_REPORTS_DIR = Path("PART1_multi_temporal_experiments/outputs/reports")

# Paths
SENTINEL2_DIR = DATA_DIR / "Sentinel"
# Splits are in the parent repo directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
SPLITS_DIR = BASE_DIR / "outputs/splits"
TRAIN_SPLIT_FILE = SPLITS_DIR / "train_refids.txt"


def load_train_refids(split_file):
    """Load training set reference IDs."""
    with open(split_file, "r") as f:
        refids = [line.strip() for line in f if line.strip()]
    return refids


def compute_simple_stats(refids, sentinel2_dir, num_bands, sample_pixels=10000):
    """
    Compute mean and std using simple accumulation (much more reliable).

    Args:
        refids: List of reference IDs for training tiles
        sentinel2_dir: Path to Sentinel-2 directory
        num_bands: Number of spectral bands (9)
        sample_pixels: Number of random pixels to sample per tile (for speed)

    Returns:
        means: Array of shape (num_bands,) with mean per band
        stds: Array of shape (num_bands,) with std per band
        mins: Array of shape (num_bands,) with min per band
        maxs: Array of shape (num_bands,) with max per band
        n_total: Total number of pixels processed
    """
    # Accumulate all values per band
    all_values = [[] for _ in range(num_bands)]
    mins = np.full(num_bands, np.inf, dtype=np.float64)
    maxs = np.full(num_bands, -np.inf, dtype=np.float64)

    print(f"\nProcessing {len(refids)} training tiles...")
    print(f"Sampling {sample_pixels} random pixels per tile for efficiency")

    for refid in tqdm(refids, desc="Loading tiles"):
        sentinel_path = sentinel2_dir / f"{refid}_RGBNIRRSWIRQ_Mosaic.tif"

        if not sentinel_path.exists():
            print(f"Warning: Missing Sentinel-2 file for {refid}, skipping")
            continue

        try:
            with rasterio.open(sentinel_path) as src:
                data = src.read()  # (126, H, W)

                num_time_steps = len(YEARS) * len(QUARTERS)
                if data.shape[0] != num_time_steps * num_bands:
                    print(f"Warning: {refid} has {data.shape[0]} bands, expected {num_time_steps * num_bands}")
                    continue

                # Reshape: (126, H, W) -> (14, 9, H, W)
                data = data.reshape(num_time_steps, num_bands, data.shape[1], data.shape[2])

                # Sample random pixels
                H, W = data.shape[2], data.shape[3]
                total_pixels = H * W

                if total_pixels > sample_pixels:
                    indices = np.random.choice(total_pixels, size=sample_pixels, replace=False)
                    h_idx = indices // W
                    w_idx = indices % W
                    samples = data[:, :, h_idx, w_idx]  # (14, 9, sample_pixels)
                else:
                    samples = data.reshape(num_time_steps, num_bands, -1)  # (14, 9, all_pixels)

                # Pool over time and space: (9, num_samples)
                for band_idx in range(num_bands):
                    # Extract band across all time steps: (14, num_samples)
                    band_data = samples[:, band_idx, :].flatten()  # Flatten time and space

                    # Update min/max
                    mins[band_idx] = min(mins[band_idx], band_data.min())
                    maxs[band_idx] = max(maxs[band_idx], band_data.max())

                    # Accumulate values
                    all_values[band_idx].append(band_data)

        except Exception as e:
            print(f"Error processing {refid}: {e}")
            continue

    # Compute statistics from accumulated values
    print("\nComputing final statistics...")

    means = np.zeros(num_bands, dtype=np.float64)
    stds = np.zeros(num_bands, dtype=np.float64)
    nan_counts = np.zeros(num_bands, dtype=np.int64)

    for band_idx in range(num_bands):
        if all_values[band_idx]:
            band_values = np.concatenate(all_values[band_idx])
            nan_counts[band_idx] = np.sum(np.isnan(band_values))
            means[band_idx] = np.nanmean(band_values)  # Use nanmean to ignore NaN values
            stds[band_idx] = np.nanstd(band_values)    # Use nanstd to ignore NaN values

    total_nan = nan_counts.sum()
    total_values = sum(len(np.concatenate(vals)) if vals else 0 for vals in all_values)
    print(f"  Found {total_nan:,} NaN values ({100*total_nan/total_values:.2f}% of data)")

    n_total = sum(len(np.concatenate(vals)) if vals else 0 for vals in all_values) // num_bands

    return means, stds, mins, maxs, n_total


def compute_online_stats(refids, sentinel2_dir, num_bands, sample_pixels=10000):
    """
    Compute mean and std for each band using Welford's online algorithm.

    This is memory-efficient and numerically stable for computing statistics
    across many large files.

    Args:
        refids: List of reference IDs for training tiles
        sentinel2_dir: Path to Sentinel-2 directory
        num_bands: Number of spectral bands (9)
        sample_pixels: Number of random pixels to sample per tile (for speed)

    Returns:
        means: Array of shape (num_bands,) with mean per band
        stds: Array of shape (num_bands,) with std per band
        mins: Array of shape (num_bands,) with min per band
        maxs: Array of shape (num_bands,) with max per band
        n_total: Total number of pixels processed
    """
    # Initialize accumulators (Welford's algorithm)
    n = np.zeros(num_bands, dtype=np.int64)  # Counter per band
    means = np.zeros(num_bands, dtype=np.float64)
    M2 = np.zeros(num_bands, dtype=np.float64)
    mins = np.full(num_bands, np.inf, dtype=np.float64)
    maxs = np.full(num_bands, -np.inf, dtype=np.float64)

    print(f"\nProcessing {len(refids)} training tiles...")
    print(f"Sampling {sample_pixels} random pixels per tile for efficiency")

    tile_count = 0
    for refid in tqdm(refids, desc="Computing stats"):
        tile_count += 1
        # Sentinel-2 file: {refid}_RGBNIRRSWIRQ_Mosaic.tif
        sentinel_path = sentinel2_dir / f"{refid}_RGBNIRRSWIRQ_Mosaic.tif"

        if not sentinel_path.exists():
            print(f"Warning: Missing Sentinel-2 file for {refid}, skipping")
            continue

        try:
            with rasterio.open(sentinel_path) as src:
                # Read all 126 bands (14 time steps × 9 spectral bands)
                data = src.read()  # Shape: (126, H, W)

                # Reshape to (num_time_steps, num_bands, H, W)
                # Sentinel-2 bands are organized as: [year1_q1_band1, year1_q1_band2, ...]
                num_time_steps = len(YEARS) * len(QUARTERS)  # 14

                if data.shape[0] != num_time_steps * num_bands:
                    print(f"Warning: {refid} has {data.shape[0]} bands, expected {num_time_steps * num_bands}")
                    continue

                # Reshape: (126, H, W) -> (14, 9, H, W)
                data = data.reshape(num_time_steps, num_bands, data.shape[1], data.shape[2])

                # Sample random pixels for efficiency
                H, W = data.shape[2], data.shape[3]
                total_pixels = H * W

                if total_pixels > sample_pixels:
                    # Random sampling
                    indices = np.random.choice(total_pixels, size=sample_pixels, replace=False)
                    h_idx = indices // W
                    w_idx = indices % W

                    # Extract samples: (14, 9, sample_pixels)
                    samples = data[:, :, h_idx, w_idx]
                else:
                    # Use all pixels if tile is small
                    samples = data.reshape(num_time_steps, num_bands, -1)

                # Flatten across time and spatial dimensions: (9, num_samples)
                # We want to compute stats per spectral band, pooling over time and space
                samples = samples.reshape(num_time_steps * samples.shape[2], num_bands).T  # (9, N)

                # Update statistics for each band using Welford's algorithm
                for band_idx in range(num_bands):
                    band_values = samples[band_idx]

                    # Update min/max
                    mins[band_idx] = min(mins[band_idx], band_values.min())
                    maxs[band_idx] = max(maxs[band_idx], band_values.max())

                    # Welford's online algorithm for mean and variance
                    for value in band_values:
                        n[band_idx] += 1
                        delta = value - means[band_idx]
                        means[band_idx] += delta / n[band_idx]
                        delta2 = value - means[band_idx]
                        M2[band_idx] += delta * delta2

                        # Debug: Check if NaN appears
                        if np.isnan(means[band_idx]) and tile_count == 1 and band_idx == 0:
                            print(f"\n!!!NaN detected at iteration {n[band_idx]}:")
                            print(f"  value={value}, delta={delta}, delta2={delta2}")
                            print(f"  n[band_idx]={n[band_idx]}, M2[band_idx]={M2[band_idx]}")
                            break

        except Exception as e:
            print(f"Error processing {refid}: {e}")
            continue

    # Compute standard deviation (per band)
    stds = np.zeros(num_bands, dtype=np.float64)
    for band_idx in range(num_bands):
        if n[band_idx] > 1:
            stds[band_idx] = np.sqrt(M2[band_idx] / n[band_idx])

    n_total = n.sum()  # Total pixels across all bands

    # Debug: Print sample counts per band
    print(f"\nDebug: Sample counts per band: {n}")
    print(f"Debug: Means shape: {means.shape}, values: {means}")
    print(f"Debug: Any NaN in means: {np.any(np.isnan(means))}")

    return means, stds, mins, maxs, n_total


def compute_stats_batch(refids, sentinel2_dir, num_bands):
    """
    Alternative: Compute stats by accumulating all pixel values (memory-intensive).

    Only use this if you have sufficient RAM (requires loading all training tiles).
    """
    all_values = [[] for _ in range(num_bands)]

    print(f"\nProcessing {len(refids)} training tiles...")

    for refid in tqdm(refids, desc="Loading tiles"):
        sentinel_path = sentinel2_dir / f"{refid}_RGBNIRRSWIRQ_Mosaic.tif"

        if not sentinel_path.exists():
            continue

        try:
            with rasterio.open(sentinel_path) as src:
                data = src.read()  # (126, H, W)

                num_time_steps = len(YEARS) * len(QUARTERS)
                data = data.reshape(num_time_steps, num_bands, data.shape[1], data.shape[2])

                # Pool over time and spatial dimensions
                # Shape: (9, num_time_steps * H * W)
                data = data.reshape(num_time_steps, num_bands, -1)
                data = data.transpose(1, 0, 2).reshape(num_bands, -1)

                for band_idx in range(num_bands):
                    all_values[band_idx].append(data[band_idx])

        except Exception as e:
            print(f"Error processing {refid}: {e}")
            continue

    # Compute statistics
    means = np.array([np.mean(np.concatenate(vals)) for vals in all_values])
    stds = np.array([np.std(np.concatenate(vals)) for vals in all_values])
    mins = np.array([np.min(np.concatenate(vals)) for vals in all_values])
    maxs = np.array([np.max(np.concatenate(vals)) for vals in all_values])
    n_total = sum(len(np.concatenate(vals)) for vals in all_values) // num_bands

    return means, stds, mins, maxs, n_total


def save_stats(means, stds, mins, maxs, n_total, output_path):
    """Save normalization statistics to CSV."""
    df = pd.DataFrame({
        "band": SENTINEL2_BANDS,
        "mean": means,
        "std": stds,
        "min": mins,
        "max": maxs,
    })

    df.to_csv(output_path, index=False)

    print(f"\n✓ Saved normalization statistics to {output_path}")
    print(f"  Total pixels processed: {n_total:,}")
    print(f"\nStatistics:")
    print(df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="Compute Sentinel-2 normalization statistics from training set"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(MT_REPORTS_DIR),
        help="Output directory for statistics CSV",
    )
    parser.add_argument(
        "--sample-pixels",
        type=int,
        default=10000,
        help="Number of random pixels to sample per tile (0 = use all pixels)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="online",
        choices=["online", "batch"],
        help="Computation method: 'online' (memory-efficient) or 'batch' (exact but memory-intensive)",
    )

    args = parser.parse_args()

    print("="*80)
    print("SENTINEL-2 NORMALIZATION STATISTICS COMPUTATION")
    print("="*80)

    # Load training refids
    print(f"\nLoading training split from {TRAIN_SPLIT_FILE}...")
    if not TRAIN_SPLIT_FILE.exists():
        print(f"ERROR: Training split file not found: {TRAIN_SPLIT_FILE}")
        print("Please run scripts/modeling/01_create_splits.py first")
        return

    train_refids = load_train_refids(TRAIN_SPLIT_FILE)
    print(f"Found {len(train_refids)} training tiles")

    # Compute statistics using simple accumulation (most reliable)
    means, stds, mins, maxs, n_total = compute_simple_stats(
        train_refids, SENTINEL2_DIR, SENTINEL2_NUM_BANDS, args.sample_pixels
    )

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "sentinel2_normalization_stats.csv"

    save_stats(means, stds, mins, maxs, n_total, output_path)

    print("\n" + "="*80)
    print("DONE - Statistics ready for z-score normalization")
    print("="*80)
    print(f"\nNext step: Use these stats in MultiTemporalSentinel2Dataset class")
    print(f"  normalized = (value - mean) / std")


if __name__ == "__main__":
    main()
