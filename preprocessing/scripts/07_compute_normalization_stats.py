#!/usr/bin/env python3
"""
Compute per-band normalization statistics (mean, std) for each modality
from training tiles only, to avoid data leakage.

Produces two output files:
    - Part 1 (Sentinel-2 only, 111 train tiles):
      preprocessing/outputs/normalization_stats_part1.csv
    - Part 2 (S2 + PS + AE + VHR, 176 train tiles):
      preprocessing/outputs/normalization_stats_part2.json

The Part 1 CSV has columns: band, mean, std, min, max, p2, p98
The Part 2 JSON follows the existing format in
    data_v1/processed/epsg3035_10m_v1/normalisation_stats.json

Nodata masking strategy (per modality):
    - S2 / AlphaEarth (float): exclude pixels where all bands == 0 or any
      band is NaN. Zero is a reliable nodata indicator for reflectance and
      learned embeddings (real values are never exactly zero across all bands).
    - PlanetScope / VHR (uint8 RGB): exclude pixels where all three bands
      are 0 AND the pixel lies in the image border (sum of neighbours is also
      zero). This avoids discarding genuine dark pixels (shadow, water) in the
      image interior. In practice, the GEE export only produces all-zero
      pixels in the triangular reprojection corners.

Outlier robustness:
    Percentile clipping at [p2, p98] per band before computing mean/std.
    The same clip bounds are stored in the output so the dataloader can
    apply identical clipping at runtime before z-scoring.
"""

import json
import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
from typing import List, Tuple, Dict

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_V2 = PROJECT_ROOT / "data_v2"
OUTPUT_DIR = PROJECT_ROOT / "preprocessing" / "outputs"
SPLITS_DIR = OUTPUT_DIR / "splits"

# Band layout
S2_BAND_NAMES = ["blue", "green", "red", "R1", "R2", "R3", "nir", "swir1", "swir2"]
S2_N_BANDS = 9
S2_N_TIMESTEPS = 14  # 7 years x 2 quarters

PS_BAND_NAMES = ["blue", "green", "red"]
PS_N_BANDS = 3
PS_N_TIMESTEPS = 14

AE_N_FEATURES = 64
AE_N_YEARS = 7

VHR_BAND_NAMES = ["red", "green", "blue"]
VHR_N_BANDS = 3
VHR_N_DATES = 2  # start year + end year

SAMPLE_PIXELS = 10_000
CLIP_LOW = 2    # percentile
CLIP_HIGH = 98  # percentile


def load_refids(path: Path) -> List[str]:
    """Load refids from a text file (one per line)."""
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _is_border_zero(frame: np.ndarray) -> np.ndarray:
    """
    Identify zero-padded border pixels for uint8 RGB data.

    A pixel is considered border nodata if all bands are 0 AND at least one
    of its 4-connected neighbours also has all bands == 0. This distinguishes
    the triangular reprojection corners (contiguous zero regions at edges)
    from isolated dark pixels in the image interior.

    Args:
        frame: (C, H, W) uint8 array

    Returns:
        (H, W) boolean mask — True for border zero-fill pixels
    """
    all_zero = np.all(frame == 0, axis=0)  # (H, W)
    if not all_zero.any():
        return all_zero

    # A zero pixel adjacent to another zero pixel is border fill.
    # Shift in 4 directions and check if any neighbour is also all-zero.
    padded = np.pad(all_zero, 1, mode="constant", constant_values=True)
    has_zero_neighbour = (
        padded[:-2, 1:-1] |  # top
        padded[2:, 1:-1]  |  # bottom
        padded[1:-1, :-2] |  # left
        padded[1:-1, 2:]     # right
    )
    return all_zero & has_zero_neighbour


def _make_valid_mask(frame: np.ndarray, is_uint8: bool) -> np.ndarray:
    """
    Create a valid-pixel mask for one timestep.

    Args:
        frame: (C, H, W) array for one timestep
        is_uint8: if True, use border-aware zero detection (for PS/VHR)

    Returns:
        (H, W) boolean mask — True for valid pixels
    """
    any_nan = np.any(np.isnan(frame), axis=0)

    if is_uint8:
        nodata = _is_border_zero(frame)
    else:
        # For float data (S2/AE), all-zero is a reliable nodata indicator
        nodata = np.all(frame == 0, axis=0)

    return ~nodata & ~any_nan


def _clip_and_compute(
    stacked: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Apply percentile clipping per band, then compute mean/std on clipped data.

    Args:
        stacked: (N, n_bands) array of sampled valid pixels

    Returns:
        dict with keys: mean, std, min, max, p2, p98 (all per band)
    """
    n_bands = stacked.shape[1]
    p2 = np.percentile(stacked, CLIP_LOW, axis=0)
    p98 = np.percentile(stacked, CLIP_HIGH, axis=0)

    # Clip each band to its [p2, p98] range
    clipped = np.empty_like(stacked)
    for b in range(n_bands):
        clipped[:, b] = np.clip(stacked[:, b], p2[b], p98[b])

    return {
        "mean": np.mean(clipped, axis=0),
        "std": np.std(clipped, axis=0),
        "min": np.min(stacked, axis=0),
        "max": np.max(stacked, axis=0),
        "p2": p2,
        "p98": p98,
    }


def compute_spectral_stats(
    refids: List[str],
    folder: Path,
    suffix: str,
    n_spectral: int,
    n_timesteps: int,
    is_uint8: bool = False,
    sample_pixels: int = SAMPLE_PIXELS,
) -> Dict[str, np.ndarray]:
    """
    Compute per-spectral-band statistics across all timesteps.

    Returns dict with keys: mean, std, min, max, p2, p98, valid_pixel_fraction.
    """
    all_samples = []
    total_pixels = 0
    valid_pixels = 0

    for refid in refids:
        matches = list(folder.glob(f"{refid}*{suffix}"))
        if not matches:
            print(f"  WARNING: no file for {refid[:30]}... in {folder.name}")
            continue

        with rasterio.open(matches[0]) as src:
            data = src.read()  # (bands, H, W)
            h, w = src.height, src.width

        total_bands = data.shape[0]
        expected = n_spectral * n_timesteps
        if total_bands != expected:
            print(f"  WARNING: {refid[:30]}... has {total_bands} bands, expected {expected}")
            continue

        data = data.reshape(n_timesteps, n_spectral, h, w)

        for t in range(n_timesteps):
            frame = data[t]  # (C, H, W)
            valid = _make_valid_mask(frame, is_uint8=is_uint8)

            total_pixels += h * w
            n_valid = int(valid.sum())
            valid_pixels += n_valid

            if n_valid == 0:
                continue

            pixels = frame[:, valid].T  # (n_valid, C)
            if pixels.shape[0] > sample_pixels:
                idx = np.random.choice(pixels.shape[0], sample_pixels, replace=False)
                pixels = pixels[idx]

            all_samples.append(pixels.astype(np.float64))

    if not all_samples:
        raise ValueError(f"No valid data found in {folder.name}")

    stacked = np.concatenate(all_samples, axis=0)
    print(f"  {folder.name}: {stacked.shape[0]:,} sample pixels from {len(refids)} tiles")

    stats = _clip_and_compute(stacked)
    stats["valid_pixel_fraction"] = valid_pixels / total_pixels if total_pixels > 0 else 0.0
    return stats


def compute_ae_stats(
    refids: List[str],
    folder: Path,
    suffix: str = "_VEY_Mosaic.tif",
    sample_pixels: int = SAMPLE_PIXELS,
) -> Dict[str, np.ndarray]:
    """
    Compute per-feature statistics for AlphaEarth embeddings.

    Layout: (n_years * n_features, H, W) -> (n_years, n_features, H, W)
    """
    all_samples = []
    total_pixels = 0
    valid_pixels = 0

    for refid in refids:
        matches = list(folder.glob(f"{refid}*{suffix}"))
        if not matches:
            print(f"  WARNING: no AE file for {refid[:30]}...")
            continue

        with rasterio.open(matches[0]) as src:
            data = src.read()  # (448, H, W)
            h, w = src.height, src.width

        expected = AE_N_FEATURES * AE_N_YEARS
        if data.shape[0] != expected:
            print(f"  WARNING: {refid[:30]}... has {data.shape[0]} bands, expected {expected}")
            continue

        data = data.reshape(AE_N_YEARS, AE_N_FEATURES, h, w)

        for y in range(AE_N_YEARS):
            frame = data[y]  # (64, H, W)
            valid = _make_valid_mask(frame, is_uint8=False)

            total_pixels += h * w
            n_valid = int(valid.sum())
            valid_pixels += n_valid

            if n_valid == 0:
                continue

            pixels = frame[:, valid].T  # (n_valid, 64)
            if pixels.shape[0] > sample_pixels:
                idx = np.random.choice(pixels.shape[0], sample_pixels, replace=False)
                pixels = pixels[idx]

            all_samples.append(pixels.astype(np.float64))

    if not all_samples:
        raise ValueError("No valid AlphaEarth data found")

    stacked = np.concatenate(all_samples, axis=0)
    print(f"  AlphaEarth: {stacked.shape[0]:,} sample pixels from {len(refids)} tiles")

    stats = _clip_and_compute(stacked)
    stats["valid_pixel_fraction"] = valid_pixels / total_pixels if total_pixels > 0 else 0.0
    return stats


def compute_vhr_stats(
    refids: List[str],
    folder: Path,
    suffix: str = "_RGBY_Mosaic.tif",
    sample_pixels: int = SAMPLE_PIXELS,
) -> Dict[str, np.ndarray]:
    """
    Compute per-band statistics for VHR Google imagery.

    Layout: (6, H, W) = 3 bands (RGB) x 2 dates (start, end).
    Stats are computed per RGB band, pooled across both dates.
    """
    all_samples = []
    total_pixels = 0
    valid_pixels = 0

    for refid in refids:
        matches = list(folder.glob(f"{refid}*{suffix}"))
        if not matches:
            print(f"  WARNING: no VHR file for {refid[:30]}...")
            continue

        with rasterio.open(matches[0]) as src:
            data = src.read()  # (6, H, W)
            h, w = src.height, src.width

        if data.shape[0] != VHR_N_BANDS * VHR_N_DATES:
            print(f"  WARNING: {refid[:30]}... has {data.shape[0]} bands, expected {VHR_N_BANDS * VHR_N_DATES}")
            continue

        data = data.reshape(VHR_N_DATES, VHR_N_BANDS, h, w)

        for t in range(VHR_N_DATES):
            frame = data[t]  # (3, H, W)
            valid = _make_valid_mask(frame, is_uint8=True)

            total_pixels += h * w
            n_valid = int(valid.sum())
            valid_pixels += n_valid

            if n_valid == 0:
                continue

            pixels = frame[:, valid].T  # (n_valid, 3)
            if pixels.shape[0] > sample_pixels:
                idx = np.random.choice(pixels.shape[0], sample_pixels, replace=False)
                pixels = pixels[idx]

            all_samples.append(pixels.astype(np.float64))

    if not all_samples:
        raise ValueError("No valid VHR data found")

    stacked = np.concatenate(all_samples, axis=0)
    print(f"  VHR Google: {stacked.shape[0]:,} sample pixels from {len(refids)} tiles")

    stats = _clip_and_compute(stacked)
    stats["valid_pixel_fraction"] = valid_pixels / total_pixels if total_pixels > 0 else 0.0
    return stats


def write_part1_csv(stats: Dict, path: Path):
    """Write Part 1 CSV (Sentinel-2 only)."""
    df = pd.DataFrame({
        "band": S2_BAND_NAMES,
        "mean": stats["mean"],
        "std": stats["std"],
        "min": stats["min"],
        "max": stats["max"],
        "p2": stats["p2"],
        "p98": stats["p98"],
    })
    df.to_csv(path, index=False)
    print(f"  Saved: {path}")


def _modality_json(band_names, stats, unit, **extra):
    """Build one modality's JSON block."""
    block = {}
    if band_names:
        block["band_names"] = band_names
        block["n_spectral_bands"] = len(band_names)
    block.update(extra)
    block["mean"] = stats["mean"].tolist()
    block["std"] = stats["std"].tolist()
    block["p2"] = stats["p2"].tolist()
    block["p98"] = stats["p98"].tolist()
    block["valid_pixel_fraction"] = stats["valid_pixel_fraction"]
    block["unit"] = unit
    return block


def write_part2_json(s2, ps, ae, vhr, n_train, path: Path):
    """Write Part 2 JSON (multi-modal)."""
    out = {
        "computed_on": f"training set ({n_train} tiles), EPSG:3035, "
                       f"border-fill and NaN excluded, clipped to [p{CLIP_LOW}, p{CLIP_HIGH}]",
        "masking": "S2/AE: all-zero = nodata; PS/VHR: border-aware zero detection (preserves dark pixels)",
        "clipping": f"percentile [{CLIP_LOW}, {CLIP_HIGH}] applied per band before computing mean/std",
        "sentinel": _modality_json(
            S2_BAND_NAMES, s2, "TOA reflectance x 10000",
            n_timesteps=S2_N_TIMESTEPS,
        ),
        "planetscope": _modality_json(
            PS_BAND_NAMES, ps, "uint8 DN [0-255]",
            n_timesteps=PS_N_TIMESTEPS,
        ),
        "alphaearth": _modality_json(
            None, ae, "learned embeddings (float64)",
            n_features=AE_N_FEATURES, n_years=AE_N_YEARS,
        ),
        "vhr_google": _modality_json(
            VHR_BAND_NAMES, vhr, "uint8 DN [0-255]",
            n_dates=VHR_N_DATES,
        ),
    }

    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved: {path}")


def _print_band_stats(names, stats):
    for i, name in enumerate(names):
        print(f"  {name:>6s}: mean={stats['mean'][i]:.2f}, std={stats['std'][i]:.2f}, "
              f"p2={stats['p2'][i]:.1f}, p98={stats['p98'][i]:.1f}")
    print(f"  Valid pixel fraction: {stats['valid_pixel_fraction']:.1%}")


def main():
    np.random.seed(42)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    s2_dir = DATA_V2 / "Sentinel"
    ps_dir = DATA_V2 / "PlanetScope"
    ae_dir = DATA_V2 / "AlphaEarth"
    vhr_dir = DATA_V2 / "VHR_google"

    # ── Part 1: Sentinel-2 only ──
    print("=" * 60)
    print("PART 1: Sentinel-2 normalization")
    print("=" * 60)

    p1_train = load_refids(SPLITS_DIR / "part1" / "train_refids.txt")
    print(f"  Loaded {len(p1_train)} Part 1 training refids")

    s2_p1 = compute_spectral_stats(
        p1_train, s2_dir, "_RGBNIRRSWIRQ_Mosaic.tif",
        S2_N_BANDS, S2_N_TIMESTEPS, is_uint8=False,
    )
    write_part1_csv(s2_p1, OUTPUT_DIR / "normalization_stats_part1.csv")

    # ── Part 2: S2 + PS + AE + VHR ──
    print("\n" + "=" * 60)
    print("PART 2: Multi-modal normalization")
    print("=" * 60)

    p2_train = load_refids(SPLITS_DIR / "part2" / "train_refids.txt")
    print(f"  Loaded {len(p2_train)} Part 2 training refids")

    print("\n  Computing Sentinel-2 stats...")
    s2_p2 = compute_spectral_stats(
        p2_train, s2_dir, "_RGBNIRRSWIRQ_Mosaic.tif",
        S2_N_BANDS, S2_N_TIMESTEPS, is_uint8=False,
    )

    print("\n  Computing PlanetScope stats...")
    ps_p2 = compute_spectral_stats(
        p2_train, ps_dir, "_RGBQ_Mosaic.tif",
        PS_N_BANDS, PS_N_TIMESTEPS, is_uint8=True,
    )

    print("\n  Computing AlphaEarth stats...")
    ae_p2 = compute_ae_stats(p2_train, ae_dir)

    print("\n  Computing VHR Google stats...")
    vhr_p2 = compute_vhr_stats(p2_train, vhr_dir)

    write_part2_json(s2_p2, ps_p2, ae_p2, vhr_p2, len(p2_train),
                     OUTPUT_DIR / "normalization_stats_part2.json")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\nPart 1 (Sentinel-2, 9 bands):")
    _print_band_stats(S2_BAND_NAMES, s2_p1)

    print(f"\nPart 2 (Sentinel-2, 9 bands):")
    _print_band_stats(S2_BAND_NAMES, s2_p2)

    print(f"\nPart 2 (PlanetScope, 3 bands):")
    _print_band_stats(PS_BAND_NAMES, ps_p2)

    print(f"\nPart 2 (AlphaEarth, 64 features):")
    print(f"  mean range: [{ae_p2['mean'].min():.4f}, {ae_p2['mean'].max():.4f}]")
    print(f"  std  range: [{ae_p2['std'].min():.4f}, {ae_p2['std'].max():.4f}]")
    print(f"  Valid pixel fraction: {ae_p2['valid_pixel_fraction']:.1%}")

    print(f"\nPart 2 (VHR Google, 3 bands):")
    _print_band_stats(VHR_BAND_NAMES, vhr_p2)

    print("\nDone!")


if __name__ == "__main__":
    main()
