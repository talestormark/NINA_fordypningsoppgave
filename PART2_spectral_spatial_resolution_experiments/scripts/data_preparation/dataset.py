#!/usr/bin/env python3
"""
PyTorch Dataset for multi-modal land-take detection (Part II).

Supports 10 experiment configurations across 4 modalities (S2, PS, AE, masks)
with configuration-driven band selection, spectral indices, temporal difference
mode, and multi-modality fusion. Reads from data/processed/epsg3035_10m_v1/.

Matches Part I conventions: return format, augmentation, NaN handling, fold
stratification.
"""

import os
import json
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
import rasterio
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict
import albumentations as A
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_DEFAULT_DATA_DIR = "epsg3035_10m_v1"
PROCESSED_DIR = REPO_ROOT / "data" / "processed" / os.environ.get("P2_DATA_DIR", _DEFAULT_DATA_DIR)
SPLITS_CSV = REPO_ROOT / "outputs" / "splits" / "split_info.csv"
STATS_JSON = PROCESSED_DIR / "normalisation_stats.json"

MODALITY_DIRS = {
    "sentinel": PROCESSED_DIR / "sentinel",
    "planetscope": PROCESSED_DIR / "planetscope_10m",
    "alphaearth": PROCESSED_DIR / "alphaearth",
    "masks": PROCESSED_DIR / "masks",
}


def set_data_dir(data_dir: str):
    """Reconfigure PROCESSED_DIR and all dependent paths at runtime."""
    global PROCESSED_DIR, STATS_JSON, MODALITY_DIRS
    PROCESSED_DIR = REPO_ROOT / "data" / "processed" / data_dir
    STATS_JSON = PROCESSED_DIR / "normalisation_stats.json"
    MODALITY_DIRS = {
        "sentinel": PROCESSED_DIR / "sentinel",
        "planetscope": PROCESSED_DIR / "planetscope_10m",
        "alphaearth": PROCESSED_DIR / "alphaearth",
        "masks": PROCESSED_DIR / "masks",
    }

MODALITY_PATTERNS = {
    "sentinel": "{refid}_RGBNIRRSWIRQ_Mosaic.tif",
    "planetscope": "{refid}_RGBQ_Mosaic.tif",
    "alphaearth": "{refid}_VEY_Mosaic.tif",
    "masks": "{refid}_mask.tif",
}

S2_BAND_NAMES = ["blue", "green", "red", "R1", "R2", "R3", "nir", "swir1", "swir2"]
S2_N_BANDS = 9
S2_N_TIMESTEPS = 14  # 7 years x 2 quarters

PS_BAND_NAMES = ["blue", "green", "red"]
PS_N_BANDS = 3
PS_N_TIMESTEPS = 14

AE_N_FEATURES = 64
AE_N_YEARS = 7  # already annual

N_YEARS = 7
N_QUARTERS = 2
CROP_SIZE = 64

# ---------------------------------------------------------------------------
# Experiment configurations
# ---------------------------------------------------------------------------

EXPERIMENT_CONFIGS = {
    "A1_s2_rgb": {
        "modalities": [
            {"source": "sentinel", "band_indices": [0, 1, 2], "stats_key": "sentinel"},
        ],
        "compute_indices": False,
        "indices_only": False,
        "temporal_diff": False,
        "expected_shape": (7, 3, CROP_SIZE, CROP_SIZE),
    },
    "A2_s2_rgbnir": {
        "modalities": [
            {"source": "sentinel", "band_indices": [0, 1, 2, 6], "stats_key": "sentinel"},
        ],
        "compute_indices": False,
        "indices_only": False,
        "temporal_diff": False,
        "expected_shape": (7, 4, CROP_SIZE, CROP_SIZE),
    },
    "A3_s2_9band": {
        "modalities": [
            {"source": "sentinel", "band_indices": list(range(9)), "stats_key": "sentinel"},
        ],
        "compute_indices": False,
        "indices_only": False,
        "temporal_diff": False,
        "expected_shape": (7, 9, CROP_SIZE, CROP_SIZE),
    },
    "A4_s2_indices": {
        "modalities": [
            {"source": "sentinel", "band_indices": list(range(9)), "stats_key": "sentinel"},
        ],
        "compute_indices": True,
        "indices_only": False,
        "temporal_diff": False,
        "expected_shape": (7, 13, CROP_SIZE, CROP_SIZE),
    },
    "A5_indices_only": {
        "modalities": [
            {"source": "sentinel", "band_indices": list(range(9)), "stats_key": "sentinel"},
        ],
        "compute_indices": True,
        "indices_only": True,
        "temporal_diff": False,
        "expected_shape": (7, 4, CROP_SIZE, CROP_SIZE),
    },
    "A6_temporal_diff": {
        "modalities": [
            {"source": "sentinel", "band_indices": list(range(9)), "stats_key": "temporal_diff"},
        ],
        "compute_indices": False,
        "indices_only": False,
        "temporal_diff": True,
        "expected_shape": (1, 9, CROP_SIZE, CROP_SIZE),
    },
    "C2_ps_rgb": {
        "modalities": [
            {"source": "planetscope", "band_indices": [0, 1, 2], "stats_key": "planetscope"},
        ],
        "compute_indices": False,
        "indices_only": False,
        "temporal_diff": False,
        "expected_shape": (7, 3, CROP_SIZE, CROP_SIZE),
    },
    "C3_s2_ps_fusion": {
        "modalities": [
            {"source": "sentinel", "band_indices": [0, 1, 2], "stats_key": "sentinel"},
            {"source": "planetscope", "band_indices": [0, 1, 2], "stats_key": "planetscope"},
        ],
        "compute_indices": False,
        "indices_only": False,
        "temporal_diff": False,
        "expected_shape": (7, 6, CROP_SIZE, CROP_SIZE),
    },
    "D2_alphaearth": {
        "modalities": [
            {"source": "alphaearth", "band_indices": list(range(64)), "stats_key": "alphaearth"},
        ],
        "compute_indices": False,
        "indices_only": False,
        "temporal_diff": False,
        "expected_shape": (7, 64, CROP_SIZE, CROP_SIZE),
    },
    "D3_s2_ae_fusion": {
        "modalities": [
            {"source": "sentinel", "band_indices": list(range(9)), "stats_key": "sentinel"},
            {"source": "alphaearth", "band_indices": list(range(64)), "stats_key": "alphaearth"},
        ],
        "compute_indices": False,
        "indices_only": False,
        "temporal_diff": False,
        "expected_shape": (7, 73, CROP_SIZE, CROP_SIZE),
    },
}

# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------


def load_normalisation_stats(path: Path = None) -> dict:
    """Load pre-computed normalisation stats from JSON."""
    path = path or STATS_JSON
    with open(path) as f:
        raw = json.load(f)
    stats = {}
    for key in ("sentinel", "planetscope", "alphaearth"):
        if key in raw:
            stats[key] = {
                "mean": np.array(raw[key]["mean"], dtype=np.float64),
                "std": np.array(raw[key]["std"], dtype=np.float64),
            }
    # indices and temporal_diff are optional (added by compute_normalisation_stats)
    for key in ("indices", "temporal_diff"):
        if key in raw:
            stats[key] = {
                "mean": np.array(raw[key]["mean"], dtype=np.float64),
                "std": np.array(raw[key]["std"], dtype=np.float64),
            }
    return stats


def compute_normalisation_stats(
    train_refids: List[str],
    sample_pixels: int = 10000,
) -> dict:
    """
    Compute per-fold z-score stats for all modalities, indices, and temporal_diff.

    Border-fill pixels (all bands == 0) and NaN pixels are excluded.
    Samples up to `sample_pixels` per tile for speed.

    Returns dict with keys: sentinel, planetscope, alphaearth, indices, temporal_diff.
    Each value is a dict with 'mean' and 'std' arrays.
    """
    rng = np.random.RandomState(42)

    # Accumulators: list-of-lists per band
    accum = {
        "sentinel": [[] for _ in range(S2_N_BANDS)],
        "planetscope": [[] for _ in range(PS_N_BANDS)],
        "alphaearth": [[] for _ in range(AE_N_FEATURES)],
        "indices": [[] for _ in range(4)],  # NDVI, NDBI, BSI, NDWI
        "temporal_diff": [[] for _ in range(S2_N_BANDS)],
    }

    for refid in tqdm(train_refids, desc="Computing normalisation stats", leave=False):
        # --- Sentinel ---
        s2_path = MODALITY_DIRS["sentinel"] / MODALITY_PATTERNS["sentinel"].format(refid=refid)
        s2_composites = None
        if s2_path.exists():
            with rasterio.open(s2_path) as src:
                raw = src.read().astype(np.float64)  # (126, H, W)
            s2_all = raw.reshape(S2_N_TIMESTEPS, S2_N_BANDS, raw.shape[1], raw.shape[2])
            s2_composites = _compose_annual(s2_all)  # (7, 9, H, W)
            _accumulate(accum["sentinel"], s2_composites, rng, sample_pixels)

            # Indices from composites
            indices = _compute_spectral_indices(s2_composites)  # (7, 4, H, W)
            _accumulate(accum["indices"], indices, rng, sample_pixels)

            # Temporal diff: mean(2022-2024) - mean(2018-2020)
            late = np.nanmean(s2_composites[4:7], axis=0)   # years 2022,2023,2024
            early = np.nanmean(s2_composites[0:3], axis=0)   # years 2018,2019,2020
            diff = late - early  # (9, H, W)
            diff_4d = diff[np.newaxis, ...]  # (1, 9, H, W)
            _accumulate(accum["temporal_diff"], diff_4d, rng, sample_pixels)

        # --- PlanetScope ---
        ps_path = MODALITY_DIRS["planetscope"] / MODALITY_PATTERNS["planetscope"].format(refid=refid)
        if ps_path.exists():
            with rasterio.open(ps_path) as src:
                raw = src.read().astype(np.float64)  # (42, H, W)
            ps_all = raw.reshape(PS_N_TIMESTEPS, PS_N_BANDS, raw.shape[1], raw.shape[2])
            ps_composites = _compose_annual(ps_all)  # (7, 3, H, W)
            _accumulate(accum["planetscope"], ps_composites, rng, sample_pixels)

        # --- AlphaEarth ---
        ae_path = MODALITY_DIRS["alphaearth"] / MODALITY_PATTERNS["alphaearth"].format(refid=refid)
        if ae_path.exists():
            with rasterio.open(ae_path) as src:
                raw = src.read().astype(np.float64)  # (448, H, W)
            ae_all = raw.reshape(AE_N_YEARS, AE_N_FEATURES, raw.shape[1], raw.shape[2])
            _accumulate(accum["alphaearth"], ae_all, rng, sample_pixels)

    # Aggregate
    stats = {}
    for key, band_lists in accum.items():
        n_bands = len(band_lists)
        means = np.zeros(n_bands, dtype=np.float64)
        stds = np.zeros(n_bands, dtype=np.float64)
        for b in range(n_bands):
            if band_lists[b]:
                vals = np.concatenate(band_lists[b])
                means[b] = np.nanmean(vals)
                stds[b] = np.nanstd(vals)
        stds[stds == 0] = 1.0
        stats[key] = {"mean": means, "std": stds}

    return stats


def _accumulate(
    band_lists: list,
    data: np.ndarray,
    rng: np.random.RandomState,
    sample_pixels: int,
):
    """Accumulate valid pixel samples from data (T, C, H, W) into band_lists."""
    _, C, H, W = data.shape
    # Valid mask: exclude border-fill (all bands 0 across all timesteps) and NaN
    all_zero = np.all(data == 0, axis=(0, 1))  # (H, W)
    any_nan = np.any(np.isnan(data), axis=(0, 1))  # (H, W)
    valid = ~all_zero & ~any_nan  # (H, W)
    valid_indices = np.where(valid.ravel())[0]

    if len(valid_indices) == 0:
        return

    if len(valid_indices) > sample_pixels:
        chosen = rng.choice(valid_indices, size=sample_pixels, replace=False)
    else:
        chosen = valid_indices

    h_idx = chosen // W
    w_idx = chosen % W

    for b in range(C):
        vals = data[:, b, h_idx, w_idx].flatten()  # (T * n_chosen,)
        band_lists[b].append(vals)


# ---------------------------------------------------------------------------
# Annual compositing (matches Part I NaN-aware Q2+Q3 averaging)
# ---------------------------------------------------------------------------


def _compose_annual(data: np.ndarray) -> np.ndarray:
    """
    Reduce 14 quarterly timesteps to 7 annual composites.

    Uses Q2+Q3 averaging with NaN-aware fallback from Part I:
    - If one quarter has >50% NaN and the other <20%, use the better one
    - Otherwise average both

    Args:
        data: (14, C, H, W) quarterly data

    Returns:
        (7, C, H, W) annual composites
    """
    composites = []
    for year_idx in range(N_YEARS):
        q2_idx = year_idx * 2
        q3_idx = year_idx * 2 + 1

        q2 = data[q2_idx]  # (C, H, W)
        q3 = data[q3_idx]

        q2_nan_pct = np.isnan(q2).sum() / q2.size * 100
        q3_nan_pct = np.isnan(q3).sum() / q3.size * 100

        if q2_nan_pct > 50 and q3_nan_pct < 20:
            composites.append(q3)
        elif q3_nan_pct > 50 and q2_nan_pct < 20:
            composites.append(q2)
        else:
            composites.append((q2 + q3) / 2.0)

    return np.stack(composites, axis=0)


# ---------------------------------------------------------------------------
# Spectral indices
# ---------------------------------------------------------------------------


def _safe_ratio(num: np.ndarray, denom: np.ndarray) -> np.ndarray:
    """Division-by-zero-safe normalised ratio."""
    return np.where(denom == 0, 0.0, num / denom)


def _compute_spectral_indices(s2_composites: np.ndarray) -> np.ndarray:
    """
    Compute NDVI, NDBI, BSI, NDWI from raw S2 annual composites.

    Input:  (7, 9, H, W) — raw unnormalised S2 composites
    Output: (7, 4, H, W) — [NDVI, NDBI, BSI, NDWI]

    Band indices: blue=0, green=1, red=2, R1=3, R2=4, R3=5, nir=6, swir1=7, swir2=8
    """
    blue = s2_composites[:, 0]   # (7, H, W)
    green = s2_composites[:, 1]
    red = s2_composites[:, 2]
    nir = s2_composites[:, 6]
    swir1 = s2_composites[:, 7]

    ndvi = _safe_ratio(nir - red, nir + red)
    ndbi = _safe_ratio(swir1 - nir, swir1 + nir)
    bsi = _safe_ratio((swir1 + red) - (nir + blue), (swir1 + red) + (nir + blue))
    ndwi = _safe_ratio(green - nir, green + nir)

    return np.stack([ndvi, ndbi, bsi, ndwi], axis=1)  # (7, 4, H, W)


# ---------------------------------------------------------------------------
# Temporal difference
# ---------------------------------------------------------------------------


def _compute_temporal_diff(composites: np.ndarray) -> np.ndarray:
    """
    Compute temporal difference: mean(2022-2024) - mean(2018-2020).

    Input:  (7, C, H, W) — annual composites (years 2018-2024)
    Output: (1, C, H, W)
    """
    late = np.nanmean(composites[4:7], axis=0)   # years 2022, 2023, 2024
    early = np.nanmean(composites[0:3], axis=0)   # years 2018, 2019, 2020
    diff = late - early  # (C, H, W)
    return diff[np.newaxis, ...]  # (1, C, H, W)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class MultiModalDataset(Dataset):
    """
    Multi-modal dataset for Part II experiments.

    Parameterised by an experiment config dict — no subclasses needed.

    Args:
        refids: List of tile reference IDs
        experiment: Experiment config key (e.g. 'A3_s2_9band')
        norm_stats: Dict of {modality_key: {mean, std}} arrays.
        transform: Albumentations spatial transform
    """

    def __init__(
        self,
        refids: List[str],
        experiment: str,
        norm_stats: dict,
        transform: Optional[A.Compose] = None,
    ):
        self.refids = refids
        self.config = EXPERIMENT_CONFIGS[experiment]
        self.transform = transform

        # Pre-slice normalisation arrays per component
        self.norm_slices = []
        for mod_spec in self.config["modalities"]:
            key = mod_spec["stats_key"]
            bands = mod_spec["band_indices"]
            source = mod_spec["source"]
            if key in norm_stats:
                m = norm_stats[key]["mean"]
                s = norm_stats[key]["std"]
                # For sentinel/planetscope, stats are per spectral band;
                # band_indices selects which bands we use
                if source in ("sentinel", "planetscope"):
                    m = m[bands]
                    s = s[bands]
                # For alphaearth, stats are per feature — band_indices selects features
                elif source == "alphaearth":
                    m = m[bands]
                    s = s[bands]
                # For temporal_diff, stats are per S2 band (9 bands)
                # band_indices from the sentinel spec selects which ones
                self.norm_slices.append({"mean": m, "std": s})
            else:
                raise ValueError(f"Missing normalisation stats for key '{key}'")

        # If indices are used, prepare their norm stats
        if self.config["compute_indices"]:
            if "indices" in norm_stats:
                self.indices_norm = norm_stats["indices"]
            else:
                raise ValueError("Missing normalisation stats for 'indices'")

        self._verify_files()

    def _verify_files(self):
        """Check that all required modality files exist for every refid."""
        sources_needed = set()
        for mod_spec in self.config["modalities"]:
            sources_needed.add(mod_spec["source"])
        sources_needed.add("masks")

        missing = []
        for refid in self.refids:
            for src in sources_needed:
                p = MODALITY_DIRS[src] / MODALITY_PATTERNS[src].format(refid=refid)
                if not p.exists():
                    missing.append((refid, src, str(p)))

        if missing:
            examples = missing[:5]
            raise FileNotFoundError(
                f"Missing {len(missing)} file(s). Examples: {examples}"
            )

    def __len__(self) -> int:
        return len(self.refids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        refid = self.refids[idx]
        cfg = self.config

        # ------------------------------------------------------------------
        # 1. Load and composite each modality
        # ------------------------------------------------------------------
        components = []       # list of (T, C, H, W) arrays
        s2_composites = None  # cached for index computation

        for mod_spec in cfg["modalities"]:
            source = mod_spec["source"]
            bands = mod_spec["band_indices"]
            path = MODALITY_DIRS[source] / MODALITY_PATTERNS[source].format(refid=refid)

            with rasterio.open(path) as src:
                raw = src.read().astype(np.float64)  # (total_bands, H, W)

            if source == "sentinel":
                all_ts = raw.reshape(S2_N_TIMESTEPS, S2_N_BANDS, raw.shape[1], raw.shape[2])
                composites = _compose_annual(all_ts)  # (7, 9, H, W)
                # Cache full composites if indices needed
                if cfg["compute_indices"]:
                    s2_composites = composites
                selected = composites[:, bands, :, :]  # (7, C_sel, H, W)

            elif source == "planetscope":
                all_ts = raw.reshape(PS_N_TIMESTEPS, PS_N_BANDS, raw.shape[1], raw.shape[2])
                composites = _compose_annual(all_ts)  # (7, 3, H, W)
                selected = composites[:, bands, :, :]

            elif source == "alphaearth":
                # Already annual: (448, H, W) -> (7, 64, H, W)
                all_ts = raw.reshape(AE_N_YEARS, AE_N_FEATURES, raw.shape[1], raw.shape[2])
                selected = all_ts[:, bands, :, :]

            else:
                raise ValueError(f"Unknown source: {source}")

            components.append(selected)

        # ------------------------------------------------------------------
        # 2. Spectral indices (from raw unnormalised S2 composites)
        # ------------------------------------------------------------------
        if cfg["compute_indices"]:
            if s2_composites is None:
                raise RuntimeError("compute_indices requires sentinel source")
            indices = _compute_spectral_indices(s2_composites)  # (7, 4, H, W)
            if cfg["indices_only"]:
                components = [indices]
            else:
                components.append(indices)

        # ------------------------------------------------------------------
        # 3. Temporal difference
        # ------------------------------------------------------------------
        if cfg["temporal_diff"]:
            components = [_compute_temporal_diff(c) for c in components]

        # ------------------------------------------------------------------
        # 4. Pad if tile < 64px in any dimension
        # ------------------------------------------------------------------
        _, _, H, W = components[0].shape
        pad_h = max(0, CROP_SIZE - H)
        pad_w = max(0, CROP_SIZE - W)

        if pad_h > 0 or pad_w > 0:
            components = [
                np.pad(c, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
                for c in components
            ]
            H += pad_h
            W += pad_w

        # Load mask
        mask_path = MODALITY_DIRS["masks"] / MODALITY_PATTERNS["masks"].format(refid=refid)
        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.float32)  # (H_mask, W_mask)
        mask = (mask > 0).astype(np.float32)

        # Resize mask if needed (should match after reprojection, but safety)
        if mask.shape != (H - pad_h, W - pad_w) and mask.shape != (H, W):
            from scipy.ndimage import zoom
            target_h = H - pad_h
            target_w = W - pad_w
            zoom_factors = (target_h / mask.shape[0], target_w / mask.shape[1])
            mask = zoom(mask, zoom_factors, order=0)

        if pad_h > 0 or pad_w > 0:
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)

        # ------------------------------------------------------------------
        # 5. Concatenate components along C -> (T, C_total, H, W)
        # ------------------------------------------------------------------
        image = np.concatenate(components, axis=1)  # (T, C_total, H, W)
        T, C_total, H, W = image.shape

        # ------------------------------------------------------------------
        # 6. Augment: reshape to (H, W, T*C) for albumentations
        # ------------------------------------------------------------------
        if self.transform is not None:
            img_hwc = image.transpose(2, 3, 0, 1).reshape(H, W, -1)  # (H, W, T*C)
            augmented = self.transform(image=img_hwc, mask=mask)
            img_aug = augmented["image"]
            mask = augmented["mask"]
            H_new, W_new = img_aug.shape[:2]
            image = img_aug.reshape(H_new, W_new, T, C_total).transpose(2, 3, 0, 1)
        else:
            image = image.astype(np.float32)
            mask = mask.astype(np.float32)

        # ------------------------------------------------------------------
        # 7. Normalise per component using pre-sliced mean/std
        # ------------------------------------------------------------------
        ch_offset = 0
        norm_specs = list(self.norm_slices)
        if cfg["compute_indices"]:
            if cfg["indices_only"]:
                norm_specs = [self.indices_norm]
            else:
                norm_specs.append(self.indices_norm)

        for ns in norm_specs:
            n_ch = len(ns["mean"])
            mean = ns["mean"].reshape(1, n_ch, 1, 1).astype(np.float32)
            std = ns["std"].reshape(1, n_ch, 1, 1).astype(np.float32)
            image[:, ch_offset:ch_offset + n_ch] = (
                (image[:, ch_offset:ch_offset + n_ch] - mean) / std
            )
            ch_offset += n_ch

        # ------------------------------------------------------------------
        # 8. NaN -> 0
        # ------------------------------------------------------------------
        np.nan_to_num(image, copy=False, nan=0.0)

        # ------------------------------------------------------------------
        # 9. Return
        # ------------------------------------------------------------------
        return {
            "image": torch.from_numpy(image).float(),   # (T, C, H, W)
            "mask": torch.from_numpy(mask).float(),      # (H, W)
            "refid": refid,
        }


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------


def get_transform(is_train: bool = True, image_size: int = CROP_SIZE) -> A.Compose:
    if is_train:
        return A.Compose([
            A.RandomCrop(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ])
    else:
        return A.Compose([
            A.CenterCrop(image_size, image_size),
        ])


# ---------------------------------------------------------------------------
# Dataloaders
# ---------------------------------------------------------------------------


def get_dataloaders(
    experiment: str,
    batch_size: int = 4,
    num_workers: int = 4,
    image_size: int = CROP_SIZE,
    fold: int = None,
    num_folds: int = 5,
    seed: int = 42,
    use_precomputed_stats: bool = True,
    data_dir: str = None,
) -> Dict[str, DataLoader]:
    """
    Create train/val/test DataLoaders for a given experiment.

    Args:
        experiment: Key into EXPERIMENT_CONFIGS
        batch_size: Batch size
        num_workers: DataLoader workers
        image_size: Crop size in pixels
        fold: Fold index (0..num_folds-1). None = original split.
        num_folds: Number of CV folds
        seed: Random seed for fold generation
        use_precomputed_stats: If True, load stats from JSON; else compute from fold
        data_dir: Data directory name under data/processed/ (overrides P2_DATA_DIR env var)

    Returns:
        Dict with 'train', 'val', 'test' DataLoaders
    """
    if data_dir is not None:
        set_data_dir(data_dir)

    if experiment not in EXPERIMENT_CONFIGS:
        raise ValueError(f"Unknown experiment: {experiment}")

    # Load splits from CSV
    df = pd.read_csv(SPLITS_CSV)
    train_refids_orig = df[df["split"] == "train"]["refid"].tolist()
    val_refids_orig = df[df["split"] == "val"]["refid"].tolist()
    test_refids = df[df["split"] == "test"]["refid"].tolist()

    # K-fold cross-validation
    if fold is not None:
        from sklearn.model_selection import StratifiedKFold

        trainval_refids = train_refids_orig + val_refids_orig
        refid_to_level = dict(zip(df["refid"], df["change_level"]))
        change_levels = [refid_to_level[r] for r in trainval_refids]

        skfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        splits = list(skfold.split(trainval_refids, change_levels))

        if fold < 0 or fold >= num_folds:
            raise ValueError(f"fold must be in [0, {num_folds - 1}], got {fold}")

        train_idx, val_idx = splits[fold]
        train_refids = [trainval_refids[i] for i in train_idx]
        val_refids = [trainval_refids[i] for i in val_idx]

        train_levels = [change_levels[i] for i in train_idx]
        val_levels = [change_levels[i] for i in val_idx]
        print(f"\nStratified K-Fold CV: fold {fold}/{num_folds - 1}")
        print(f"  Train: {len(train_refids)} (low:{train_levels.count('low')} "
              f"mod:{train_levels.count('moderate')} high:{train_levels.count('high')})")
        print(f"  Val:   {len(val_refids)} (low:{val_levels.count('low')} "
              f"mod:{val_levels.count('moderate')} high:{val_levels.count('high')})")
    else:
        train_refids = train_refids_orig
        val_refids = val_refids_orig
        print(f"\nUsing original train/val split")
        print(f"  Train: {len(train_refids)}, Val: {len(val_refids)}")

    print(f"  Test:  {len(test_refids)} (held out)")
    print(f"  Experiment: {experiment}")

    # Normalisation stats
    if use_precomputed_stats:
        print("  Loading pre-computed normalisation stats...")
        norm_stats = load_normalisation_stats()
        # Check if indices/temporal_diff stats need to be computed
        cfg = EXPERIMENT_CONFIGS[experiment]
        needs_indices = cfg["compute_indices"] and "indices" not in norm_stats
        needs_tdiff = cfg["temporal_diff"] and "temporal_diff" not in norm_stats
        if needs_indices or needs_tdiff:
            print("  Computing missing index/temporal_diff stats from training set...")
            computed = compute_normalisation_stats(train_refids)
            if needs_indices:
                norm_stats["indices"] = computed["indices"]
            if needs_tdiff:
                norm_stats["temporal_diff"] = computed["temporal_diff"]
    else:
        print(f"  Computing normalisation stats from {len(train_refids)} training tiles...")
        norm_stats = compute_normalisation_stats(train_refids)

    print(f"  Stats keys available: {list(norm_stats.keys())}")

    # Create datasets
    train_ds = MultiModalDataset(
        refids=train_refids,
        experiment=experiment,
        norm_stats=norm_stats,
        transform=get_transform(is_train=True, image_size=image_size),
    )
    val_ds = MultiModalDataset(
        refids=val_refids,
        experiment=experiment,
        norm_stats=norm_stats,
        transform=get_transform(is_train=False, image_size=image_size),
    )
    test_ds = MultiModalDataset(
        refids=test_refids,
        experiment=experiment,
        norm_stats=norm_stats,
        transform=get_transform(is_train=False, image_size=image_size),
    )

    # Create DataLoaders
    loader_kwargs = dict(num_workers=num_workers, pin_memory=True)
    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, **loader_kwargs),
    }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def _smoke_test(experiment: str):
    """Run a single-experiment smoke test."""
    cfg = EXPERIMENT_CONFIGS[experiment]
    expected = cfg["expected_shape"]

    loaders = get_dataloaders(
        experiment=experiment,
        batch_size=2,
        num_workers=0,
        image_size=CROP_SIZE,
        fold=0,
    )

    batch = next(iter(loaders["train"]))
    img = batch["image"]
    mask = batch["mask"]

    actual = tuple(img.shape[1:])  # drop batch dim
    print(f"\n  [{experiment}]")
    print(f"    Image shape: {tuple(img.shape)} (expected batch+{expected})")
    print(f"    Mask shape:  {tuple(mask.shape)}")
    print(f"    Image stats: mean={img.mean():.3f}, std={img.std():.3f}, "
          f"min={img.min():.3f}, max={img.max():.3f}")
    print(f"    NaN count:   {torch.isnan(img).sum().item()}")
    print(f"    Mask unique: {torch.unique(mask).tolist()}")

    assert actual == expected, f"Shape mismatch: {actual} != {expected}"
    assert torch.isnan(img).sum() == 0, "NaN in output!"
    print(f"    PASSED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-modal dataset smoke test")
    parser.add_argument("--experiment", type=str, default=None,
                        help="Experiment key (e.g. A3_s2_9band). Omit for all.")
    parser.add_argument("--all", action="store_true", help="Test all experiments")
    args = parser.parse_args()

    if args.all or args.experiment is None:
        experiments = list(EXPERIMENT_CONFIGS.keys())
    else:
        experiments = [args.experiment]

    print(f"Smoke testing {len(experiments)} experiment(s)...")
    for exp in experiments:
        _smoke_test(exp)

    print(f"\nAll {len(experiments)} experiment(s) passed!")

    