#!/usr/bin/env python3
"""
Configuration for multi-temporal land-take detection experiments.

This config extends the baseline config.py with settings specific to
multi-temporal Sentinel-2 modeling.
"""

from pathlib import Path
import sys

# Add parent directory to path to import base config
parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

try:
    from config import *  # Import all baseline config settings
except ImportError:
    print("Warning: Could not import base config.py")

# Define BASE_DIR (base repository directory)
BASE_DIR = Path(__file__).resolve().parent.parent

# Define DATA_DIR if not already defined from base config
if 'DATA_DIR' not in globals():
    DATA_DIR = BASE_DIR / "data" / "raw"

# Define reference ID list file (GeoJSON with tile metadata)
REFID_LIST_FILE = BASE_DIR / "land_take_bboxes_650m_v1.geojson"

# ============================================================================
# MULTI-TEMPORAL EXPERIMENT DIRECTORIES
# ============================================================================

MULTITEMPORAL_DIR = BASE_DIR / "PART1_multi_temporal_experiments"

# Script directories
MT_SCRIPTS_DIR = MULTITEMPORAL_DIR / "scripts"
MT_DATA_PREP_DIR = MT_SCRIPTS_DIR / "data_preparation"
MT_MODELING_DIR = MT_SCRIPTS_DIR / "modeling"
MT_EVALUATION_DIR = MT_SCRIPTS_DIR / "evaluation"
MT_ANALYSIS_DIR = MT_SCRIPTS_DIR / "analysis"
MT_SLURM_DIR = MT_SCRIPTS_DIR / "slurm"

# Output directories
MT_OUTPUTS_DIR = MULTITEMPORAL_DIR / "outputs"
MT_EXPERIMENTS_DIR = MT_OUTPUTS_DIR / "experiments"
MT_REPORTS_DIR = MT_OUTPUTS_DIR / "reports"
MT_FIGURES_DIR = MT_OUTPUTS_DIR / "figures"
MT_LOGS_DIR = MT_OUTPUTS_DIR / "logs"

# Documentation directories
MT_DOCS_DIR = MULTITEMPORAL_DIR / "docs"
MT_NOTEBOOKS_DIR = MULTITEMPORAL_DIR / "notebooks"

# ============================================================================
# DATA CONFIGURATION - SENTINEL-2
# ============================================================================

# Sentinel-2 file patterns (from base config if available, else define)
try:
    SENTINEL2_DIR = DATA_DIR / "Sentinel"
    SENTINEL2_PATTERN = "{refid}_RGBNIRRSWIRQ_Mosaic.tif"
except:
    pass

# Sentinel-2 band configuration
SENTINEL2_BANDS = ["blue", "green", "red", "R1", "R2", "R3", "nir", "swir1", "swir2"]
SENTINEL2_NUM_BANDS = len(SENTINEL2_BANDS)  # 9 spectral bands

# Temporal configuration
YEARS = list(range(2018, 2025))  # [2018, 2019, 2020, 2021, 2022, 2023, 2024]
QUARTERS = [2, 3]  # Q2 (Apr-Jun) and Q3 (Jul-Sep)

# Temporal sampling modes
TEMPORAL_SAMPLING_MODES = {
    "bi_temporal": {
        "num_steps": 2,
        "years": [2018, 2024],
        "quarters": [2],  # Use Q2 only
        "description": "Start and end years only"
    },
    "annual": {
        "num_steps": 7,
        "years": YEARS,
        "quarters": [2, 3],  # Average Q2+Q3 per year
        "description": "One composite per year (Q2+Q3 average)"
    },
    "quarterly": {
        "num_steps": 14,
        "years": YEARS,
        "quarters": QUARTERS,
        "description": "Q2 and Q3 for each year (full temporal density)"
    },
}

# Default temporal sampling mode
DEFAULT_TEMPORAL_SAMPLING = "annual"

# ============================================================================
# MODEL CONFIGURATION - MULTI-TEMPORAL ARCHITECTURES
# ============================================================================

# Available multi-temporal models
MULTITEMPORAL_MODELS = [
    "lstm_unet",        # LSTM over time → 2D U-Net decoder
    "unet_3d",          # 3D U-Net (spatiotemporal convolutions)
    "hybrid_lstm_3d",   # LSTM → 3D decoder
]

# Model-specific configurations
MODEL_CONFIGS = {
    "lstm_unet": {
        "input_format": "BTCHW",  # (Batch, Time, Channels, Height, Width)
        "lstm_hidden_dim": 256,
        "lstm_num_layers": 2,
        "lstm_bidirectional": False,
        "decoder_name": "unet",
        "expected_memory_gb": 24,
    },
    "unet_3d": {
        "input_format": "BCTHW",  # (Batch, Channels, Time, Height, Width)
        "conv_3d": True,
        "expected_memory_gb": 48,
    },
    "hybrid_lstm_3d": {
        "input_format": "BTCHW",
        "lstm_hidden_dim": 256,
        "lstm_num_layers": 2,
        "decoder_3d": True,
        "expected_memory_gb": 40,
    },
}

# ============================================================================
# PREPROCESSING CONFIGURATION
# ============================================================================

# Normalization strategy
NORMALIZATION_MODE = "zscore"  # Options: "zscore", "minmax", "per_tile"

# Normalization statistics file (computed from training set)
NORMALIZATION_STATS_FILE = MT_REPORTS_DIR / "normalization_stats.csv"

# Expected Sentinel-2 value ranges (for validation)
SENTINEL2_MIN_VALUE = 0
SENTINEL2_MAX_VALUE = 10000  # Top-of-atmosphere reflectance × 10000

# NoData value
SENTINEL2_NODATA = 0

# Cloud/NoData thresholds
MAX_NODATA_PERCENT = 5.0  # Reject quarters with >5% NoData
MAX_CLOUD_PERCENT = 20.0  # Warn if >20% cloudy pixels

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

# Default hyperparameters for multi-temporal models
MT_DEFAULT_HYPERPARAMS = {
    "batch_size": 4,  # May need to reduce for 3D models
    "image_size": 512,
    "epochs": 200,
    "learning_rate": 0.01,
    "optimizer": "sgd",
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "loss": "focal",
    "focal_alpha": 0.25,
    "focal_gamma": 2.0,
    "num_workers": 4,
    "mixed_precision": True,  # Use FP16 to save memory
    "gradient_checkpointing": False,  # Enable if OOM
}

# Seeds for reproducibility (same as baseline)
RANDOM_SEEDS = [42, 123, 456]

# ============================================================================
# EXPERIMENT TRACKING
# ============================================================================

# Experiment naming format
EXPERIMENT_NAME_FORMAT = "exp{id:03d}_{model}_{sampling}_seed{seed}"

# Example: exp001_lstm_annual_seed42

# Wandb configuration
WANDB_PROJECT = "landtake-multitemporal"
WANDB_ENTITY = "NINA_Fordypningsoppgave"  # Update with your wandb username

# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================

# Metrics to track
EVALUATION_METRICS = [
    "iou",           # Intersection over Union
    "f1",            # F1-Score
    "precision",     # Precision
    "recall",        # Recall
    "accuracy",      # Overall accuracy (less important due to imbalance)
]

# Stratified evaluation (by change level)
CHANGE_LEVEL_BINS = {
    "low": (0, 5),        # 0-5% change
    "moderate": (5, 30),  # 5-30% change
    "high": (30, 100),    # ≥30% change
}

# ============================================================================
# COMPUTATIONAL RESOURCES
# ============================================================================

# SLURM configuration for multi-temporal experiments
SLURM_CONFIG = {
    "account": "share-ie-idi",
    "partition": "GPUQ",
    "gres": "gpu:1",
    "constraint": "gpu80g",  # Request 80GB GPU for 3D models
    "mem": "128G",           # 128GB RAM recommended
    "time": "0-12:00:00",    # 12 hours max per job
    "mail_user": "tmstorma@stud.ntnu.no",
    "mail_type": "ALL",
}

# Memory profiling settings
PROFILE_MEMORY = False  # Set True to profile GPU memory usage
PROFILE_BATCH_SIZES = [1, 2, 4, 8]  # Test these batch sizes during profiling

# ============================================================================
# DATA VALIDATION
# ============================================================================

# Validation report files
TEMPORAL_QUALITY_REPORT = MT_REPORTS_DIR / "sentinel2_temporal_quality.csv"
TEMPORAL_SUMMARY_REPORT = MT_REPORTS_DIR / "sentinel2_temporal_summary.txt"

# Quarters to validate
VALIDATE_ALL_QUARTERS = True  # Check all 14 quarters per tile

# ============================================================================
# VISUALIZATION
# ============================================================================

# Figure settings
FIGURE_DPI = 300
FIGURE_FORMAT = "png"
FIGURE_SIZE = (12, 8)

# Temporal visualization settings
TEMPORAL_VIZ_BANDS = ["red", "green", "blue"]  # RGB for visualization
TEMPORAL_VIZ_TILES = 3  # Number of example tiles to visualize

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_experiment_dir(exp_id: str, seed: int = None) -> Path:
    """
    Get experiment output directory.

    Args:
        exp_id: Experiment ID (e.g., "exp001_lstm_annual")
        seed: Random seed (optional)

    Returns:
        Path to experiment directory
    """
    if seed is not None:
        exp_dir = MT_EXPERIMENTS_DIR / f"{exp_id}_seed{seed}"
    else:
        exp_dir = MT_EXPERIMENTS_DIR / exp_id

    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def get_temporal_sampling_config(mode: str) -> dict:
    """
    Get configuration for temporal sampling mode.

    Args:
        mode: Sampling mode ("bi_temporal", "annual", "quarterly")

    Returns:
        Dictionary with sampling configuration
    """
    if mode not in TEMPORAL_SAMPLING_MODES:
        raise ValueError(f"Unknown sampling mode: {mode}. "
                        f"Available: {list(TEMPORAL_SAMPLING_MODES.keys())}")

    return TEMPORAL_SAMPLING_MODES[mode]


def get_model_config(model_name: str) -> dict:
    """
    Get configuration for model architecture.

    Args:
        model_name: Model name ("lstm_unet", "unet_3d", "hybrid_lstm_3d")

    Returns:
        Dictionary with model configuration
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. "
                        f"Available: {list(MODEL_CONFIGS.keys())}")

    return MODEL_CONFIGS[model_name]


# ============================================================================
# PRINT CONFIGURATION SUMMARY
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("MULTI-TEMPORAL EXPERIMENT CONFIGURATION")
    print("=" * 80)
    print(f"\nBase directory: {BASE_DIR}")
    print(f"Multi-temporal directory: {MULTITEMPORAL_DIR}")
    print(f"\nData source: Sentinel-2 ({SENTINEL2_NUM_BANDS} bands)")
    print(f"Temporal sampling modes: {list(TEMPORAL_SAMPLING_MODES.keys())}")
    print(f"Available models: {MULTITEMPORAL_MODELS}")
    print(f"\nNormalization mode: {NORMALIZATION_MODE}")
    print(f"Default temporal sampling: {DEFAULT_TEMPORAL_SAMPLING}")
    print(f"\nRandom seeds: {RANDOM_SEEDS}")
    print(f"Wandb project: {WANDB_PROJECT}")
    print("\n" + "=" * 80)
