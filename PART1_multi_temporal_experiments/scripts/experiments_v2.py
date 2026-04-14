#!/usr/bin/env python3
"""
Shared experiment registry for Part 1 v2 analysis scripts.

Single source of truth for all v2 experiment configurations, data paths,
and comparison families. All analysis scripts import from here instead of
defining their own EXPERIMENTS dicts.

v2 changes from v1:
- Data in EPSG:3035 (square 10m pixels) from data_v2/
- Masks in Land_take_masks_coarse/
- Splits in preprocessing/outputs/splits/part1/
- Outputs in outputs_v2/
- 10 experiments (exp001-exp010), no exp011
- exp names simplified: exp001, exp002, ... (no _v2/_v3 suffixes)
- exp009 = LSTM-2-lite (was lstm_lite), exp010 = LSTM-7-lite (was lstm7_lite)
- n = number of CV tiles determined by splits (not hardcoded 45)
"""

from pathlib import Path

# =============================================================================
# PATH CONSTANTS
# =============================================================================

_SCRIPT_DIR = Path(__file__).resolve().parent
_MT_DIR = _SCRIPT_DIR.parent  # PART1_multi_temporal_experiments/
_BASE_DIR = _MT_DIR.parent    # NINA_fordypningsoppgave/

V2_OUTPUTS_DIR = _MT_DIR / "outputs_v3"
V2_ANALYSIS_DIR = V2_OUTPUTS_DIR / "analysis"

V2_DATA_DIR = _BASE_DIR / "data_v2"
V2_SENTINEL_DIR = V2_DATA_DIR / "Sentinel"
V2_MASK_SUBDIR = "Land_take_masks_coarse"
V2_MASK_DIR = V2_DATA_DIR / V2_MASK_SUBDIR

V2_SPLITS_DIR = _BASE_DIR / "preprocessing" / "outputs" / "splits" / "unified_fullwindow"
V2_CHANGE_LEVEL_PATH = V2_SPLITS_DIR / "split_info.csv"

# GeoJSON metadata (for stratified analysis)
GEOJSON_PATH = _BASE_DIR / "land_take_bboxes_650m_v1.geojson"


# =============================================================================
# EXPERIMENT CONFIGURATIONS
# =============================================================================

EXPERIMENTS_V2 = {
    # ---- RQ1: Temporal Sampling ----
    'annual': {
        'name': 'exp001',
        'temporal_sampling': 'annual',
        'T': 7,
        'model_name': 'lstm_unet',
        'lstm_hidden_dim': 256,
        'lstm_num_layers': 2,
        'convlstm_kernel_size': 3,
        'description': 'Annual (T=7)',
        'rq': 'RQ1',
    },
    'bi_seasonal': {
        'name': 'exp002',
        'temporal_sampling': 'quarterly',
        'T': 14,
        'model_name': 'lstm_unet',
        'lstm_hidden_dim': 256,
        'lstm_num_layers': 2,
        'convlstm_kernel_size': 3,
        'description': 'Bi-seasonal (T=14)',
        'rq': 'RQ1',
    },
    'bi_temporal': {
        'name': 'exp003',
        'temporal_sampling': 'bi_temporal',
        'T': 2,
        'model_name': 'lstm_unet',
        'lstm_hidden_dim': 256,
        'lstm_num_layers': 2,
        'convlstm_kernel_size': 3,
        'description': 'Bi-temporal (T=2)',
        'rq': 'RQ1',
    },

    # ---- RQ2f: Kernel Ablation ----
    'k1x1': {
        'name': 'exp004',
        'temporal_sampling': 'annual',
        'T': 7,
        'model_name': 'lstm_unet',
        'lstm_hidden_dim': 256,
        'lstm_num_layers': 2,
        'convlstm_kernel_size': 1,
        'description': 'LSTM-1x1 (T=7)',
        'rq': 'RQ2f',
    },

    # ---- RQ2a-b: Bi-temporal Baselines ----
    'early_fusion': {
        'name': 'exp005',
        'temporal_sampling': 'bi_temporal',
        'T': 2,
        'model_name': 'early_fusion_unet',
        'description': 'Early-Fusion (T=2)',
        'rq': 'RQ2a',
    },
    'late_fusion': {
        'name': 'exp006',
        'temporal_sampling': 'bi_temporal',
        'T': 2,
        'model_name': 'late_fusion_concat',
        'description': 'Late-Fusion Concat (T=2)',
        'rq': 'RQ2b',
    },

    # ---- RQ2c-d: T=7 Architecture Comparisons ----
    'late_fusion_pool': {
        'name': 'exp007',
        'temporal_sampling': 'annual',
        'T': 7,
        'model_name': 'late_fusion_pool',
        'description': 'Pool-7 (T=7)',
        'rq': 'RQ2c',
    },
    'conv3d_fusion': {
        'name': 'exp008',
        'temporal_sampling': 'annual',
        'T': 7,
        'model_name': 'conv3d_fusion',
        'description': 'Conv3D-7 (T=7)',
        'rq': 'RQ2d',
    },

    # ---- RQ2e: Parameter-matched Controls ----
    # exp009 and exp010 are identical to exp003_lite and exp001_lite respectively.
    # Not re-run — symlinked in outputs_v2/ (exp009_fold* → exp003_lite_fold*,
    # exp010_fold* → exp001_lite_fold*). Analysis scripts resolve via symlinks.
    'lstm_lite': {
        'name': 'exp009',
        'temporal_sampling': 'bi_temporal',
        'T': 2,
        'model_name': 'lstm_unet',
        'lstm_hidden_dim': 32,
        'lstm_num_layers': 1,
        'convlstm_kernel_size': 3,
        'description': 'LSTM-2-lite (T=2)',
        'rq': 'RQ2e',
        'aliased_from': 'exp003_lite',  # identical config, reused results
    },
    'lstm7_lite': {
        'name': 'exp010',
        'temporal_sampling': 'annual',
        'T': 7,
        'model_name': 'lstm_unet',
        'lstm_hidden_dim': 32,
        'lstm_num_layers': 1,
        'convlstm_kernel_size': 3,
        'description': 'LSTM-7-lite (T=7)',
        'rq': 'RQ2e',
        'aliased_from': 'exp001_lite',  # identical config, reused results
    },

    # ---- RQ1-lite: Reduced-capacity temporal sampling ----
    'annual_lite': {
        'name': 'exp001_lite',
        'temporal_sampling': 'annual',
        'T': 7,
        'model_name': 'lstm_unet',
        'lstm_hidden_dim': 32,
        'lstm_num_layers': 1,
        'convlstm_kernel_size': 3,
        'description': 'LSTM-7-lite-RQ1 (T=7)',
        'rq': 'RQ1',
    },
    'bi_seasonal_lite': {
        'name': 'exp002_lite',
        'temporal_sampling': 'quarterly',
        'T': 14,
        'model_name': 'lstm_unet',
        'lstm_hidden_dim': 32,
        'lstm_num_layers': 1,
        'convlstm_kernel_size': 3,
        'description': 'LSTM-14-lite (T=14)',
        'rq': 'RQ1',
    },
    'bi_temporal_lite': {
        'name': 'exp003_lite',
        'temporal_sampling': 'bi_temporal',
        'T': 2,
        'model_name': 'lstm_unet',
        'lstm_hidden_dim': 32,
        'lstm_num_layers': 1,
        'convlstm_kernel_size': 3,
        'description': 'LSTM-2-lite-RQ1 (T=2)',
        'rq': 'RQ1',
    },
}

# =============================================================================
# CONDITION GROUPS
# =============================================================================

TEMPORAL_CONDITIONS = ['annual', 'bi_temporal', 'bi_seasonal']

TEMPORAL_LITE_CONDITIONS = ['annual_lite', 'bi_temporal_lite', 'bi_seasonal_lite']

ARCHITECTURE_CONDITIONS = [
    'early_fusion', 'late_fusion', 'late_fusion_pool', 'conv3d_fusion',
    'lstm_lite', 'lstm7_lite',
    # k1x1 (exp004) dropped: v1 null result (p=0.855), kernel ablation superseded
]

# =============================================================================
# COMPARISON FAMILIES (for Holm-Bonferroni correction)
# =============================================================================

COMPARISON_FAMILIES = {
    'Temporal regime (RQ1)': {
        'comparisons': [
            ('Annual vs Bi-temporal', 'annual', 'bi_temporal'),
            ('Annual vs Bi-seasonal', 'annual', 'bi_seasonal'),
        ],
        'm': 2,
    },
    'Bi-temporal baselines (RQ2, T=2)': {
        'comparisons': [
            ('Annual vs Early-Fusion', 'annual', 'early_fusion'),
            ('Annual vs Late-Fusion', 'annual', 'late_fusion'),
            ('Bi-temporal vs Early-Fusion', 'bi_temporal', 'early_fusion'),
            ('Bi-temporal vs Late-Fusion', 'bi_temporal', 'late_fusion'),
            ('Late-Fusion vs Early-Fusion', 'late_fusion', 'early_fusion'),
            ('Bi-temporal (LSTM) vs LSTM-lite', 'bi_temporal', 'lstm_lite'),
            ('LSTM-lite vs Early-Fusion', 'lstm_lite', 'early_fusion'),
            ('LSTM-lite vs Late-Fusion Concat', 'lstm_lite', 'late_fusion'),
        ],
        'm': 8,
    },
    'Extended baselines (RQ2, T=7)': {
        'comparisons': [
            ('Annual (LSTM) vs Pool-7', 'annual', 'late_fusion_pool'),
            ('Annual (LSTM) vs Conv3D-7', 'annual', 'conv3d_fusion'),
            ('Annual vs LSTM-7-lite', 'annual', 'lstm7_lite'),
            ('LSTM-7-lite vs Pool-7', 'lstm7_lite', 'late_fusion_pool'),
            ('LSTM-7-lite vs Conv3D-7', 'lstm7_lite', 'conv3d_fusion'),
            ('Pool-7 vs Conv3D-7', 'late_fusion_pool', 'conv3d_fusion'),
            # k1x1 comparison dropped: v1 null result (p=0.855)
        ],
        'm': 6,
    },
}

# =============================================================================
# DISPLAY NAMES (for LaTeX tables and figures)
# =============================================================================

DISPLAY_NAMES = {
    'annual': 'Annual (LSTM-7)',
    'bi_temporal': 'Bi-temporal (LSTM-2)',
    'bi_seasonal': 'Bi-seasonal (LSTM-14)',
    'early_fusion': 'Early-Fusion (T=2)',
    'late_fusion': 'Late-Fusion (T=2)',
    'late_fusion_pool': 'Pool-7 (T=7)',
    'conv3d_fusion': 'Conv3D-7 (T=7)',
    'lstm_lite': 'LSTM-2-lite (T=2)',
    'k1x1': 'LSTM-1x1 (T=7)',
    'lstm7_lite': 'LSTM-7-lite (T=7)',
    'annual_lite': 'LSTM-7-lite-RQ1 (T=7)',
    'bi_seasonal_lite': 'LSTM-14-lite (T=14)',
    'bi_temporal_lite': 'LSTM-2-lite-RQ1 (T=2)',
}

# Plot colors (consistent across all figures)
PLOT_COLORS = {
    'annual': '#2ecc71',      # green
    'bi_temporal': '#e74c3c',  # red
    'bi_seasonal': '#3498db',  # blue
    'early_fusion': '#f39c12', # orange
    'late_fusion': '#9b59b6',  # purple
    'late_fusion_pool': '#1abc9c',  # teal
    'conv3d_fusion': '#e67e22',     # dark orange
    'lstm_lite': '#95a5a6',         # gray
    'k1x1': '#34495e',             # dark blue-gray
    'lstm7_lite': '#d35400',        # rust
    'annual_lite': '#27ae60',        # darker green
    'bi_seasonal_lite': '#2980b9',   # darker blue
    'bi_temporal_lite': '#c0392b',   # darker red
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_experiment_dir(exp_key: str, fold: int) -> Path:
    """Get full path to experiment checkpoint directory."""
    exp_name = EXPERIMENTS_V2[exp_key]['name']
    return V2_OUTPUTS_DIR / f"{exp_name}_fold{fold}"


def get_history_path(exp_key: str, fold: int) -> Path:
    """Get path to training history JSON."""
    return get_experiment_dir(exp_key, fold) / "history.json"


def get_checkpoint_path(exp_key: str, fold: int) -> Path:
    """Get path to best model checkpoint."""
    return get_experiment_dir(exp_key, fold) / "best_model.pth"


def get_config_path(exp_key: str, fold: int) -> Path:
    """Get path to experiment config JSON."""
    return get_experiment_dir(exp_key, fold) / "config.json"


def filter_experiments(keys: list = None) -> dict:
    """
    Filter EXPERIMENTS_V2 to a subset.

    Args:
        keys: List of experiment keys to include. If None, return all.

    Returns:
        Filtered dict of experiments.
    """
    if keys is None:
        return EXPERIMENTS_V2.copy()
    return {k: v for k, v in EXPERIMENTS_V2.items() if k in keys}


def check_experiment_completeness(exp_key: str, num_folds: int = 5) -> dict:
    """
    Check if all checkpoints exist for an experiment.

    Returns:
        dict with 'complete' bool and 'missing' list of fold indices.
    """
    missing = []
    for fold in range(num_folds):
        if not get_checkpoint_path(exp_key, fold).exists():
            missing.append(fold)
    return {
        'complete': len(missing) == 0,
        'missing': missing,
    }


def check_all_completeness(num_folds: int = 5) -> dict:
    """Check completeness of all experiments."""
    status = {}
    for key in EXPERIMENTS_V2:
        status[key] = check_experiment_completeness(key, num_folds)
    return status
