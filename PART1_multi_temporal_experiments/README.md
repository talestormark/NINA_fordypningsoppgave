# Multi-Temporal Land-Take Detection Experiments

**Author**: tmstorma@stud.ntnu.no
**Start Date**: January 2026
**Status**: v2 re-run in progress (data_v2, EPSG:3035)

---

## Purpose

This directory contains all experiments for the multi-temporal research phase, investigating temporal sampling strategies and fusion architectures for land-take detection using Sentinel-2 imagery.

## Research Questions

- **RQ1 — Temporal sampling**: How does the density of temporal observations affect land-take detection performance? We compare bi-temporal (T=2), annual (T=7), and bi-seasonal (T=14) input sequences while holding the architecture fixed.
- **RQ2 — Architecture and fusion**: Does the ConvLSTM-based recurrent bottleneck provide advantages over simpler fusion strategies?
  - **RQ2a**: Does explicit temporal modelling improve over simple channel stacking (T=2)?
  - **RQ2b**: Does ConvLSTM recurrence improve over static feature concatenation (T=2)?
  - **RQ2c**: Does recurrence help with longer sequences, or does simple pooling suffice (T=7)?
  - **RQ2d**: Does learned non-recurrent temporal filtering (Conv3D) match recurrence (T=7)?
  - **RQ2e**: Does reducing ConvLSTM capacity affect performance (T=2 and T=7)?
  - **RQ2f**: Does spatial context inside the temporal module (kernel size) matter (T=7)?

---

## Dataset Versions

| Property | v1 (completed) | v2 (in progress) |
|----------|----------------|------------------|
| Tiles | 54 (53 usable) | 163 full-window |
| CRS | EPSG:4326 (non-square ~6.6×10m) | EPSG:3035 (square 10m) |
| Train+Val (CV) | 45 | 135 |
| Test (held-out) | 8 | 28 |
| Mask folder | `Land_take_masks/` | `Land_take_masks_coarse/` |
| Splits | `outputs/splits/` | `preprocessing/outputs/splits/part1/` |

---

## Directory Structure

```
PART1_multi_temporal_experiments/
├── README.md                         # This file
├── EXPERIMENT_LOG_v1.md              # v1 experiment log (completed)
├── EXPERIMENT_LOG_v2.md              # v2 experiment log (in progress)
├── config.py                         # Configuration for multi-temporal experiments
│
├── scripts/
│   ├── data_preparation/
│   │   ├── dataset_multitemporal.py  # MultiTemporalSentinel2Dataset class
│   │   ├── 01_validate_sentinel2_temporal.py  # v1 data validation
│   │   └── 03_compute_normalization_stats.py  # v1 per-fold stats
│   │
│   ├── modeling/
│   │   ├── train_multitemporal.py    # Training script
│   │   ├── models_multitemporal.py   # Model architectures
│   │   ├── convlstm.py              # ConvLSTM module
│   │   ├── evaluate_test_set.py      # Test set evaluation
│   │   ├── evaluate_test_final.py    # Final test evaluation
│   │   ├── boundary_f_score_analysis.py
│   │   ├── statistical_analysis_persample.py
│   │   ├── statistical_analysis_1dvs2d.py
│   │   └── profile_memory.py
│   │
│   ├── analysis/
│   │   ├── statistical_analysis.py
│   │   ├── plot_training_curves.py
│   │   ├── plot_iou_distributions.py
│   │   ├── qualitative_cv_analysis.py
│   │   ├── qualitative_test_analysis.py
│   │   ├── stratified_by_change_type.py
│   │   └── temporal_importance_analysis.py
│   │
│   └── slurm/
│       ├── v1/                       # All v1 SLURM scripts (archived)
│       └── v2/
│           └── train_all_experiments_v3.sh  # v2 launcher
│
├── outputs_v1/                       # v1 results (archived)
│   ├── experiments/                  # Per-experiment checkpoints & logs
│   ├── analysis/                     # Statistical analysis results
│   ├── figures/
│   ├── reports/
│   ├── logs/
│   ├── sample_change_levels.csv      # v1 change levels (45 samples)
│   └── final_data_rerun/             # Earlier EPSG:3035 attempt
│
├── outputs_v2/                       # v2 results (in progress)
│   └── logs/
│
└── docs/
    └── REPORT/
        ├── part1_data_v1/            # Archived v1 report (.tex files)
        └── part1_data_v2/            # v2 report (TODO placeholders)
```

---

## v2 Experiments

All models trained with unified protocol: 400 epochs, no early stopping, AdamW, cosine annealing LR, A100 80GB GPU.

### Phase 1: RQ1 — Temporal Sampling

| Exp | Model | T | Batch | RQ | Status |
|-----|-------|---|-------|-----|--------|
| exp001 | LSTM-7 | 7 (annual) | 4 | RQ1 | TODO |
| exp002 | LSTM-14 | 14 (bi-seasonal) | 2×2 accum | RQ1 | TODO |
| exp003 | LSTM-2 | 2 (bi-temporal) | 8 | RQ1 | TODO |

### Phase 2: RQ2 — Architecture and Fusion

| Exp | Model | T | Batch | RQ | Status |
|-----|-------|---|-------|-----|--------|
| exp004 | LSTM-1×1 | 7 (annual) | 4 | RQ2f | TODO |
| exp005 | EarlyFusion | 2 (bi-temporal) | 8 | RQ2a | TODO |
| exp006 | LateFusion | 2 (bi-temporal) | 8 | RQ2b | TODO |
| exp007 | Pool-7 | 7 (annual) | 4 | RQ2c | TODO |
| exp008 | Conv3D-7 | 7 (annual) | 4 | RQ2d | TODO |
| exp009 | LSTM-2-lite | 2 (bi-temporal) | 8 | RQ2e | TODO |
| exp010 | LSTM-7-lite | 7 (annual) | 4 | RQ2e | TODO |

---

## Quick Start (v2)

```bash
# 1. Activate environment
module load Anaconda3/2024.02-1 && conda activate masterthesis

# 2. Smoke test (interactive GPU node)
python PART1_multi_temporal_experiments/scripts/modeling/train_multitemporal.py \
    --model-name lstm_unet --temporal-sampling annual \
    --encoder-name resnet50 --encoder-weights imagenet \
    --lstm-hidden-dim 256 --lstm-num-layers 2 \
    --batch-size 4 --image-size 64 --num-workers 2 \
    --epochs 2 --lr 0.01 --optimizer adamw --scheduler cosine \
    --loss focal --focal-alpha 0.25 --focal-gamma 2.0 \
    --output-dir /tmp/smoke_test_v3 --seed 42 --fold 0 --num-folds 5 \
    --data-dir data_v2 --mask-subdir Land_take_masks_coarse \
    --splits-dir preprocessing/outputs/splits/part1 \
    --change-level-path preprocessing/outputs/splits/part1/split_info.csv

# 3. Launch Phase 1: RQ1 (15 jobs)
for exp in exp001 exp002 exp003; do
  for fold in 0 1 2 3 4; do
    sbatch PART1_multi_temporal_experiments/scripts/slurm/v2/train_all_experiments_v3.sh $exp $fold
  done
done

# 4. Launch Phase 2: RQ2 (35 jobs) — after Phase 1 analysis
for exp in exp004 exp005 exp006 exp007 exp008 exp009 exp010; do
  for fold in 0 1 2 3 4; do
    sbatch PART1_multi_temporal_experiments/scripts/slurm/v2/train_all_experiments_v3.sh $exp $fold
  done
done

# Or launch all 50 jobs at once
for exp in exp001 exp002 exp003 exp004 exp005 exp006 exp007 exp008 exp009 exp010; do
  for fold in 0 1 2 3 4; do
    sbatch PART1_multi_temporal_experiments/scripts/slurm/v2/train_all_experiments_v3.sh $exp $fold
  done
done
```

---

## Model Configuration

| Aspect | Value |
|--------|-------|
| **Architecture** | LSTM-UNet (ResNet-50 encoder + ConvLSTM) |
| **Parameters** | 54.4M (2-layer, h=256) |
| **Data source** | Sentinel-2 (10m, 9 bands, EPSG:3035) |
| **Time steps** | 2, 7, or 14 (2018-2024) |
| **Input shape** | (B, T, C=9, H=64, W=64) |
| **Bands** | Blue, Green, Red, R1, R2, R3, NIR, SWIR1, SWIR2 |
| **Loss** | Focal Loss (alpha=0.25, gamma=2.0) |
| **Optimizer** | AdamW (lr=0.01, weight_decay=5e-4) |
| **LR Scheduler** | Cosine annealing (eta_min=0.0001) |
| **Epochs** | 400 (no early stopping) |
| **Cross-validation** | 5-fold stratified by change level |

---

## Key Files

- **Experiment logs**: `EXPERIMENT_LOG_v1.md` (completed), `EXPERIMENT_LOG_v2.md` (active)
- **Training script**: `scripts/modeling/train_multitemporal.py`
- **Model definitions**: `scripts/modeling/models_multitemporal.py`
- **Dataset loader**: `scripts/data_preparation/dataset_multitemporal.py`
- **SLURM launcher**: `scripts/slurm/v2/train_all_experiments_v3.sh`
- **v2 report**: `docs/REPORT/part1_data_v2/`

---

## v1 Results (archived)

| Model | T | Val IoU | Test IoU |
|-------|---|---------|----------|
| LSTM-7 | 7 | 58.38% | 63.9% |
| LSTM-14 | 14 | 54.12% | 61.8% |
| LSTM-2 | 2 | 54.65% | 53.0% |

**v1 Conclusions**: Annual sampling (T=7) optimal; ConvLSTM recurrence provides no benefit over simple fusion; kernel size (1×1 vs 3×3) doesn't matter.

---

## Contact

**Student**: tmstorma@stud.ntnu.no
**Data Provider**: zander.venter@nina.no
**Institution**: NTNU / NINA
