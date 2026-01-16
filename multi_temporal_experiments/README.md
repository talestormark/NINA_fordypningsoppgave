# Multi-Temporal Land-Take Detection Experiments

**Author**: tmstorma@stud.ntnu.no
**Start Date**: January 2026
**Status**: Setup Phase
**Timeline**: 3-4 months

---

## Purpose

This directory contains all experiments for the multi-temporal research phase, investigating three research questions:

- **RQ1**: Multi-temporal vs bi-temporal performance
- **RQ2**: Temporal sampling density effects
- **RQ3**: Temporal modeling paradigms (1D vs 2D)

## Baseline Performance (Reference)

**Model**: SiamConc + ResNet-50 (bi-temporal VHR)
**Test IoU**: 68.37% ± 0.35%
**Goal**: Improve by 5-10% using multi-temporal Sentinel-2 data

---

## Directory Structure

```
multi_temporal_experiments/
├── README.md                          # This file
├── EXPERIMENT_LOG.md                  # Running log of all experiments
├── config.py                          # Configuration for multi-temporal experiments
│
├── scripts/
│   ├── data_preparation/              # Data loading and preprocessing
│   │   ├── 01_validate_sentinel2_temporal.py
│   │   ├── 02_create_temporal_splits.py
│   │   ├── 03_compute_normalization_stats.py
│   │   └── dataset_multitemporal.py  # MultiTemporalSentinel2Dataset class
│   │
│   ├── modeling/                      # Model architectures
│   │   ├── lstm_unet.py               # LSTM-UNet (1D temporal)
│   │   ├── unet_3d.py                 # 3D U-Net (2D spatiotemporal)
│   │   ├── hybrid_models.py           # Hybrid architectures
│   │   └── train_multitemporal.py     # Training script
│   │
│   ├── evaluation/                    # Evaluation scripts
│   │   ├── evaluate_multitemporal.py
│   │   └── compare_architectures.py
│   │
│   ├── analysis/                      # Analysis and visualization
│   │   ├── rq1_analysis.py            # Multi-temporal benefit analysis
│   │   ├── rq2_analysis.py            # Sampling density analysis
│   │   ├── rq3_analysis.py            # Architecture comparison
│   │   └── visualize_temporal.py      # Temporal sequence visualization
│   │
│   └── slurm/                         # SLURM batch scripts
│       ├── train_lstm_unet.slurm
│       ├── train_3d_unet.slurm
│       └── run_experiment_suite.slurm
│
├── outputs/
│   ├── experiments/                   # Per-experiment results
│   │   ├── rq1_baseline_bitemporal/   # Baseline (already done)
│   │   ├── rq1_lstm_annual/           # LSTM with annual sampling
│   │   ├── rq1_lstm_quarterly/        # LSTM with quarterly sampling
│   │   ├── rq2_sampling_comparison/   # Sampling density study
│   │   └── rq3_architecture_comparison/ # 1D vs 2D modeling
│   │
│   ├── reports/                       # Analysis reports
│   │   ├── data_quality_temporal.txt
│   │   ├── normalization_stats.csv
│   │   ├── rq1_results_summary.md
│   │   ├── rq2_results_summary.md
│   │   └── rq3_results_summary.md
│   │
│   ├── figures/                       # Plots and visualizations
│   │   ├── temporal_sequences/        # Time series visualizations
│   │   ├── learning_curves/           # Training curves
│   │   └── architecture_comparison/   # Model comparison plots
│   │
│   └── logs/                          # Training logs and slurm outputs
│
├── docs/                              # Documentation
│   ├── SETUP_GUIDE.md                 # Setup instructions
│   ├── EXPERIMENT_PROTOCOL.md         # Standard experiment protocol
│   └── RESULTS_TEMPLATE.md            # Template for reporting results
│
└── notebooks/                         # Jupyter notebooks for exploration
    ├── 01_explore_sentinel2_temporal.ipynb
    ├── 02_test_data_loading.ipynb
    └── 03_visualize_results.ipynb
```

---

## Experiment Tracking

All experiments are logged in `EXPERIMENT_LOG.md` with:
- Experiment ID
- Research question
- Model architecture
- Hyperparameters
- Results (IoU, F1, etc.)
- Training time
- Notes/observations

---

## Key Differences from Baseline

| Aspect | Baseline (bi-temporal) | Multi-temporal |
|--------|------------------------|----------------|
| **Data source** | VHR Google (1m RGB) | Sentinel-2 (10m, 9 bands) |
| **Time steps** | 2 (2018, 2025) | 2-14 (2018-2024) |
| **Input shape** | (B, 6, H, W) | (B, T, C, H, W) or (B, C, T, H, W) |
| **Architecture** | SiamConc + ResNet-50 | LSTM-UNet / 3D U-Net |
| **Resolution** | 1m | 10m |
| **Bands** | RGB only | RGB + NIR + Red Edge + SWIR |

---

## Setup Status

### Phase 0: Setup (Current)
- [x] Create directory structure
- [ ] Validate Sentinel-2 temporal data quality
- [ ] Implement MultiTemporalSentinel2Dataset class
- [ ] Compute normalization statistics
- [ ] Test data loading pipeline
- [ ] Memory profiling on 80GB GPU

### Phase 1: RQ1 - Multi-temporal Benefit (Weeks 2-4)
- [ ] Train LSTM-UNet with annual sampling (7 time steps)
- [ ] Train LSTM-UNet with quarterly sampling (14 time steps)
- [ ] Compare to bi-temporal baseline (68.37% IoU)
- [ ] Analysis: Does multi-temporal improve performance?

### Phase 2: RQ2 - Sampling Density (Weeks 5-6)
- [ ] Train with bi-temporal (2 steps)
- [ ] Train with annual (7 steps)
- [ ] Train with quarterly (14 steps)
- [ ] Analysis: Accuracy vs computational cost trade-offs

### Phase 3: RQ3 - Architecture Comparison (Weeks 7-10)
- [ ] Train LSTM-UNet (1D temporal)
- [ ] Train 3D U-Net (2D spatiotemporal)
- [ ] Train hybrid model
- [ ] Analysis: 1D vs 2D temporal modeling

### Phase 4: Analysis & Writing (Weeks 11-12)
- [ ] Statistical significance testing
- [ ] Generate all figures for thesis
- [ ] Write results section
- [ ] Final documentation

---

## Quick Start

Once setup is complete, typical workflow:

```bash
# 1. Activate environment
conda activate landtake_env

# 2. Validate Sentinel-2 data
python scripts/data_preparation/01_validate_sentinel2_temporal.py

# 3. Compute normalization stats
python scripts/data_preparation/03_compute_normalization_stats.py

# 4. Train model (example)
python scripts/modeling/train_multitemporal.py \
    --model-name lstm_unet \
    --temporal-sampling annual \
    --epochs 200 \
    --batch-size 4 \
    --wandb

# 5. Evaluate model
python scripts/evaluation/evaluate_multitemporal.py \
    --checkpoint outputs/experiments/rq1_lstm_annual/best_model.pth \
    --output-dir outputs/evaluation/rq1_lstm_annual

# 6. Analyze results
python scripts/analysis/rq1_analysis.py
```

---

## Resource Requirements

**Computational**:
- GPU: 80GB A100/H100 (available ✓)
- RAM: 64-128GB recommended
- Storage: ~100GB for models, predictions, logs

**Estimated Training Time per Model**:
- LSTM-UNet (annual, 7 steps): ~2-3 hours
- LSTM-UNet (quarterly, 14 steps): ~4-6 hours
- 3D U-Net (quarterly, 14 steps): ~8-12 hours

---

## Contact

**Student**: tmstorma@stud.ntnu.no
**Data Provider**: zander.venter@nina.no
**Institution**: NTNU / NINA

---

## References

- Baseline results: `/outputs/evaluation/TEST_RESULTS_SUMMARY.md`
- Original plan: `/TENTATIVE_PLAN.md`
- Data documentation: `/docs/DATASETS.md`
- Git repo: `/cluster/home/tmstorma/NINA_fordypningsoppgave`
