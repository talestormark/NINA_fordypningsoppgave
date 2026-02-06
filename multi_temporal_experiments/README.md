# Multi-Temporal Land-Take Detection Experiments

**Author**: tmstorma@stud.ntnu.no
**Start Date**: January 2026
**Status**: ✅ Experiments Complete (Analysis Phase)
**Timeline**: 3-4 months

---

## Purpose

This directory contains all experiments for the multi-temporal research phase, investigating temporal sampling strategies for land-take detection using Sentinel-2 imagery.

## Research Questions

- **RQ1**: What is the optimal temporal sampling density for land-take detection?
- **RQ2**: Do alternative temporal fusion architectures match ConvLSTM recurrence?

## Key Results

All models trained with unified protocol: 400 epochs, no early stopping, AdamW, cosine annealing LR.

| Model | Code | T | Val IoU | Test IoU | vs LSTM-7 |
|-------|------|---|---------|----------|-----------|
| LSTM-7 | `exp010` | 7 | 58.38% ± 10.44% | 63.9% | — |
| LSTM-14 | `exp002_v3` | 14 | 54.12% ± 10.96% | 61.8% | −4.26 pp |
| LSTM-2 | `exp003_v3` | 2 | 54.65% ± 11.52% | 53.0% | −3.73 pp |
| LSTM-1×1 | `exp004_v2` | 7 | 58.27% ± 10.76% | TBD | +0.39 pp (p=0.855) |

### Conclusions

1. **Annual sampling (T=7) is optimal**: Best validation and test performance
2. **More temporal data hurts**: LSTM-14 (T=14) underperforms due to seasonal noise overfitting
3. **Kernel size doesn't matter**: 1×1 matches 3×3 (p=0.855, n.s.)

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
│   │   ├── exp010_lstm7_no_es_fold{0-4}/ # LSTM-7: Annual sampling (T=7)
│   │   ├── exp002_v3_fold{0-4}/       # LSTM-14: Bi-seasonal sampling (T=14)
│   │   ├── exp003_v3_fold{0-4}/       # LSTM-2: Bi-temporal sampling (T=2)
│   │   └── exp004_v2_fold{0-4}/       # LSTM-1×1: 1×1 kernel ablation
│   │
│   ├── analysis/                      # Statistical analysis results
│   │   ├── boundary_f_score_d2.json   # BF@2 analysis for all experiments
│   │   └── 1d_vs_2d_analysis.json     # Kernel size comparison
│   │
│   ├── reports/                       # Analysis reports
│   │   └── normalization_stats.csv
│   │
│   └── logs/                          # SLURM logs
│
├── docs/                              # Documentation and reports
│   ├── 1Dvs2D.md                      # RQ2c: Kernel size ablation report
│   ├── Boundary-F-Score.md            # Boundary quality analysis report
│   ├── EXPERIMENT_002_REPORT.md       # Bi-seasonal experiment report
│   ├── EXPERIMENT_003_REPORT.md       # Bi-temporal experiment report
│   └── Experiments.tex                # LaTeX summary for thesis
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

## Model Configuration

| Aspect | Value |
|--------|-------|
| **Architecture** | LSTM-UNet (ResNet-50 encoder + ConvLSTM) |
| **Parameters** | 54.4M (2-layer, h=256) |
| **Data source** | Sentinel-2 (10m, 9 bands) |
| **Time steps** | 2, 7, or 14 (2018-2024) |
| **Input shape** | (B, T, C=9, H=64, W=64) |
| **Bands** | Blue, Green, Red, R1, R2, R3, NIR, SWIR1, SWIR2 |
| **Loss** | Focal Loss (α=0.25, γ=2.0) |
| **Optimizer** | AdamW (lr=0.01, weight_decay=5e-4) |
| **LR Scheduler** | Cosine annealing (eta_min=0.0001) |
| **Epochs** | 400 (no early stopping) |
| **Cross-validation** | 5-fold stratified by change level |

---

## Experiment Status

### Phase 0: Setup ✅
- [x] Create directory structure
- [x] Validate Sentinel-2 temporal data quality
- [x] Implement MultiTemporalSentinel2Dataset class
- [x] Compute normalization statistics
- [x] Test data loading pipeline
- [x] Memory profiling on GPU

### Phase 1: Temporal Sampling Experiments ✅
- [x] exp010: LSTM-7 with annual sampling (T=7) — 5-fold CV complete (58.38% val, 63.9% test)
- [x] exp002_v3: LSTM-14 with bi-seasonal sampling (T=14) — 5-fold CV complete (54.12% val, 61.8% test)
- [x] exp003_v3: LSTM-2 with bi-temporal sampling (T=2) — 5-fold CV complete (54.65% val, 53.0% test)
- [x] Per-sample statistical analysis (n=45 tiles, paired permutation tests)

### Phase 2: Temporal Modeling Ablation ✅
- [x] exp004_v2: LSTM-1×1 with 1×1 ConvLSTM kernel — 5-fold CV complete (58.27% val)
- [x] Compare 3×3 vs 1×1 kernel (RQ2f)
- [x] Statistical analysis (no significant difference, p=0.855)

### Phase 3: Boundary Quality Analysis ✅
- [x] Implement Boundary F-score (BF@2) metric
- [x] Compute BF for all experiments
- [x] Statistical comparison of boundary quality



---

## Quick Start

```bash
# 1. Activate environment
conda activate masterthesis

# 2. Train model (5-fold CV via SLURM)
sbatch scripts/slurm/train_exp010_lstm7_no_es.sh  # LSTM-7

# 3. Run statistical analysis
python scripts/modeling/statistical_analysis_persample.py

# 4. Run 1D vs 2D analysis
python scripts/modeling/statistical_analysis_1dvs2d.py

# 5. Evaluate on held-out test set
python scripts/modeling/evaluate_test_final.py --all-conditions
```

### Training a Single Fold

```bash
python scripts/modeling/train_multitemporal.py \
    --model-name lstm_unet \
    --temporal-sampling annual \
    --fold 0 \
    --epochs 400 \
    --batch-size 4 \
    --experiment-name exp010_lstm7_no_es \
    --wandb
```

### ConvLSTM Kernel Ablation

```bash
python scripts/modeling/train_multitemporal.py \
    --model-name lstm_unet \
    --temporal-sampling annual \
    --convlstm-kernel-size 1 \
    --epochs 400 \
    --experiment-name exp004_v2 \
    --wandb
```

---

## Resource Requirements

**Computational**:
- GPU: Tesla V100-PCIE-32GB
- RAM: 64GB
- Storage: ~10GB for models, predictions, logs

**Actual Training Times** (400 epochs, no early stopping):
- LSTM-7 (annual, T=7): ~25-30 minutes per fold
- LSTM-14 (bi-seasonal, T=14): ~40-50 minutes per fold
- LSTM-2 (bi-temporal, T=2): ~15-20 minutes per fold
- Full 5-fold CV: ~2-4 hours per experiment

---

## Contact

**Student**: tmstorma@stud.ntnu.no
**Data Provider**: zander.venter@nina.no
**Institution**: NTNU / NINA

---

## Key Files

- **Experiment log**: `EXPERIMENT_LOG.md` — Full experiment tracking with results
- **Statistical analysis**: `scripts/modeling/statistical_analysis_1dvs2d.py`
- **Boundary F-score**: `scripts/modeling/boundary_f_score_analysis.py`
- **Training script**: `scripts/modeling/train_multitemporal.py`
- **Model definitions**: `scripts/modeling/models_multitemporal.py`

## References

- VHR Baseline results: `/outputs/evaluation/TEST_RESULTS_SUMMARY.md`
- Data documentation: `/docs/DATASETS.md`
- WandB: https://wandb.ai/NINA_Fordypningsoppgave/landtake-multitemporal
