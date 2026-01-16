# Experiment 001: Multi-Temporal Land-Take Detection with LSTM-UNet

## 1. Introduction

### 1.1 Goal

This experiment investigates whether **multi-temporal Sentinel-2 imagery** can be used for automated land-take detection using deep learning. Specifically, we aim to:

1. Develop and validate an LSTM-UNet architecture that processes temporal sequences of satellite images
2. Optimize hyperparameters for the small dataset context
3. Establish baseline performance metrics for annual temporal sampling (T=7 timesteps)

Land-take refers to the conversion of natural or agricultural land to artificial surfaces (buildings, roads, infrastructure). Detecting land-take from satellite imagery enables large-scale environmental monitoring.

---

## 2. Data

### 2.1 Dataset Overview

| Property | Value |
|----------|-------|
| **Source** | Sentinel-2 Level-2A |
| **Resolution** | 10 meters/pixel |
| **Temporal coverage** | 2018-2024 (7 years) |
| **Observations per year** | 2 (Q2: Apr-Jun, Q3: Jul-Sep) |
| **Spectral bands** | 9 (Blue, Green, Red, RedEdge1-3, NIR, SWIR1, SWIR2) |
| **Samples** | 53 total (45 train/val + 8 test) |
| **Tile size** | 64×64 pixels (~640m × 640m) |
| **Labels** | Binary masks (land-take vs. no change) |

### 2.2 Temporal Sampling

For this experiment, we use **annual sampling** where the two quarterly observations per year are averaged:

```
Input shape: (Batch, T=7, Channels=9, H=64, W=64)
```

Each of the 7 timesteps represents one year (2018-2024). Averaging Q2 and Q3 reduces noise while preserving annual change patterns.

**Threshold-based selection**: If one quarter has >50% missing data and the other <20%, the cleaner quarter is used instead of averaging.

### 2.3 Data Split

We use **stratified 5-fold cross-validation** to obtain robust performance estimates:

| Split | Samples | Purpose |
|-------|---------|---------|
| Train | 36 per fold | Model training |
| Validation | 9 per fold | Hyperparameter selection |
| Test | 8 (held out) | Final evaluation |

**Stratification** is based on change level (percentage of positive pixels):
- Low (<5%): 15 samples
- Moderate (5-30%): 25 samples
- High (≥30%): 5 samples

Each validation fold contains 3 low, 5 moderate, and 1 high sample, ensuring balanced difficulty across folds.

### 2.4 Preprocessing

1. **Z-score normalization** per band using training set statistics
2. **NaN handling**: Missing values replaced with 0 after normalization
3. **Data augmentation**: Random horizontal/vertical flip, 90° rotation

---

## 3. Architecture

### 3.1 Model Overview

We use an **LSTM-UNet** architecture that combines:
- A shared CNN encoder for spatial feature extraction
- A ConvLSTM module for temporal fusion
- A U-Net decoder for segmentation

```
Input: (B, T=7, C=9, H=64, W=64)
  │
  ▼
┌─────────────────────────────────┐
│  Shared ResNet-50 Encoder       │  ← Applied to each timestep
│  (ImageNet pretrained)          │
│  Output: 5 feature scales       │
└─────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────┐
│  ConvLSTM (2 layers)            │  ← Temporal fusion at bottleneck
│  Hidden dim: 256                │
│  Output: (B, 256, H/32, W/32)   │
└─────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────┐
│  U-Net Decoder                  │  ← Skip connections with max-pool
│  5 upsampling stages            │     aggregation over time
└─────────────────────────────────┘
  │
  ▼
Output: (B, 1, H=64, W=64)
```

### 3.2 Key Components

**Encoder**: ResNet-50 pretrained on ImageNet. First convolution modified from 3→9 input channels. ImageNet weights used for RGB channels; scaled random initialization for remaining 6 bands.

**Temporal Fusion**: 2-layer ConvLSTM at the bottleneck (1/32 resolution). Processes the sequence of 7 bottleneck features and outputs the final hidden state.

**Skip Connections**: Max-pooling over the temporal dimension aggregates encoder features at each scale before concatenation with decoder features.

**Decoder**: Standard U-Net decoder with transposed convolutions for upsampling.

### 3.3 Model Statistics

| Property | Value |
|----------|-------|
| Total parameters | ~70M (with hidden dim=256) |
| GPU memory | ~5-6 GB (batch size=4) |
| Training time | ~12 min per fold |

---

## 4. Method

### 4.1 Experimental Design

The experiment consists of two phases:

**Phase 1: Hyperparameter Optimization**
- Single-fold pilot experiments on Fold 3
- Grid search over learning rate, optimizer, scheduler, LSTM dimension
- Goal: Find optimal configuration before full cross-validation

**Phase 2: Full Validation**
- 5-fold stratified cross-validation with optimized hyperparameters
- Statistical comparison to baseline configuration

### 4.2 Training Configuration

#### Baseline Configuration (Phase 1 starting point)

| Parameter | Value |
|-----------|-------|
| Optimizer | SGD (momentum=0.9) |
| Learning rate | 0.001 |
| LR scheduler | Linear decay → 1% |
| Weight decay | 5×10⁻⁴ |
| Loss function | Focal Loss (α=0.25, γ=2.0) |
| Epochs | 200 |
| Batch size | 4 |
| LSTM hidden dim | 512 |

#### Optimized Configuration (Phase 2)

| Parameter | Value |
|-----------|-------|
| Optimizer | **AdamW** |
| Learning rate | **0.01** |
| LR scheduler | **Cosine annealing** |
| Weight decay | 5×10⁻⁴ |
| Loss function | Focal Loss (α=0.25, γ=2.0) |
| Epochs | 200 |
| Batch size | 4 |
| LSTM hidden dim | **256** |

### 4.3 Hyperparameter Search

We tested the following hyperparameters on Fold 3:

| Experiment | Values Tested | Best |
|------------|---------------|------|
| Learning rate | 0.0001, 0.001, **0.01** | 0.01 |
| Optimizer | SGD, **AdamW** | AdamW |
| LR scheduler | Linear, **Cosine** | Cosine |
| LSTM hidden dim | **256**, 512, 1024 | 256 |
| Image size | 64, 128 | 64 (128 failed*) |
| Epochs | 200, 300 | 200 |

*Image size 128 failed because some samples are smaller than 128×128 pixels.

### 4.4 Evaluation Metrics

- **IoU** (Intersection over Union): Primary metric
- **F1-score**: Harmonic mean of precision and recall
- **Best checkpoint**: Model saved at epoch with highest validation IoU

### 4.5 Statistical Analysis

We compare baseline and optimized configurations using:
- **Paired t-test**: Same folds enable paired comparison
- **Cohen's d**: Effect size measure
- **Coefficient of variation**: Fold-to-fold consistency

---

## 5. Results

### 5.1 Phase 1: Hyperparameter Optimization

Single-fold experiments on Fold 3 identified optimal hyperparameters:

#### Learning Rate

| LR | Best IoU | Best Epoch | Observation |
|----|----------|------------|-------------|
| 0.0001 | 22.38% | 0 | Too slow |
| 0.001 | 39.80% | 166 | Slow convergence |
| **0.01** | **51.37%** | 29 | Best, but overfits |

#### Optimizer & Scheduler

| Optimizer | Scheduler | Best IoU | Δ vs SGD+Linear |
|-----------|-----------|----------|-----------------|
| SGD | Linear | 51.37% | (baseline) |
| SGD | Cosine | 52.89% | +1.5 pp |
| AdamW | Linear | 57.76% | +6.4 pp |
| **AdamW** | **Cosine** | **61.66%** | **+10.3 pp** |

#### LSTM Hidden Dimension

| Hidden Dim | Best IoU | Parameters |
|------------|----------|------------|
| **256** | **53.77%** | ~70M |
| 512 | 51.37% | ~95M |
| 1024 | 46.56% | ~140M |

**Key Finding**: Smaller LSTM dimension (256) generalizes better on this small dataset.

### 5.2 Phase 2: Full Cross-Validation

#### Optimized Configuration Results (5-Fold CV)

| Fold | Best IoU | Best Epoch | Best F1 |
|------|----------|------------|---------|
| 0 | 52.44% | 126 | 68.80% |
| 1 | 66.39% | 111 | 79.80% |
| 2 | 69.11% | 147 | 81.73% |
| 3 | 60.73% | 25 | 75.57% |
| 4 | 39.44% | 90 | 56.57% |
| **Mean** | **57.62% ± 10.73%** | - | **72.49% ± 9.12%** |

#### Comparison to Baseline

| Metric | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| **Mean IoU** | 25.61% ± 8.07% | **57.62% ± 10.73%** | **+32.0 pp (+125%)** |
| **Mean F1** | 40.17% ± 9.58% | **72.49% ± 9.12%** | **+32.3 pp (+80%)** |
| **CV** | 35.2% | 18.6% | More consistent |

#### Per-Fold Improvement

| Fold | Baseline | Optimized | Δ |
|------|----------|-----------|---|
| 0 | 24.69% | 52.44% | +27.8 pp |
| 1 | 21.82% | 66.39% | +44.6 pp |
| 2 | 22.89% | 69.11% | +46.2 pp |
| 3 | 41.07% | 60.73% | +19.7 pp |
| 4 | 17.59% | 39.44% | +21.9 pp |

### 5.3 Statistical Significance

**Paired t-test** comparing baseline vs. optimized across 5 folds:

| Statistic | Value |
|-----------|-------|
| Mean difference | +32.01 pp |
| Standard error | 5.63 pp |
| t-statistic | 5.687 |
| p-value | **< 0.01** |
| Cohen's d | 2.54 (large effect) |

**Conclusion**: The improvement is statistically significant (p < 0.01) with a large effect size.

### 5.4 Overfitting Analysis

| Fold | Best IoU | Final IoU | Drop | Train-Val Gap |
|------|----------|-----------|------|---------------|
| 0 | 52.44% | 45.99% | -6.4 pp | 24.5 pp |
| 1 | 66.39% | 57.80% | -8.6 pp | 9.4 pp |
| 2 | 69.11% | 60.03% | -9.1 pp | 4.6 pp |
| 3 | 60.73% | 37.60% | -23.1 pp | 30.2 pp |
| 4 | 39.44% | 36.81% | -2.6 pp | 29.2 pp |

**Observations**:
- Folds 1 and 2 show healthy training dynamics (small train-val gap)
- Fold 3 peaks very early (epoch 25) and overfits severely
- We mitigate overfitting by using the best checkpoint (not final)

---

## 6. Discussion

### 6.1 Key Findings

1. **Hyperparameter optimization is critical**: The optimized configuration achieved 57.62% IoU compared to 25.61% with baseline settings—a 125% relative improvement.

2. **AdamW outperforms SGD**: AdamW with cosine annealing provided the largest single improvement (+10.3 pp over SGD with linear decay).

3. **Smaller models generalize better**: LSTM hidden dimension of 256 outperformed 512 and 1024, suggesting the small dataset benefits from reduced model capacity.

4. **Higher learning rate works with proper optimizer**: LR=0.01 with AdamW achieves much faster convergence than LR=0.001 with SGD.

### 6.2 Limitations

1. **Small dataset**: Only 45 samples for training/validation limits model complexity and causes high variance.

2. **Overfitting**: Despite best-checkpoint selection, some folds show significant overfitting (Fold 3: 23 pp drop).

3. **Spatial resolution**: 10m Sentinel-2 resolution limits detection of small land-take features.

4. **No test set evaluation**: Results are on validation folds; held-out test performance unknown.

### 6.3 Comparison to VHR Baseline

A previous experiment using 1m VHR imagery achieved 68.37% IoU. The Sentinel-2 model achieves 57.62% IoU with optimized hyperparameters—a gap of ~11 pp. However, direct comparison is inappropriate due to:
- Different spatial resolution (10m vs 1m)
- Different spectral configuration (9 bands vs 3)
- Different temporal information (7 years vs bi-temporal)

---

## 7. Conclusion

This experiment demonstrates that **LSTM-UNet can effectively detect land-take from multi-temporal Sentinel-2 imagery**, achieving **57.62% ± 10.73% validation IoU** with optimized hyperparameters.

### Key Contributions

1. **Validated LSTM-UNet architecture** for multi-temporal Sentinel-2 land-take detection
2. **Identified optimal hyperparameters**: AdamW, cosine scheduler, LR=0.01, LSTM dim=256
3. **Achieved +32 pp improvement** over baseline through systematic optimization
4. **Established reproducible methodology** with stratified 5-fold CV

### Recommended Configuration

```bash
python train_multitemporal.py \
    --optimizer adamw \
    --scheduler cosine \
    --lr 0.01 \
    --lstm-hidden-dim 256 \
    --epochs 200 \
    --batch-size 4
```

### Next Steps

1. Compare temporal sampling strategies (bi-temporal T=2, quarterly T=14)
2. Evaluate on held-out test set
3. Investigate overfitting mitigation (early stopping, dropout)

---

## Appendix

### A. Files and Resources

**Code**:
- Model: `multi_temporal_experiments/scripts/modeling/models_multitemporal.py`
- Training: `multi_temporal_experiments/scripts/modeling/train_multitemporal.py`
- Dataset: `multi_temporal_experiments/scripts/data_preparation/dataset_multitemporal.py`

**Checkpoints** (optimized configuration):
```
exp001_optimized_fold0/best_model.pth (52.44% IoU)
exp001_optimized_fold1/best_model.pth (66.39% IoU)
exp001_optimized_fold2/best_model.pth (69.11% IoU)
exp001_optimized_fold3/best_model.pth (60.73% IoU)
exp001_optimized_fold4/best_model.pth (39.44% IoU)
```

**Monitoring**: https://wandb.ai/NINA_Fordypningsoppgave/landtake-multitemporal

### B. Experiment Metadata

| Property | Value |
|----------|-------|
| Experiment ID | exp001 |
| Date | January 14-16, 2026 |
| Hardware | NVIDIA V100/A100 GPUs |
| Framework | PyTorch 2.2.0 |
| Report version | 4.0 |

### C. Change Log

- v4.0: Complete rewrite with clearer structure (Goal, Data, Architecture, Method, Results)
- v3.6: Phase 2 complete with statistical analysis
- v3.5: Hyperparameter optimization complete
- v3.0: Stratified 5-fold CV results
- v1.0: Initial experiment
