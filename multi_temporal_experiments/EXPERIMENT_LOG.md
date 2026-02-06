# Multi-Temporal Experiments - Running Log

**Purpose**: Track all experiments with IDs, configurations, and results
**Format**: Each experiment gets unique ID and standardized reporting

---

## Report ↔ Code Naming Map

The LaTeX report uses short, descriptive names; the codebase uses `exp00X` directory names.
This table maps between the two.

| Report name  | Code directory              | Architecture                    | *T* | Params | Report section |
|--------------|-----------------------------|---------------------------------|-----|--------|----------------|
| LSTM-7       | `exp010_lstm7_no_es`        | 2-layer ConvLSTM (*h*=256)      | 7   | 54.4 M | 2.2 (RQ1)      |
| LSTM-14      | `exp002_v3`                 | 2-layer ConvLSTM (*h*=256)      | 14  | 54.4 M | 2.2 (RQ1)      |
| LSTM-2       | `exp003_v3`                 | 2-layer ConvLSTM (*h*=256)      | 2   | 54.4 M | 2.2 (RQ1)      |
| EarlyFusion  | `exp005_early_fusion`       | Stacked-channel U-Net           | 2   | 32.6 M | 2.3 (RQ2a)     |
| LateFusion   | `exp006_late_fusion`        | Shared encoder + concat         | 2   | 31.1 M | 2.3 (RQ2b)     |
| Pool-7       | `exp007_late_fusion_pool`   | Shared encoder + mean pool      | 7   | 30.1 M | 2.3 (RQ2c)     |
| Conv3D-7     | `exp008_conv3d_fusion`      | Shared encoder + 3D conv        | 7   | 32.9 M | 2.3 (RQ2d)     |
| LSTM-2-lite  | `exp009_lstm_lite`          | 1-layer ConvLSTM (*h*=32)       | 2   | 30.3 M | 2.3 (RQ2e)     |
| LSTM-1×1     | `exp004_v2`                 | 2-layer ConvLSTM (1×1 kernel)   | 7   | 54.1 M | 2.3 (RQ2f)     |
| LSTM-7-lite  | `exp011_lstm7_lite`         | 1-layer ConvLSTM (*h*=32)       | 7   | 30.3 M | 2.3 (RQ2e)     |

---

## Experiment Index

All LSTM models trained with unified protocol: 400 epochs, no early stopping, AdamW, cosine annealing LR.

| ID | RQ | Model | Sampling | Status | Val IoU | Test IoU | Notes |
|----|----|-------|----------|--------|---------|----------|-------|
| `baseline` | - | SiamConc-ResNet50 | Bi-temporal VHR | ✅ Complete | 54.57% ± 0.52% | 68.37% ± 0.35% | VHR baseline (1m resolution) |
| `exp010` | RQ1 | LSTM-UNet | Annual (7 steps) | ✅ Complete | **58.38% ± 10.44%** | **63.9%** | LSTM-7: canonical annual model |
| `exp002_v3` | RQ1 | LSTM-UNet | Bi-seasonal (14 steps) | ✅ Complete | **54.12% ± 10.96%** | **61.8%** | LSTM-14: -4.26 pp vs LSTM-7 |
| `exp003_v3` | RQ1 | LSTM-UNet | Bi-temporal (2 steps) | ✅ Complete | **54.65% ± 11.52%** | **53.0%** | LSTM-2: -3.73 pp vs LSTM-7 |
| `exp004_v2` | RQ2f | LSTM-UNet (1×1) | Annual (7 steps) | ✅ Complete | **58.27% ± 10.76%** | TBD | LSTM-1×1: +0.39 pp vs 3×3 (p=0.855, n.s.) |
| `exp005` | RQ2a | Early-Fusion U-Net | Bi-temporal (stacked) | ✅ Complete | **50.81% ± 9.09%** | TBD | No temporal modeling |
| `exp006` | RQ2b | Late-Fusion Concat | Bi-temporal | ✅ Complete | **54.82% ± 10.69%** | TBD | Concat vs recurrence |
| `exp007` | RQ2c | Late-Fusion Pool | Annual (7 steps) | ✅ Complete | **57.75% ± 9.82%** | TBD | Pool vs LSTM at T=7 |
| `exp008` | RQ2d | Conv3D Fusion | Annual (7 steps) | ✅ Complete | **58.88% ± 11.37%** | TBD | 3D conv vs LSTM at T=7 |
| `exp009` | RQ2e | ConvLSTM-lite | Bi-temporal (2 steps) | ✅ Complete | **56.27% ± 10.11%** | TBD | Param-matched LSTM at T=2 |
| `exp011` | RQ2e | ConvLSTM-lite | Annual (7 steps) | ✅ Complete | **59.88% ± 11.01%** | TBD | Param-matched LSTM at T=7 (~30.3M) |


---

## Baseline (Reference)

**Experiment ID**: `baseline`
**Date**: November 2025
**Research Question**: Baseline for comparison
**Status**: ✅ Complete

### Configuration
```yaml
Model: SiamConc + ResNet-50
Data Source: VHR Google RGB
Resolution: 1m
Time Steps: 2 (2018, 2025)
Input Shape: (B, 6, H, W)
Batch Size: 4
Image Size: 512
Epochs: 200
Loss: Focal Loss (α=0.25, γ=2.0)
Optimizer: SGD (lr=0.01, momentum=0.9, weight_decay=5e-4)
```

### Results (3 random seeds: 42, 123, 456)
```
Test IoU:       68.37% ± 0.35%
Test F1:        81.22% ± 0.25%
Test Precision: 77.98% ± 2.82%
Test Recall:    84.87% ± 2.72%

Val IoU:        54.57% ± 0.52%
```

### Key Observations
- Excellent generalization (test >> validation)
- High reproducibility (CV = 0.51%)
- Strong recall, moderate precision
- Training time: ~2 hours per seed

### Files
- Training: `outputs/training/siam_conc_resnet50_seed{42,123,456}/`
- Evaluation: `outputs/evaluation/siam_conc_resnet50_seed{42,123,456}/`
- Report: `outputs/evaluation/TEST_RESULTS_SUMMARY.md`
- Wandb: https://wandb.ai/NINA_Fordypningsoppgave/landtake-detection/

---

## Experiment 001: LSTM-UNet Annual (Pilot - Seed 42)

**Experiment ID**: `exp001`
**Date**: January 14, 2026
**Research Question**: RQ1 - Multi-temporal vs Bi-temporal
**Status**: ✅ Complete (Pilot with seed 42)

### Configuration
```yaml
Model: LSTM-UNet + ResNet-50
Parameters: 95.07M
Data Source: Sentinel-2
Resolution: 10m
Time Steps: 7 (annual, 2018-2024)
Bands per Step: 9 (blue, green, red, R1, R2, R3, nir, swir1, swir2)
Input Shape: (B=4, T=7, C=9, H=64, W=64)
Batch Size: 4
Image Size: 64x64 (actual tile size: 66×92)
Epochs: 200
Loss: Focal Loss (α=0.25, γ=2.0)
Optimizer: SGD (lr=0.001, momentum=0.9, weight_decay=5e-4)
LR Scheduler: Linear decay (0.001 → 0.00001)
Gradient Clipping: max_norm=1.0
Normalization: Z-score per band (train set statistics)
Seeds: 1 (42 only - pilot run)
GPU: Tesla V100-PCIE-32GB
Training Time: ~12 minutes
```

### Results (Seed 42)
```yaml
Best Validation IoU:  39.62% (Epoch 173)
Best Validation F1:   56.75% (Epoch 173)
Best Validation Loss: 0.0432 (Epoch 173)

Final Validation IoU:  35.29% (Epoch 200)
Final Validation F1:   52.17% (Epoch 200)
Final Validation Loss: 0.0357 (Epoch 200)

Training Convergence:
  Epoch 1:   14.51% IoU
  Epoch 80:  28.09% IoU
  Epoch 173: 39.62% IoU (BEST)
  Epoch 200: 35.29% IoU

Test IoU: TBD (not yet evaluated)
```

**💡 Insights**:
- **Resolution is limiting factor**: 10m S2 << 1m VHR in detection capability
- **Image size matters**: 64×64 provides limited spatial context vs 512×512
- **Unrealistic comparison**: S2 cannot match VHR performance (different sensors)
- **Revised goal**: Compare temporal samplings within S2, not against VHR baseline

### Files
- Training: `multi_temporal_experiments/outputs/experiments/exp001_lstm_unet_annual_seed42/`
- Best checkpoint: `exp001_lstm_unet_annual_seed42/best_model.pth` (Epoch 173)
- Config: `exp001_lstm_unet_annual_seed42/config.json`
- History: `exp001_lstm_unet_annual_seed42/history.json`
- SLURM log: `outputs/logs/slurm_lstm_unet_train_23919997.log`
- WandB: https://wandb.ai/NINA_Fordypningsoppgave/landtake-multitemporal/runs/5vtfyao7
- Analysis: `multi_temporal_experiments/exp001_planning/PILOT_RESULTS_ANALYSIS.md`

### Next Steps
1. **Evaluate on test set** for reliable IoU (validation set only 8 samples)
2. **Run additional seeds** (123, 456) to compute mean ± std
3. **Run exp003** (bi-temporal S2) to establish S2 baseline
4. **Compare temporal samplings** (bi-temporal vs annual vs quarterly) on S2
5. **Revise expectations**: S2 multi-temporal vs S2 bi-temporal (not vs VHR)


## Experiment 002: LSTM-14 (Bi-seasonal T=14)

**Experiment ID**: `exp002_v3`
**Date**: January 19, 2026 (retrained 2026-02-05)
**Research Question**: RQ1 - Does higher temporal density (T=14) improve over annual (T=7)?
**Status**: ✅ Complete

### Configuration
```yaml
Model: LSTM-UNet + ResNet-50
Parameters: 54.4M (with LSTM hidden dim=256)
Data Source: Sentinel-2
Resolution: 10m
Time Steps: 14 (quarterly: Q2 and Q3 for each year 2018-2024)
Bands per Step: 9 (blue, green, red, R1, R2, R3, nir, swir1, swir2)
Input Shape: (B=2, T=14, C=9, H=64, W=64)
Batch Size: 2
Accumulation Steps: 2
Effective Batch Size: 4
Image Size: 64x64
Epochs: 400 (no early stopping)
Loss: Focal Loss (α=0.25, γ=2.0)
Optimizer: AdamW (lr=0.01, weight_decay=5e-4)
LR Scheduler: Cosine annealing (eta_min=0.0001)
LSTM Hidden Dim: 256
LSTM Layers: 2
Skip Aggregation: max (temporal pooling)
Normalization: Z-score per band (computed per fold from training set)
Cross-Validation: 5-fold stratified (by change level)
Seed: 42
```

### Results (5-Fold CV)
| Fold | Best IoU | Best Epoch | Best F1 |
|------|----------|------------|---------|
| **Mean** | **54.12% ± 10.96%** | - | - |
| **Test** | **61.8%** | - | - |

### Comparison to LSTM-7 (exp010)
- **Mean difference**: -4.26 pp (LSTM-7 better)
- **Conclusion**: LSTM-14 underperforms LSTM-7

### Key Finding
**Hypothesis REJECTED**: More temporal data HURTS performance!
- Bi-seasonal (T=14) is 4.26 pp WORSE than Annual (T=7)
- Causes: overfitting to seasonal noise, longer sequences harder to learn

### Files
- SLURM script: `scripts/slurm/retrain_exp002_quarterly.sh`
- Checkpoints: `outputs/experiments/exp002_v3_fold{0-4}/best_model.pth`

---

## Experiment 003: LSTM-2 (Bi-temporal T=2)

**Experiment ID**: `exp003_v3`
**Date**: January 19, 2026 (retrained 2026-02-05)
**Research Question**: RQ1 - Sentinel-2 baseline with minimal temporal information
**Status**: ✅ Complete

### Configuration
```yaml
Model: LSTM-UNet + ResNet-50
Parameters: 54.4M (with LSTM hidden dim=256)
Data Source: Sentinel-2
Resolution: 10m
Time Steps: 2 (bi-temporal: 2018 Q2, 2024 Q3)
Bands per Step: 9 (blue, green, red, R1, R2, R3, nir, swir1, swir2)
Input Shape: (B=8, T=2, C=9, H=64, W=64)
Batch Size: 8
Image Size: 64x64
Epochs: 400 (no early stopping)
Loss: Focal Loss (α=0.25, γ=2.0)
Optimizer: AdamW (lr=0.01, weight_decay=5e-4)
LR Scheduler: Cosine annealing (eta_min=0.0001)
LSTM Hidden Dim: 256
LSTM Layers: 2
Skip Aggregation: max (temporal pooling)
Normalization: Z-score per band (computed per fold from training set)
Cross-Validation: 5-fold stratified (by change level)
Seed: 42
```

### Results (5-Fold CV)
| Fold | Best IoU | Best Epoch | Best F1 |
|------|----------|------------|---------|
| **Mean** | **54.65% ± 11.52%** | - | - |
| **Test** | **53.0%** | - | - |

### Comparison to LSTM-7 (exp010)
- **Mean difference**: -3.73 pp (LSTM-7 better)
- **Conclusion**: LSTM-2 underperforms LSTM-7, but difference is moderate

### Key Finding
Bi-temporal (T=2) achieves reasonable performance but:
- LSTM-7 test IoU (63.9%) is 10.9 pp better than LSTM-2 test (53.0%)
- Confirms value of multi-temporal sequences for land-take detection

### Files
- SLURM script: `scripts/slurm/retrain_exp003_bitemporal.sh`
- Checkpoints: `outputs/experiments/exp003_v3_fold{0-4}/best_model.pth`

---

## Experiment 004: LSTM-1×1 (Per-pixel Temporal Modeling)

**Experiment ID**: `exp004_v2`
**Date**: January 27, 2026 (retrained 2026-02-05)
**Research Question**: RQ2f - Does spatial context in temporal modeling (3×3 vs 1×1 ConvLSTM kernel) matter?
**Status**: ✅ Complete

### Configuration
```yaml
Model: LSTM-UNet + ResNet-50 (1×1 ConvLSTM kernel)
Parameters: 54.1M (vs 54.4M for 3×3 kernel)
Data Source: Sentinel-2
Resolution: 10m
Time Steps: 7 (annual, 2018-2024)
Bands per Step: 9 (blue, green, red, R1, R2, R3, nir, swir1, swir2)
Input Shape: (B=4, T=7, C=9, H=64, W=64)
Batch Size: 4
Image Size: 64x64
Epochs: 400 (no early stopping)
Loss: Focal Loss (α=0.25, γ=2.0)
Optimizer: AdamW (lr=0.01, weight_decay=5e-4)
LR Scheduler: Cosine annealing (eta_min=0.0001)
LSTM Hidden Dim: 256
LSTM Layers: 2
ConvLSTM Kernel Size: 1 (per-pixel temporal modeling)
Cross-Validation: 5-fold stratified (by change level)
Seed: 42
```

### Results (5-Fold CV)
| Fold | Best IoU | Best Epoch | Best F1 |
|------|----------|------------|---------|
| **Mean** | **58.27% ± 10.76%** | - | - |

### Statistical Analysis (1D vs 2D, per-sample n=45)
- **3×3 (exp010)**: Mean IoU 46.20%
- **1×1 (exp004_v2)**: Mean IoU 45.81%
- **Mean difference**: +0.39 pp (3×3 better)
- **Permutation test p-value**: 0.855
- **Bootstrap 95% CI**: [−3.53, +4.40] pp
- **Probability**: 3×3 wins 51.1% of tiles
- **Conclusion**: **No significant difference** between 1×1 and 3×3 kernels

### Key Finding
**H1 REJECTED**: Spatial context in ConvLSTM kernel does NOT improve performance!
- Per-pixel (1×1) temporal modeling matches patch-based (3×3)
- Suggests temporal dynamics are captured at per-pixel level for this task

### Files
- SLURM script: `scripts/slurm/retrain_exp004_1x1.sh`
- Analysis: `scripts/modeling/statistical_analysis_1dvs2d.py`
- Checkpoints: `outputs/experiments/exp004_v2_fold{0-4}/best_model.pth`

---

## Statistical Summary: All Temporal Experiments

### Per-Sample Analysis (n=45 tiles, out-of-fold predictions)

All comparisons use models trained with unified 400-epoch protocol.

| Comparison | Metric | Mean Δ (pp) | 95% CI | p-value | Significant |
|------------|--------|-------------|--------|---------|-------------|
| LSTM-7 vs LSTM-14 | IoU | +4.26 | TBD | TBD | TBD |
| LSTM-7 vs LSTM-2 | IoU | +3.73 | TBD | TBD | TBD |
| 3×3 vs 1×1 kernel | IoU | +0.39 | [−3.53, +4.40] | 0.855 | ❌ No |

*Note: Full per-sample statistical analysis pending completion of stats_analysis job.*

### Conclusions
1. **Annual sampling (T=7) is optimal**: Best val IoU (58.38%) and test IoU (63.9%)
2. **More temporal data hurts**: LSTM-14 (54.12%) underperforms LSTM-7
3. **Kernel size doesn't matter**: 1×1 matches 3×3 (p=0.855)

### Analysis Files
- Statistical analysis: `scripts/modeling/statistical_analysis_persample.py`
- 1D vs 2D analysis: `scripts/modeling/statistical_analysis_1dvs2d.py`
- Boundary F-score: `scripts/modeling/boundary_f_score_analysis.py`
- Results: `outputs/analysis/`

---

## Experiment 005: Early-Fusion U-Net (No Temporal Modeling)

**Experiment ID**: `exp005`
**Date**: 2026-02-02
**Research Question**: RQ0 - Do we need temporal modeling at all?
**Status**: ✅ Complete

### Motivation
Establish whether a simple U-Net with stacked bi-temporal input (18 channels) can match
the LSTM-UNet performance. If LSTM-UNet significantly outperforms this baseline, it
validates that temporal modeling architecture matters, not just having multi-date input.

### Configuration
```yaml
Model: Plain U-Net (smp.Unet)
Encoder: ResNet-50 (ImageNet pretrained)
Input: Stack 2018+2024 → 18 channels (9 bands × 2 timesteps)
Input Shape: (B=8, C=18, H=64, W=64)
Parameters: ~25M (no ConvLSTM)
Temporal Modeling: None (early fusion via channel stacking)
Batch Size: 8
Epochs: 400
Loss: Focal Loss (α=0.25, γ=2.0)
Optimizer: AdamW (lr=0.01, weight_decay=5e-4)
LR Scheduler: Cosine annealing (eta_min=0.0001)
Cross-Validation: 5-fold stratified
Seed: 42
```

### Hypothesis
LSTM-UNet (exp003, 53.29% IoU) should outperform early-fusion U-Net because:
- ConvLSTM captures temporal dynamics that channel stacking cannot
- Shared encoder learns better features than treating timesteps as extra channels

### Success Criteria
- [x] Training converges without instability → ✅ Achieved
- [ ] LSTM-UNet (exp003) significantly outperforms this baseline (p < 0.05)
- [ ] Difference is ≥5 pp to be practically meaningful

### Files
- SLURM script: `scripts/slurm/train_exp005_early_fusion.sh`
- Checkpoints: `outputs/experiments/exp005_early_fusion_fold{0-4}/`

### Results (5-Fold CV)
| Fold | Best IoU | Best Epoch | Best F1 |
|------|----------|------------|---------|
| 0 | 45.36% | 198 | 62.41% |
| 1 | 67.94% | 227 | 80.91% |
| 2 | 50.85% | 249 | 67.42% |
| 3 | 48.25% | 20 | 65.09% |
| 4 | 41.66% | 236 | 58.82% |
| **Mean** | **50.81% ± 9.09%** | - | **66.93% ± 7.55%** |

### Notes
- Lowest of the T=2 baselines (50.81% vs 54.82% LateFusion, 53.29% LSTM-2)
- Stacking channels destroys temporal structure — shared encoder approach is better

---

## Experiment 006: Late-Fusion Concat (Multi-View Aggregation)

**Experiment ID**: `exp006`
**Date**: 2026-02-02
**Research Question**: RQ0 - Does recurrence help beyond simple multi-view aggregation?
**Status**: ✅ Complete

### Motivation
Test whether the ConvLSTM's recurrent structure adds value over simple feature concatenation.
This baseline uses the same shared encoder as LSTM-UNet but replaces ConvLSTM with
a simple concat + 1×1 conv fusion. If LSTM-UNet outperforms this, recurrence specifically
helps (not just multi-view encoding).

### Configuration
```yaml
Model: Late-Fusion Concat (custom)
Encoder: ResNet-50 (ImageNet pretrained, shared across timesteps)
Input Shape: (B=8, T=2, C=9, H=64, W=64)
Bottleneck Fusion: Concat → 1×1 Conv (2*2048 → 512)
Skip Aggregation: max (same as LSTM-UNet)
Parameters: ~27M (no ConvLSTM recurrence, just 1×1 conv fusion)
Batch Size: 8
Epochs: 400
Loss: Focal Loss (α=0.25, γ=2.0)
Optimizer: AdamW (lr=0.01, weight_decay=5e-4)
LR Scheduler: Cosine annealing (eta_min=0.0001)
Cross-Validation: 5-fold stratified
Seed: 42
```

### Architecture Comparison

| Component | LSTM-UNet (exp003) | Late-Fusion Concat (exp006) |
|-----------|-------------------|----------------------------|
| Encoder | Shared ResNet-50 | Shared ResNet-50 |
| Bottleneck | ConvLSTM (recurrent) | Concat + 1×1 Conv (static) |
| Skip Agg | max over time | max over time |
| Decoder | UnetDecoder | UnetDecoder |
| Params | ~70M | ~27M |

### Hypothesis
LSTM-UNet should outperform late-fusion concat because:
- Recurrent gates model temporal dependencies explicitly
- ConvLSTM captures change dynamics, not just feature differences

If exp006 matches exp003:
- Simple aggregation suffices for this task
- LSTM adds complexity without benefit

### Success Criteria
- [x] Training converges without instability → ✅ Achieved
- [ ] LSTM-UNet (exp003) significantly outperforms this baseline (p < 0.05)
- [ ] Difference is ≥3 pp to be practically meaningful

### Files
- SLURM script: `scripts/slurm/train_exp006_late_fusion.sh`
- Checkpoints: `outputs/experiments/exp006_late_fusion_fold{0-4}/`

### Results (5-Fold CV)
| Fold | Best IoU | Best Epoch | Best F1 |
|------|----------|------------|---------|
| 0 | 46.74% | 164 | 63.70% |
| 1 | 66.09% | 174 | 79.58% |
| 2 | 64.87% | 179 | 78.69% |
| 3 | 57.96% | 12 | 73.38% |
| 4 | 38.43% | 212 | 55.53% |
| **Mean** | **54.82% ± 10.69%** | - | **70.18% ± 9.25%** |

### Notes
- Outperforms Early-Fusion (54.82% vs 50.81%) — shared encoder with separate processing helps
- Very close to LSTM-2 (53.29%) — recurrence adds little over simple concat at T=2

---

## Experiment 007: Late-Fusion Temporal Pooling (T=7)

**Experiment ID**: `exp007`
**Date**: 2026-02-03
**Research Question**: RQ0 - Does recurrence help with longer sequences, or does simple pooling suffice at T=7?
**Status**: ✅ Complete

### Motivation
Test whether the ConvLSTM recurrence at the bottleneck provides any benefit over simple mean pooling when processing longer temporal sequences (T=7). This is the most important of the gap-closing experiments because it directly tests whether temporal ordering matters at all.

### Configuration
```yaml
Model: Late-Fusion Pool (custom)
Encoder: ResNet-50 (ImageNet pretrained, shared across timesteps)
Input Shape: (B=4, T=7, C=9, H=64, W=64)
Bottleneck Fusion: Mean pool over T -> 1x1 Conv (2048 -> 512) + BN + ReLU
Skip Aggregation: max (same as LSTM-UNet)
Parameters: ~30.1M
Batch Size: 4
Epochs: 400
Early Stopping: No
Loss: Focal Loss (alpha=0.25, gamma=2.0)
Optimizer: AdamW (lr=0.01, weight_decay=5e-4)
LR Scheduler: Cosine annealing (eta_min=0.0001)
Cross-Validation: 5-fold stratified
Seed: 42
```

### Architecture Comparison

| Component | LSTM-UNet (exp001) | Late-Fusion Pool (exp007) |
|-----------|-------------------|--------------------------|
| Encoder | Shared ResNet-50 | Shared ResNet-50 |
| Bottleneck | ConvLSTM (recurrent) | Mean pool + 1x1 conv |
| Skip Agg | max over time | max over time |
| Decoder | UnetDecoder | UnetDecoder |
| Params | ~54M | ~30M |

### Hypothesis
If LSTM-UNet (exp001) outperforms this pooling baseline, recurrence captures meaningful temporal ordering at T=7.
If pooling matches or beats LSTM-UNet, simple aggregation suffices and temporal ordering is uninformative.

### Success Criteria
- [x] Training converges without instability → ✅ Achieved
- [ ] Clear evidence for or against recurrence benefit at T=7

### Files
- SLURM script: `scripts/slurm/train_exp007_late_fusion_pool.sh`
- Checkpoints: `outputs/experiments/exp007_late_fusion_pool_fold{0-4}/`

### Results (5-Fold CV)
| Fold | Best IoU | Best Epoch | Best F1 |
|------|----------|------------|---------|
| 0 | 56.48% | 269 | 72.19% |
| 1 | 68.10% | 200 | 81.02% |
| 2 | 65.80% | 208 | 79.37% |
| 3 | 58.19% | 28 | 73.57% |
| 4 | 40.18% | 329 | 57.33% |
| **Mean** | **57.75% ± 9.82%** | - | **72.70% ± 8.38%** |

### Notes
- Matches LSTM-7 (57.62%) almost exactly — simple pooling suffices at T=7
- Strongest non-recurrent T=7 baseline

---

## Experiment 008: 3D Conv Bottleneck Fusion (T=7)

**Experiment ID**: `exp008`
**Date**: 2026-02-03
**Research Question**: RQ0 - Does learned spatiotemporal filtering (non-recurrent) match ConvLSTM recurrence?
**Status**: ✅ Complete

### Motivation
Test whether learned temporal filtering via 3D convolutions can match ConvLSTM recurrence. Unlike simple pooling (exp007), 3D conv can learn temporal filters but without recurrent state. This isolates whether the benefit of ConvLSTM (if any) comes from its recurrent gating or from learned temporal filtering.

### Configuration
```yaml
Model: Conv3D Fusion (custom)
Encoder: ResNet-50 (ImageNet pretrained, shared across timesteps)
Input Shape: (B=4, T=7, C=9, H=64, W=64)
Bottleneck Fusion: Conv3d(2048,512,k=(3,1,1)) + BN3d + ReLU ->
                   Conv3d(512,512,k=(3,1,1)) + BN3d + ReLU ->
                   mean pool over T
Skip Aggregation: max (same as LSTM-UNet)
Parameters: ~35M
Batch Size: 4
Epochs: 400
Early Stopping: No
Loss: Focal Loss (alpha=0.25, gamma=2.0)
Optimizer: AdamW (lr=0.01, weight_decay=5e-4)
LR Scheduler: Cosine annealing (eta_min=0.0001)
Cross-Validation: 5-fold stratified
Seed: 42
```

### Hypothesis
If 3D conv matches LSTM, learned temporal filtering is sufficient (no need for recurrence).
If LSTM beats 3D conv, recurrent gating provides unique benefit beyond learned filtering.

### Success Criteria
- [x] Training converges without instability → ✅ Achieved
- [ ] Clear evidence for or against 3D conv as LSTM replacement

### Files
- SLURM script: `scripts/slurm/train_exp008_conv3d_fusion.sh`
- Checkpoints: `outputs/experiments/exp008_conv3d_fusion_fold{0-4}/`

### Results (5-Fold CV)
| Fold | Best IoU | Best Epoch | Best F1 |
|------|----------|------------|---------|
| 0 | 55.56% | 187 | 71.43% |
| 1 | 70.22% | 246 | 82.51% |
| 2 | 68.98% | 264 | 81.64% |
| 3 | 60.78% | 22 | 75.60% |
| 4 | 38.85% | 195 | 55.96% |
| **Mean** | **58.88% ± 11.37%** | - | **73.43% ± 9.63%** |

### Notes
- Highest T=7 baseline (58.88%), slightly above LSTM-7 (57.62%) and Pool-7 (57.75%)
- Learned 3D temporal filtering matches or exceeds recurrence
- Higher variance than Pool-7 (11.37% vs 9.82%)

---

## Experiment 009: ConvLSTM-lite (T=2, Parameter-Matched)

**Experiment ID**: `exp009`
**Date**: 2026-02-03
**Research Question**: RQ0 - Is boundary degradation in exp003 from recurrence or over-parameterisation?
**Status**: ✅ Complete

### Motivation
Exp003 (full ConvLSTM at T=2, ~54M params) showed boundary degradation (lower BF@2) compared to non-temporal baselines (~30M params). This could be caused by:
1. **Over-parameterisation**: Too many parameters for T=2 data causes overfitting/smoothing
2. **Recurrence itself**: ConvLSTM gating is harmful for simple before/after comparison

By reducing ConvLSTM to 1 layer with hidden_dim=32 (~30.4M total params), we match baseline parameter counts and isolate the effect of recurrence vs capacity.

### Configuration
```yaml
Model: LSTM-UNet + ResNet-50 (reduced ConvLSTM)
Encoder: ResNet-50 (ImageNet pretrained, shared across timesteps)
Input Shape: (B=8, T=2, C=9, H=64, W=64)
ConvLSTM: 1 layer, hidden_dim=32 (vs 2 layers, 256 in exp003)
ConvLSTM Kernel Size: 3x3
Skip Aggregation: max
Parameters: ~30.4M (vs ~54.4M in exp003)
Batch Size: 8
Epochs: 400
Early Stopping: No
Loss: Focal Loss (alpha=0.25, gamma=2.0)
Optimizer: AdamW (lr=0.01, weight_decay=5e-4)
LR Scheduler: Cosine annealing (eta_min=0.0001)
Cross-Validation: 5-fold stratified
Seed: 42
```

### Key Comparisons

| Comparison | If LSTM-lite wins | If other wins/ties |
|------------|------------------|-------------------|
| exp009 vs exp003 (full LSTM) | Over-parameterisation caused boundary degradation | Full LSTM needed despite param cost |
| exp009 vs exp005 (Early-Fusion) | Recurrence helps even parameter-matched | Recurrence not helpful |
| exp009 vs exp006 (Late-Fusion) | Recurrence helps even parameter-matched | Recurrence not helpful |

**Key scenario**: If exp009 BF@2 > exp003 BF@2, over-parameterisation caused boundary smoothing, not recurrence itself.

### Success Criteria
- [x] Training converges without instability → ✅ Achieved
- [ ] Parameter count confirmed ~30.4M (matching baselines)
- [ ] Clear evidence isolating boundary degradation cause

### Files
- SLURM script: `scripts/slurm/train_exp009_lstm_lite.sh`
- Checkpoints: `outputs/experiments/exp009_lstm_lite_fold{0-4}/`

### Results (5-Fold CV)
| Fold | Best IoU | Best Epoch | Best F1 |
|------|----------|------------|---------|
| 0 | 51.43% | 129 | 67.92% |
| 1 | 65.98% | 95 | 79.50% |
| 2 | 65.98% | 120 | 79.50% |
| 3 | 58.79% | 84 | 74.04% |
| 4 | 39.17% | 88 | 56.29% |
| **Mean** | **56.27% ± 10.11%** | - | **71.45% ± 8.70%** |

### Notes
- Outperforms full LSTM-2 (56.27% vs 53.29%) with ~44% fewer parameters
- Close to Late-Fusion Concat (54.82%) — recurrence provides modest benefit at T=2
- Suggests over-parameterisation in exp003 was a factor in boundary degradation

---

## Experiment 010: LSTM-7 (Canonical Annual Model)

**Experiment ID**: `exp010`
**Date**: 2026-02-03 (retrained 2026-02-05)
**Research Question**: RQ1 - Multi-temporal LSTM-UNet with annual sampling (T=7)
**Status**: ✅ Complete

### Motivation
This is the canonical LSTM-7 model using the unified training protocol (400 epochs, no early stopping) applied to all temporal conditions. Serves as the primary model for evaluating annual multi-temporal sequences.

### Configuration
```yaml
Model: LSTM-UNet + ResNet-50 (identical to exp001_v2)
Encoder: ResNet-50 (ImageNet pretrained, shared across timesteps)
Input Shape: (B=4, T=7, C=9, H=64, W=64)
ConvLSTM: 2 layers, hidden_dim=256
Skip Aggregation: max
Parameters: ~54.4M (same as exp001_v2)
Batch Size: 4
Epochs: 400 (NO early stopping)
Loss: Focal Loss (alpha=0.25, gamma=2.0)
Optimizer: AdamW (lr=0.01, weight_decay=5e-4)
LR Scheduler: Cosine annealing (eta_min=0.0001)
Cross-Validation: 5-fold stratified
Seed: 42
```

### Key Comparisons
- exp010 vs exp007 (Pool-7): LSTM vs Pool at T=7 with same training budget
- exp010 vs exp008 (Conv3D-7): LSTM vs Conv3D at T=7 with same training budget
- exp010 vs exp011 (LSTM-7-lite): Full vs lite LSTM at T=7

### Files
- SLURM script: `scripts/slurm/train_exp010_lstm7_no_es.sh`
- Checkpoints: `outputs/experiments/exp010_lstm7_no_es_fold{0-4}/`

### Results (5-Fold CV)
| Fold | Best IoU | Best Epoch | Best F1 |
|------|----------|------------|---------|
| 0 | 54.31% | 191 | 70.39% |
| 1 | 65.45% | 230 | 79.12% |
| 2 | 69.37% | 258 | 81.92% |
| 3 | 60.06% | 33 | 75.04% |
| 4 | 42.71% | 330 | 59.85% |
| **Mean** | **58.38% ± 9.34%** | - | **73.26% ± 7.75%** |

### Notes
- Canonical LSTM-7 model with unified 400-epoch training protocol
- Best checkpoints at epochs 33–330
- Mean Val IoU: 58.38% ± 10.44%, Test IoU: 63.9%
- Outperforms Pool-7 (57.75%) and comparable to Conv3D-7 (58.88%)
- Serves as anchor for all temporal sampling comparisons (RQ1) and architecture ablations (RQ2)

---

## Experiment 011: LSTM-7-lite (T=7, Parameter-Matched)

**Experiment ID**: `exp011`
**Date**: 2026-02-03
**Research Question**: RQ2e - Close the parameter gap at T=7, isolating capacity from recurrence
**Status**: ✅ Complete

### Motivation
Both T=7 comparisons (Pool-7 ~30.1M, Conv3D-7 ~32.9M vs LSTM-7 54.4M) have a ~23M parameter gap. LSTM-7-lite reduces the ConvLSTM to 1 layer, h=32 (~30.3M total), matching the baselines. This is the T=7 analogue of LSTM-2-lite (exp009).

### Configuration
```yaml
Model: LSTM-UNet + ResNet-50 (reduced ConvLSTM)
Encoder: ResNet-50 (ImageNet pretrained, shared across timesteps)
Input Shape: (B=4, T=7, C=9, H=64, W=64)
ConvLSTM: 1 layer, hidden_dim=32 (vs 2 layers, 256 in exp001)
ConvLSTM Kernel Size: 3x3
Skip Aggregation: max
Parameters: ~30.3M (matches Pool-7 and Conv3D-7)
Batch Size: 4
Epochs: 400 (NO early stopping)
Loss: Focal Loss (alpha=0.25, gamma=2.0)
Optimizer: AdamW (lr=0.01, weight_decay=5e-4)
LR Scheduler: Cosine annealing (eta_min=0.0001)
Cross-Validation: 5-fold stratified
Seed: 42
```

### Key Comparisons

| Comparison | If LSTM-7-lite wins | If other wins/ties |
|------------|--------------------|--------------------|
| exp011 vs exp007 (Pool-7) | Recurrence helps even param-matched at T=7 | Pooling suffices at T=7 |
| exp011 vs exp008 (Conv3D-7) | Recurrence > 3D conv, param-matched | 3D conv matches/beats recurrence |
| exp011 vs exp010 (full LSTM-7) | Lite sufficient, full LSTM over-parameterised | Full LSTM needed at T=7 |

### Files
- SLURM script: `scripts/slurm/train_exp011_lstm7_lite.sh`
- Checkpoints: `outputs/experiments/exp011_lstm7_lite_fold{0-4}/`

### Results (5-Fold CV)
| Fold | Best IoU | Best Epoch | Best F1 |
|------|----------|------------|---------|
| 0 | 54.67% | 111 | 70.69% |
| 1 | 69.90% | 275 | 82.28% |
| 2 | 69.97% | 192 | 82.33% |
| 3 | 63.96% | 132 | 78.02% |
| 4 | 40.90% | 123 | 58.06% |
| **Mean** | **59.88% ± 11.01%** | - | **74.27% ± 9.15%** |

### Notes
- Highest mean IoU of all T=7 models (59.88% vs 58.88% Conv3D-7, 58.38% LSTM-7 no ES, 57.75% Pool-7)
- Parameter-matched to baselines (~30.3M) — recurrence is not harmful when capacity is controlled
- Outperforms full LSTM-7 (58.38%) with ~44% fewer parameters at T=7
- Confirms the over-parameterisation finding from exp009 (T=2) extends to T=7

---

## Expected Outcome Matrix (exp007-009)

| Comparison | If LSTM wins | If baseline wins/ties |
|------------|-------------|----------------------|
| exp001 vs exp007 (Pool T=7) | Recurrence helps at T=7 | Simple pooling suffices at T=7 |
| exp001 vs exp008 (3D Conv T=7) | Recurrence > learned 3D filtering | 3D conv captures temporal patterns without recurrence |
| exp003 vs exp009 (LSTM-lite T=2) | Full LSTM needed despite param cost | Over-parameterisation caused boundary degradation |
| exp009 vs exp005/006 | Recurrence helps even parameter-matched | Recurrence still not helpful |

---

## Expected Outcome Matrix (exp005/exp006)

| Comparison | If LSTM-UNet wins | If Baseline wins |
|------------|-------------------|------------------|
| exp003 vs exp005 (early-fusion) | Temporal architecture matters | Stacked channels sufficient |
| exp003 vs exp006 (late-fusion) | Recurrence helps | Simple concat sufficient |
| exp005 vs exp006 | Shared encoder helps | Early fusion competitive |

### Interpretation Guide

**Best case for LSTM**: exp003 >> exp006 >> exp005
→ "Both temporal modeling AND recurrence are valuable"

**Worst case for LSTM**: exp003 ≈ exp006 ≈ exp005
→ "Temporal modeling adds no value; simpler models suffice"

**Middle ground**: exp003 ≈ exp006 > exp005
→ "Shared encoder matters, but recurrence doesn't add much"

---

## Experiment Template (Copy for New Experiments)

```markdown
## Experiment XXX: [Name]

**Experiment ID**: `expXXX`
**Date**: YYYY-MM-DD
**Research Question**: RQX - [Question]
**Status**: ⬜ Planned / 🔄 Running / ✅ Complete / ❌ Failed

### Configuration
```yaml
Model:
Data Source:
Resolution:
Time Steps:
Input Shape:
Batch Size:
Epochs:
Loss:
Optimizer:
Seeds:
```

### Expected Results
```
Expected Test IoU:
Training time:
```

### Hypothesis
[What you expect and why]

### Success Criteria
- [ ] Criterion 1
- [ ] Criterion 2

### Files
- Training: `outputs/experiments/expXXX_*/`

### Results
```
TBD
```

### Notes
```
TBD
```
```

---

## Experiment Naming Convention

```
exp[NNN]_[model]_[sampling]_seed[XXX]

Examples:
- exp001_lstm_annual_seed42
- exp002_lstm_quarterly_seed123
- exp003_lstm_bitemporal_seed456

```

---

## Status Codes

- ⬜ **Planned**: Not yet started
- 🔄 **Running**: Currently training
- ✅ **Complete**: Finished successfully
- ❌ **Failed**: Training failed or results invalid
- ⚠️ **Partial**: Completed but with issues

---

## Quick Reference: Research Questions

**RQ1**: Multi-temporal vs bi-temporal performance
→ Compare baseline (bi-temporal VHR) vs LSTM (annual/quarterly S2)

**RQ2**: Temporal sampling density effects
→ Compare bi-temporal vs annual vs bi-seasonal using same architecture

**RQ3**: Temporal modeling paradigms
→ Compare 1D vs 2D

---

## Update Log

| Date | Experiment | Action | Notes |
|------|-----------|--------|-------|
| 2026-01-13 | - | Created experiment tracking system | Initial setup |
| 2026-01-14 | exp001 | Pilot run complete (seed 42) | Val IoU: 39.62%, 12min training time |
| 2026-01-14 | exp001 | Fixed NaN issues | Device mismatch + nodata handling |
| 2026-01-14 | exp001 | Pipeline validated | All components functional |
| 2026-01-19 | exp002 | Initial run complete | 49.87% IoU (batch_size=2 confound) |
| 2026-01-19 | exp003 | Complete | 53.29% IoU, -4.33 pp vs exp001 (n.s.) |
| 2026-01-20 | exp002 | Re-run with gradient accumulation | 48.06% IoU, -9.57 pp vs exp001 (p<0.05) |
| 2026-01-27 | exp004 | 1×1 kernel ablation complete | 41.19% IoU, no diff vs 3×3 (p=0.897) |
| 2026-01-27 | all | Per-sample statistical analysis | Paired permutation tests, n=45 tiles |
| 2026-01-27 | all | Boundary F-score analysis | BF@2 confirms temporal findings |
| 2026-02-02 | exp005 | Created early-fusion U-Net baseline | No temporal modeling (stacked channels) |
| 2026-02-02 | exp006 | Created late-fusion concat baseline | Concat + 1×1 conv vs LSTM recurrence |
| 2026-02-03 | exp007 | Created late-fusion pool (T=7) | Pool vs LSTM at T=7 |
| 2026-02-03 | exp008 | Created 3D conv fusion (T=7) | Learned temporal filtering vs recurrence |
| 2026-02-03 | exp009 | Created ConvLSTM-lite (T=2) | Parameter-matched LSTM to isolate boundary degradation cause |
| 2026-02-03 | exp010 | Created LSTM-7 no ES (T=7) | LSTM-7 retrained without early stopping for fair T=7 comparison |
| 2026-02-03 | exp011 | Created LSTM-7-lite (T=7) | Parameter-matched LSTM at T=7 (~30.3M), analogous to exp009 |
| 2026-02-03 | analysis | Updated test eval + stats scripts | Added exp010/011 to evaluate_test_final, statistical_analysis, boundary_f_score |
| 2026-02-05 | all | Unified training protocol | Retrained exp002→v3, exp003→v3, exp004→v2 with 400 epochs, no ES |
| 2026-02-05 | all | Canonical naming update | exp010 is now THE LSTM-7; removed early/late version complexity |
