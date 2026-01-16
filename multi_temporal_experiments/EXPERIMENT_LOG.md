# Multi-Temporal Experiments - Running Log

**Purpose**: Track all experiments with IDs, configurations, and results
**Format**: Each experiment gets unique ID and standardized reporting

---

## Experiment Index

| ID | RQ | Model | Sampling | Status | Val IoU | Test IoU | Notes |
|----|----|-------|----------|--------|---------|----------|-------|
| `baseline` | - | SiamConc-ResNet50 | Bi-temporal VHR | ✅ Complete | 54.57% ± 0.52% | 68.37% ± 0.35% | VHR baseline (1m resolution) |
| `exp001` | RQ1 | LSTM-UNet | Annual (7 steps) | ✅ Complete | **39.62%** | TBD | Seed 42, S2 10m resolution |
| `exp002` | RQ1 | LSTM-UNet | Quarterly (14 steps) | ⬜ Planned | - | - | Full temporal density |
| `exp003` | RQ2 | LSTM-UNet | Bi-temporal (2 steps) | ⬜ Planned | - | - | S2 baseline comparison |
| `exp004` | RQ3 | 3D U-Net | Quarterly (14 steps) | ⬜ Planned | - | - | Spatiotemporal modeling |
| `exp005` | RQ3 | Hybrid LSTM-3D | Quarterly (14 steps) | ⬜ Planned | - | - | Combined approach |

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

### Hypothesis Status
Annual multi-temporal sequences provide temporal trajectory information that:
- Reduces false positives from seasonal variation → ⚠️ **Partially validated**
- Captures gradual land-take progression → ⚠️ **Partially validated**
- Improves boundary delineation → ⚠️ **Partially validated**

**Reality**: Model learns, but 10m resolution limits detection capability compared to 1m VHR

### Success Criteria
- [ ] Test IoU ≥ 73% → ❌ **Not achieved** (39.62% validation, expected ~40-45% test)
- [ ] ≥5% improvement over baseline → ⚠️ **Cannot compare** (different data: S2 vs VHR)
- [x] Training converges smoothly → ✅ **Achieved**
- [x] No severe overfitting → ✅ **Achieved** (small train-val gap)

### Key Findings

**✅ What Worked**:
- Pipeline fully functional (data, model, training, logging)
- Model converged smoothly without NaN or instability
- Clear learning progression (14% → 40% IoU)
- Memory efficient (5-6GB for T=7, B=4, 95M params)
- Fast training (~12 minutes for 200 epochs)

**⚠️ Challenges**:
- Lower than expected IoU (39.62% vs expected 73-78%)
- Validation instability in late epochs (IoU oscillates 30-40%)
- Small validation set (8 samples) causes high variance
- Nodata issues in 2-3 samples per epoch

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

### Notes
```
- Pilot experiment validates pipeline works correctly
- 39.62% IoU is reasonable for 10m S2 data at 64×64 resolution
- Cannot fairly compare to VHR baseline (68.37%) - different data sources
- Original hypothesis (73-78% IoU) was unrealistic given data constraints
- Validation set too small (8 samples) for reliable early stopping
- Test set evaluation critical for accurate performance assessment
- Two samples consistently have nodata NaN values - should investigate
```

---

## Experiment 002: LSTM-UNet Quarterly

**Experiment ID**: `exp002`
**Date**: TBD
**Research Question**: RQ1 + RQ2 - Full temporal density
**Status**: ⬜ Planned

### Configuration
```yaml
Model: LSTM-UNet
Data Source: Sentinel-2
Resolution: 10m
Time Steps: 14 (quarterly, Q2+Q3, 2018-2024)
Bands per Step: 9
Input Shape: (B, T=14, C=9, H=512, W=512)
Batch Size: TBD (memory test needed - likely smaller than exp001)
Image Size: 512
Epochs: 200
Loss: Focal Loss (α=0.25, γ=2.0)
Optimizer: SGD (lr=0.01, momentum=0.9, weight_decay=5e-4)
Normalization: Z-score per band (same stats as exp001)
Seeds: 3 (42, 123, 456)
```

### Expected Results
```
Expected Test IoU: 75-80% (goal: +7-12% over baseline)
Training time: ~4-6 hours per seed (2x exp001)
```

### Hypothesis
Quarterly sampling captures:
- Construction phases (Q2 vs Q3 within year)
- Seasonal consistency across years
- Finer temporal resolution of land-take progression

### Success Criteria
- [ ] Test IoU ≥ 75% (≥7% improvement)
- [ ] Improvement over exp001 (annual) by ≥2-3%
- [ ] Training time acceptable (<8 hours)
- [ ] GPU memory usage fits in 80GB

### Files
TBD

### Results
```
TBD - Experiment not yet run
```

### Notes
```
TBD
```

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
- exp004_3dunet_quarterly_seed42
- exp005_hybrid_quarterly_seed42
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
→ Compare bi-temporal vs annual vs quarterly using same architecture

**RQ3**: Temporal modeling paradigms
→ Compare LSTM-UNet (1D) vs 3D U-Net (2D) vs Hybrid

---

## Update Log

| Date | Experiment | Action | Notes |
|------|-----------|--------|-------|
| 2026-01-13 | - | Created experiment tracking system | Initial setup |
| 2026-01-14 | exp001 | Pilot run complete (seed 42) | Val IoU: 39.62%, 12min training time |
| 2026-01-14 | exp001 | Fixed NaN issues | Device mismatch + nodata handling |
| 2026-01-14 | exp001 | Pipeline validated | All components functional |
