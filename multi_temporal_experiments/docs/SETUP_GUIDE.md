# Multi-Temporal Experiments - Setup Guide

**Author**: tmstorma@stud.ntnu.no
**Date**: January 2026

---

## Prerequisites

### 1. Baseline Results
Ensure you have completed the baseline bi-temporal experiments:
- ✅ SiamConc + ResNet-50 trained (68.37% IoU)
- ✅ Test set evaluation complete
- ✅ Splits are frozen (train/val/test)

### 2. Environment
Same conda environment as baseline:
```bash
conda activate landtake_env
```

**Required packages** (should already be installed):
- PyTorch ≥ 2.0
- torchvision
- rasterio
- numpy
- pandas
- albumentations
- wandb
- tqdm
- matplotlib
- seaborn

### 3. Data Access
Ensure you have access to:
- ✅ Sentinel-2 data: `data/Sentinel/`
- ✅ Land-take masks: `data/Land_take_masks/`
- ✅ GeoJSON with tile metadata: `land_take_bboxes_650m_v1.geojson`

### 4. Computational Resources
- GPU: 80GB A100/H100 (confirmed available ✅)
- RAM: 64-128GB recommended
- Storage: ~100GB free space

---

## Setup Steps

### Step 1: Validate Sentinel-2 Temporal Data

**Purpose**: Check data quality for all time steps (cloud coverage, gaps, NoData)

```bash
cd multi_temporal_experiments

# Run validation script (to be created)
python scripts/data_preparation/01_validate_sentinel2_temporal.py \
    --output-dir outputs/reports \
    --check-clouds \
    --visualize-sample
```

**Expected outputs**:
- `outputs/reports/sentinel2_temporal_quality.csv` - Per-tile, per-quarter quality metrics
- `outputs/reports/sentinel2_temporal_summary.txt` - Summary statistics
- `outputs/figures/temporal_sequences/` - Example visualizations

**Success criteria**:
- All 53 tiles have complete Sentinel-2 data (126 bands)
- Average NoData < 5% per quarter
- No completely missing quarters
- Cloud coverage acceptable (<20% per quarter)

---

### Step 2: Compute Normalization Statistics

**Purpose**: Calculate mean and std per band from training set for z-score normalization

```bash
python scripts/data_preparation/03_compute_normalization_stats.py \
    --split train \
    --output outputs/reports/normalization_stats.csv
```

**Expected output**:
```csv
band,mean,std,min,max
blue,1234.56,567.89,0,9876
green,1456.78,678.90,0,9543
...
```

**This file will be used** by all training scripts for consistent normalization.

---

### Step 3: Test Data Loading

**Purpose**: Verify MultiTemporalSentinel2Dataset class works correctly

```bash
# Interactive test (Jupyter notebook recommended)
jupyter notebook notebooks/02_test_data_loading.ipynb
```

Or run test script:
```bash
python scripts/data_preparation/dataset_multitemporal.py --test
```

**What to verify**:
- [ ] Data loads without errors
- [ ] Correct shapes: (T, C, H, W) for LSTM or (C, T, H, W) for 3D
- [ ] Normalization applied correctly (values ≈ N(0,1))
- [ ] Augmentations work across time steps
- [ ] Batch loading is reasonably fast (<2 sec/batch)

---

### Step 4: Memory Profiling

**Purpose**: Determine optimal batch size for each model architecture

```bash
python scripts/modeling/profile_memory.py \
    --model lstm_unet \
    --temporal-sampling quarterly \
    --image-size 512 \
    --test-batch-sizes 1,2,4,8
```

**Expected output**:
```
Testing lstm_unet with quarterly sampling (14 time steps)
Batch size 1: 12.3 GB ✓
Batch size 2: 23.8 GB ✓
Batch size 4: 46.9 GB ✓
Batch size 8: 92.1 GB ✗ (OOM)

Recommended batch size: 4
```

Repeat for each architecture:
- `lstm_unet` (expected: batch_size=4-8)
- `unet_3d` (expected: batch_size=2-4)
- `hybrid_lstm_3d` (expected: batch_size=2-4)

---

### Step 5: Baseline Reproduction Test

**Purpose**: Verify your environment can reproduce baseline results

```bash
# Use existing baseline code to ensure environment is stable
cd ..  # Back to main repo
python scripts/modeling/evaluate.py \
    --checkpoint outputs/training/siam_conc_resnet50_seed42/best_model.pth \
    --output-dir multi_temporal_experiments/outputs/reports/baseline_check \
    --batch-size 4
```

**Success criteria**:
- Evaluation runs without errors
- Test IoU matches expected: 68.04% (± 0.5%)
- Takes reasonable time (~5 minutes)

If this fails, debug environment before proceeding with new experiments.

---

## Configuration

### Edit `config.py` (if needed)

The default configuration should work, but you may want to adjust:

```python
# Temporal sampling (default: annual)
DEFAULT_TEMPORAL_SAMPLING = "annual"  # or "quarterly", "bi_temporal"

# Normalization (default: z-score)
NORMALIZATION_MODE = "zscore"  # or "minmax", "per_tile"

# Training hyperparameters
MT_DEFAULT_HYPERPARAMS = {
    "batch_size": 4,  # May need to reduce for 3D models
    "image_size": 512,  # Or 256 if memory constrained
    "epochs": 200,
    "learning_rate": 0.01,
    # ...
}

# Wandb tracking
WANDB_PROJECT = "landtake-multitemporal"
WANDB_ENTITY = "your_username"  # ← UPDATE THIS
```

---

## Verification Checklist

Before starting experiments, verify:

### Data
- [ ] Sentinel-2 temporal validation complete (`01_validate_sentinel2_temporal.py`)
- [ ] Normalization statistics computed (`normalization_stats.csv` exists)
- [ ] Data loading tested (no errors, correct shapes)
- [ ] Example temporal sequences visualized

### Environment
- [ ] Conda environment activated
- [ ] All packages installed (import test passes)
- [ ] GPU accessible (nvidia-smi shows 80GB A100/H100)
- [ ] Baseline evaluation works

### Code
- [ ] `MultiTemporalSentinel2Dataset` class implemented
- [ ] `LSTM-UNet` model architecture implemented
- [ ] Training script adapted for multi-temporal input
- [ ] Evaluation script handles multi-temporal models

### Resources
- [ ] Memory profiling complete (know max batch sizes)
- [ ] SLURM access confirmed
- [ ] Storage space available (~100GB)

---

## Troubleshooting

### Issue: OOM (Out of Memory)
**Solutions**:
1. Reduce batch size
2. Reduce image size (512 → 256)
3. Enable gradient checkpointing
4. Use mixed precision (FP16)
5. Reduce number of time steps (quarterly → annual)

### Issue: Data loading slow
**Solutions**:
1. Increase num_workers (4 → 8)
2. Use SSD storage if available
3. Cache normalized data to disk
4. Reduce image size

### Issue: Training not converging
**Solutions**:
1. Check normalization (values should be ~N(0,1))
2. Verify loss function (Focal Loss with α=0.25, γ=2)
3. Check learning rate (may need adjustment for new architecture)
4. Visualize predictions during training (sanity check)

### Issue: Results worse than baseline
**Possible causes**:
1. Normalization incorrect (Sentinel-2 ≠ VHR normalization)
2. Model architecture issue (check forward pass)
3. Data quality (clouds, gaps)
4. Hyperparameters need tuning

---

## Next Steps

Once setup is complete:

1. **Week 1**: Run Experiment 001 (LSTM-UNet annual)
2. **Week 2**: Analyze results, adjust if needed
3. **Week 3**: Run Experiment 002 (LSTM-UNet quarterly)
4. **Week 4+**: Continue with RQ2 and RQ3 experiments

See `EXPERIMENT_LOG.md` for detailed experiment tracking.

---

## Contact

**Questions?** tmstorma@stud.ntnu.no

**Data issues?** zander.venter@nina.no

**SLURM help?** NTNU HPC support
