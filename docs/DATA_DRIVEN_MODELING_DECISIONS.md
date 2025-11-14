# Data-Driven Modeling Decisions

This document connects findings from our comprehensive data analysis to specific modeling decisions implemented in the baseline models.

## Data Analysis Summary

### Dataset Characteristics
- **53 tiles** validated across 5 data sources
- **14.65% average change ratio** (class imbalance: 1:6)
- **Patch size range**: 24 pixels (median) to 4,779 pixels (max) - **4 orders of magnitude**
- **Tile categories**: 0 zero-change, 17 low (<5%), 30 moderate (5-30%), 6 high (≥30%)
- **Resolution mismatch**: VHR (1m, ~650×900px) vs Masks (10m, ~65×90px)
- **Perfect spatial alignment**: All sources aligned in EPSG:4326
- **Excellent data quality**: 0-0.31% NoData across sources

---

## Modeling Decisions Based on Data Analysis

### 1. Dataset Implementation (scripts/modeling/dataset.py)

#### Decision: Resample Masks to Match VHR Resolution
**Data Finding:**
- VHR images: ~655×913 pixels at 1m resolution
- Masks: ~66×92 pixels at 10m resolution
- Perfect spatial alignment verified (docs/DATA_ANALYSIS_SUMMARY.md:89-92)

**Implementation:**
```python
# dataset.py:105-131
def _load_mask(self, refid: str, target_shape: tuple = None):
    # Resample mask to match VHR resolution using nearest-neighbor
    if target_shape is not None:
        zoom_factors = (target_shape[0] / mask.shape[0],
                       target_shape[1] / mask.shape[1])
        mask = zoom(mask, zoom_factors, order=0)  # Preserve binary
```

**Rationale:**
- Allows augmentation pipeline to work (albumentations requires matching dimensions)
- Preserves fine-grained VHR details (1m resolution)
- Nearest-neighbor interpolation maintains binary mask integrity

---

#### Decision: Resize All Images to 512×512
**Data Finding:**
- Variable tile sizes: 655×913, 660×1042, 657×890, etc.
- Cannot batch variable-sized tensors in PyTorch

**Implementation:**
```python
# dataset.py:205-206
A.Resize(height=image_size, width=image_size, interpolation=1, p=1.0)
```

**Rationale:**
- Enables batching for efficient training
- 512×512 balances memory usage and spatial context
- Covers median patch size (24 pixels) with ample context
- Can adjust to 256×256 or 1024×1024 based on GPU memory

---

### 2. Data Splits (scripts/modeling/01_create_splits.py)

#### Decision: Stratified Split by Change Ratio
**Data Finding:**
- Change ratio distribution: 17 low, 30 moderate, 6 high (docs/DATA_ANALYSIS_SUMMARY.md:173-177)
- No zero-change tiles (all tiles have signal)
- Geographic diversity: 20 countries

**Implementation:**
```python
# 01_create_splits.py - Stratified split
# 70/15/15 → 37 train / 8 val / 8 test
# Stratified by change_ratio categories
```

**Rationale:**
- Maintains change distribution across train/val/test
- Ensures validation set represents all difficulty levels
- Prevents overfitting to specific change levels
- Tile-level split prevents spatial leakage

---

### 3. Loss Function Selection

#### Decision: Use Focal Loss (Primary) or Dice Loss (Alternative)
**Data Finding:**
- **14.65% change pixels** → 1:6 imbalance ratio (docs/DATA_ANALYSIS_SUMMARY.md:158-162)
- Standard accuracy misleading: predicting all "no-change" = 85.35% accuracy
- Recommended in data analysis: Focal Loss or Dice Loss (docs/DATA_ANALYSIS_SUMMARY.md:315-321)

**Planned Implementation:**
```python
# Focal Loss configuration
alpha = 0.25  # Down-weight easy negatives
gamma = 2.0   # Focus on hard examples

# Alternative: Dice Loss for direct IoU optimization
```

**Rationale:**
- Focal Loss handles class imbalance by down-weighting easy examples
- Alpha=0.25 balances positive/negative class weights
- Gamma=2.0 focuses learning on hard misclassified pixels
- Dice Loss directly optimizes for IoU metric

---

### 4. Evaluation Metrics

#### Decision: F1-Score (Primary), IoU, Precision, Recall (NOT Accuracy)
**Data Finding:**
- Class imbalance makes accuracy misleading (docs/DATA_ANALYSIS_SUMMARY.md:315-321)
- Data analysis explicitly recommends F1/IoU (docs/DATA_ANALYSIS_SUMMARY.md:182)

**Planned Implementation:**
```python
# Evaluation metrics
metrics = {
    'f1_score': F1Score(task='binary'),
    'iou': JaccardIndex(task='binary'),
    'precision': Precision(task='binary'),
    'recall': Recall(task='binary'),
}
# Accuracy reported for reference but NOT primary metric
```

**Rationale:**
- F1-score balances precision and recall
- IoU directly measures spatial overlap (standard for segmentation)
- Precision: minimize false alarms
- Recall: ensure change detection completeness

---

### 5. Architecture: U-Net with Skip Connections

#### Decision: U-Net Architecture with Multi-Scale Features
**Data Finding:**
- **Multi-scale challenge**: Patches span 24 to 4,779 pixels (docs/DATA_ANALYSIS_SUMMARY.md:166-170)
- 4 orders of magnitude variation requires multi-scale features
- Data analysis recommendation: U-Net well-suited (docs/DATA_ANALYSIS_SUMMARY.md:322-327)

**Implementation:**
```python
# models.py:30-65
# U-Net with ResNet-50 encoder
encoder_name='resnet50',
encoder_weights='imagenet',  # Pretrained features
# Skip connections preserve multi-scale information
```

**Rationale:**
- Skip connections preserve fine details for small patches (24 pixels)
- Encoder captures large-scale context for big patches (4,779 pixels)
- ResNet-50 pretrained weights provide strong feature extraction
- Proven top performer on change detection benchmarks

---

### 6. Data Augmentation

#### Decision: Geometric + Photometric Augmentations (No Hue/Saturation)
**Data Finding:**
- 6-channel bi-temporal images (2018 RGB + 2025 RGB)
- RGB values well-exposed: 85.7/255 (2018), 90.1/255 (2025) (docs/DATA_ANALYSIS_SUMMARY.md:120-122)

**Implementation:**
```python
# dataset.py:194-234
# Geometric: Flip, Rotate, ShiftScaleRotate
# Photometric: BrightnessContrast, Gamma, GaussNoise, Blur
# EXCLUDED: HueSaturationValue (requires 3 channels, we have 6)
```

**Rationale:**
- Geometric transforms preserve temporal relationships
- Photometric transforms simulate lighting variations
- Excludes color space transforms incompatible with 6-channel input
- Augmentation reduces overfitting on small dataset (53 tiles)

---

### 7. Input Data Selection

#### Decision: Start with VHR 6-Channel (2018+2025 RGB)
**Data Finding:**
- VHR: 1m resolution, 0.04% NoData, excellent quality (docs/DATA_ANALYSIS_SUMMARY.md:118-122)
- Highest resolution available (can detect 24-pixel patches)
- Temporal signal strength: All 53 tiles show measurable change (docs/DATA_ANALYSIS_SUMMARY.md:329-334)

**Implementation:**
```python
# dataset.py:89-103
# Load VHR 6 bands: [2018_R, 2018_G, 2018_B, 2025_R, 2025_G, 2025_B]
img_2018 = data[0:3, :, :].transpose(1, 2, 0)
img_2025 = data[3:6, :, :].transpose(1, 2, 0)
```

**Rationale:**
- Highest resolution → best small patch detection
- Bi-temporal input captures change explicitly
- Strong baseline before exploring multi-temporal Sentinel-2
- Can later add Sentinel-2 multispectral bands or AlphaEarth embeddings

---

### 8. Batch Size and Training Configuration

#### Decision: Batch Size = 4-8, Based on GPU Memory
**Data Finding:**
- 53 tiles → 37 train, 8 val, 8 test
- After resize: 512×512×6 channels per image

**Implementation:**
```python
# Batch size: 4 (tested in test_dataset.py)
# 37 train tiles → ~10 batches per epoch
# Memory: ~4 × 6 × 512 × 512 × 4 bytes = ~25 MB per batch (images only)
```

**Rationale:**
- Small dataset (53 tiles) limits batch size benefit
- Batch size 4-8 provides stable gradients
- Enables training on single GPU (16GB+)
- Can increase with gradient accumulation if needed

---

### 9. Training Duration

#### Decision: 200 Epochs with Early Stopping
**Data Finding:**
- Small dataset (37 training tiles)
- High-quality data (0% corrupted tiles)

**Planned Implementation:**
```python
# 200 epochs sufficient for convergence
# Early stopping: patience=20 (if val loss doesn't improve)
# Save best checkpoint by validation IoU
```

**Rationale:**
- Small dataset requires more epochs to converge
- Early stopping prevents overfitting
- 200 epochs empirically effective in change detection (Corley et al., 2024)

---

### 10. Separate Images Mode for Siamese Architectures

#### Decision: Support Both Concatenated and Separate Image Modes
**Data Finding:**
- Siamese architectures process temporal images separately
- Early fusion concatenates before encoder

**Implementation:**
```python
# dataset.py:175-179
if self.return_separate_images:
    output['image_2018'] = img_2018  # (3, H, W)
    output['image_2025'] = img_2025  # (3, H, W)
else:
    output['image'] = img_concat      # (6, H, W)
```

**Rationale:**
- Early Fusion: Concatenated 6-channel input
- SiamDiff/SiamConc: Separate 3-channel inputs
- Single dataset class supports all three baseline models

---

## Validation of Decisions

### Dataset Loading Test Results
Ran `scripts/modeling/test_dataset.py` to verify data-driven decisions:

✅ **Mask Resampling Works**:
- VHR: torch.Size([6, 655, 913])
- Mask: torch.Size([655, 913]) ← upsampled from (66, 92)
- Spatial alignment preserved

✅ **Resizing Enables Batching**:
- After augmentation: torch.Size([6, 512, 512])
- Batch shape: torch.Size([4, 6, 512, 512])
- All tiles fit in batches

✅ **Value Ranges Correct**:
- Images: [0.0, 1.0] (normalized from uint8)
- Masks: {0.0, 1.0} (binary)

✅ **DataLoader Works**:
- Train: 37 tiles → 10 batches (batch_size=4)
- Val: 8 tiles → 2 batches
- Test: 8 tiles → 2 batches

---

## Summary: Data Analysis → Modeling Pipeline

| Data Finding | Modeling Decision | Implementation |
|-------------|------------------|----------------|
| 14.65% change ratio | Focal Loss / Dice Loss | Training script (planned) |
| 1:6 class imbalance | F1/IoU metrics (not accuracy) | Evaluation (planned) |
| 24-4,779 pixel patches | U-Net multi-scale architecture | models.py:30-65 |
| VHR 1m resolution | Use VHR as primary input | dataset.py:89-103 |
| Resolution mismatch | Resample masks to VHR | dataset.py:105-131 |
| Variable tile sizes | Resize to 512×512 | dataset.py:205-206 |
| Change distribution | Stratified train/val/test split | 01_create_splits.py |
| 53 tiles (small dataset) | Heavy augmentation | dataset.py:194-234 |
| 6-channel bi-temporal | No Hue/Saturation transforms | dataset.py:220-225 |
| All tiles have signal | No zero-change filtering | All 53 tiles used |

---

## Next Steps Based on Data Analysis

From the data analysis, the immediate next steps are clear:

1. ✅ **Dataset loading verified** → Ready for model testing
2. 🔄 **Test forward pass with models** → Verify shapes and memory
3. ⏳ **Implement training script** → Focal Loss, SGD optimizer, F1/IoU metrics
4. ⏳ **Train baseline U-Net** → VHR input, stratified splits
5. ⏳ **Analyze results by change level** → Low/moderate/high performance

The data analysis has provided concrete guidance for every modeling decision, ensuring our approach is data-driven rather than arbitrary.

---

## References

- **Data Analysis Summary**: docs/DATA_ANALYSIS_SUMMARY.md
- **Mask Analysis Script**: scripts/analysis/06_analyze_masks.py
- **Edge Cases Analysis**: scripts/analysis/07_identify_edge_cases.py
- **Dataset Implementation**: scripts/modeling/dataset.py
- **Baseline Models Doc**: docs/BASELINE_MODELS.md
- **Reality Check Paper**: Corley et al. (2024) - docs/2402.06994v2-2.pdf
