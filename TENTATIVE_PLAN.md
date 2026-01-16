# TENTATIVE RESEARCH PLAN
# Multi-Temporal Deep Learning for Land-Take Detection

**Author**: tmstorma@stud.ntnu.no
**Project**: NINA Fordypningsoppgave
**Date**: January 2026
**Status**: TENTATIVE - Research Questions and Experimental Design

---

## RESEARCH QUESTIONS

This research plan addresses three core questions about multi-temporal approaches to land-take detection:

### RQ1: Multi-Temporal vs Bi-Temporal Performance

**Do multi-temporal image sequences covering 2018–2024 improve land take detection and boundary delineation compared with bi-temporal image pairs (2018, 2024)?**

**Hypothesis**: Multi-temporal sequences provide temporal trajectory information that reduces false positives from seasonal variation and improves change boundary delineation.

---

### RQ2: Temporal Sampling Density Effects

**Given the same overall time span, how does temporal sampling density (e.g., quarterly versus annual composites) affect land take segmentation performance and boundary quality?**

**Hypothesis**: Performance improves with temporal density up to a saturation point, with diminishing returns beyond 5-7 time steps due to temporal redundancy.

---

### RQ3: Temporal Modeling Paradigms

**For multi-temporal data, how does modelling time as per-pixel 1D temporal sequences, compared to 2D spatiotemporal patch-based features, influence segmentation accuracy and robustness?**

**Hypothesis**: 1D temporal models (LSTM) excel at temporal trajectory learning and noise robustness, while 2D spatiotemporal models (3D CNN) excel at boundary delineation through joint spatial-temporal processing. Hybrid approaches may achieve best overall performance.

---

## CURRENT BASELINE

### Existing Results (Completed Work)

**Model**: SiamConc + ResNet-50
**Data**: VHR Google bi-temporal (2018 RGB + 2025 RGB, 1m resolution)
**Performance** (3-seed average on test set):

| Metric | Mean ± Std | Range |
|--------|-----------|-------|
| **IoU** | **68.37% ± 0.35%** | 68.04%-68.74% |
| **F1-Score** | **81.22% ± 0.25%** | 80.98%-81.48% |
| **Precision** | **77.98% ± 2.82%** | 75.72%-81.14% |
| **Recall** | **84.87% ± 2.72%** | 81.81%-87.03% |

**Key Observations**:
- Excellent generalization: Test IoU (68.37%) >> Validation IoU (54.57%)
- High reproducibility: CV = 0.51% across random seeds
- Strong recall but moderate precision (some false positives)
- Current approach: bi-temporal only, no temporal sequence modeling

---

## AVAILABLE DATA

### Multi-Source Remote Sensing Dataset

**Geographic Coverage**: 53 validated tiles across 20 European countries
**Tile Size**: 650m × 650m
**Time Span**: 2018-2024
**Projection**: EPSG:4326 (WGS 84)

| Data Source | Resolution | Bands | Temporal Coverage | Status |
|-------------|-----------|-------|-------------------|--------|
| **VHR Google** | 1m | 6 | 2018 + 2025 RGB | Currently used |
| **Sentinel-2** | 10m | 126 | 2018-2024 Q2+Q3 (14 steps) | Available, unused |
| **PlanetScope** | 3-5m | 42 | 2018-2024 Q2+Q3 (14 steps) | Available, unused |
| **AlphaEarth** | 10m | 448 | 2018-2024 annual (7 steps) | Available, unused |
| **Land-Take Masks** | 10m | 1 | Binary change labels | Ground truth |

**Key Insight**: We have rich multi-temporal data (Sentinel-2: 126 bands, PlanetScope: 42 bands) that is currently unused. This data is perfect for answering our research questions.

### Sentinel-2 Band Structure

**126 total bands** = 7 years × 2 quarters × 9 spectral bands

**Spectral bands per quarter**:
- blue, green, red (visible)
- R1, R2, R3 (red edge)
- nir (near infrared)
- swir1, swir2 (shortwave infrared)

**Temporal sampling**:
- **Quarterly**: Q2 (Apr-Jun) + Q3 (Jul-Sep) = 14 time steps
- **Annual**: Average Q2+Q3 per year = 7 time steps
- **Bi-annual**: 2018, 2020, 2022, 2024 = 4 time steps
- **Bi-temporal**: 2018, 2024 = 2 time steps

---

## EXPERIMENTAL DESIGN

### Complete Experimental Matrix

```
┌─────────────────────────────────────────────────────────────────┐
│              EXPERIMENTAL DESIGN MATRIX                         │
└─────────────────────────────────────────────────────────────────┘

DIMENSION 1: Data Source (RQ1)
├─ VHR Google (1m, bi-temporal 2018/2025) ← Current baseline
├─ Sentinel-2 (10m, multi-temporal 2018-2024)
├─ PlanetScope (3-5m, multi-temporal 2018-2024)
└─ Hybrid (VHR + Sentinel-2 fusion)

DIMENSION 2: Temporal Sampling (RQ2)
├─ Bi-temporal (2 time steps)
├─ Bi-annual (4 time steps)
├─ Annual (7 time steps)
└─ Quarterly (14 time steps)

DIMENSION 3: Architecture (RQ3)
├─ 1D Temporal Models:
│   ├─ Pixel-LSTM-UNet
│   ├─ Temporal Transformer
│   └─ 1D-CNN-UNet
├─ 2D Spatiotemporal Models:
│   ├─ 3D U-Net
│   ├─ (2+1)D U-Net
│   └─ Video Transformer
└─ Hybrid Models:
    ├─ LSTM → 3D decoder
    └─ Attention over time + 2D spatial encoder

TOTAL EXPERIMENTS: ~40-50 model variants
```

---

## RESEARCH QUESTION 1: MULTI-TEMPORAL VS BI-TEMPORAL

### Experimental Comparison

| Experiment | Input Data | Architecture | Expected IoU |
|------------|-----------|--------------|--------------|
| **Baseline (current)** | VHR 2018+2025 RGB (6 bands) | SiamConc + ResNet-50 | 68.37% |
| **Multi-temporal S2** | Sentinel-2 2018-2024 (126 bands) | LSTM-UNet | 73-78% (predicted) |
| **Multi-temporal PS** | PlanetScope 2018-2024 (42 bands) | LSTM-UNet | 70-75% (predicted) |
| **Hybrid** | VHR 2018/2025 + S2 time series | Multi-stream fusion | 75-80% (predicted) |

### Why Multi-Temporal Should Help

1. **Trajectory Analysis**: Distinguishes gradual change from abrupt events
2. **Seasonal Robustness**: Reduces false positives from vegetation phenology
3. **Construction Detection**: Captures intermediate construction phases
4. **Temporal Context**: "Before → During → After" > "Before → After"

### Key Architectures to Test

#### A. LSTM-UNet (1D Temporal)
```
Input: (B, T, H, W, C) - T time steps
   ↓
Per-pixel LSTM over time dimension
   ↓
LSTM features → U-Net decoder
   ↓
Output: (B, H, W) - Change mask
```

**Pros**: Explicitly models temporal trajectory, efficient for long sequences
**Cons**: Ignores spatial context within each time step

#### B. 3D U-Net (2D Spatiotemporal)
```
Input: (B, C, T, H, W) - Spatiotemporal volume
   ↓
3D convolutions (space + time)
   ↓
3D decoder with skip connections
   ↓
Output: (B, H, W) - Change mask
```

**Pros**: Joint spatiotemporal features, better boundaries
**Cons**: Memory-intensive, slower training

#### C. Temporal Attention U-Net
```
Input: (B, T, H, W, C)
   ↓
Self-attention over time dimension
   ↓
Attention features → 2D U-Net decoder
   ↓
Output: (B, H, W)
```

**Pros**: Learns which time steps are most relevant
**Cons**: More parameters, requires more data

### Evaluation Metrics

| Metric | Purpose | Current Baseline |
|--------|---------|------------------|
| **IoU** | Overall segmentation accuracy | 68.37% |
| **F1-Score** | Balanced precision-recall | 81.22% |
| **Boundary F1** | Edge delineation quality | Not measured yet |
| **False Positive Rate** | Seasonal confusion errors | Not measured yet |
| **Small/Large Object IoU** | Size-dependent performance | Not measured yet |

---

## RESEARCH QUESTION 2: TEMPORAL SAMPLING DENSITY

### Sampling Strategy Comparison

| Sampling Strategy | Time Steps | Input Bands (S2) | Expected IoU | Training Time |
|-------------------|-----------|------------------|--------------|---------------|
| **Bi-temporal** | 2 (2018, 2024) | 18 bands | Baseline | Fastest |
| **Bi-annual** | 4 (2018, 2020, 2022, 2024) | 36 bands | +2-4% | Fast |
| **Annual** | 7 (2018-2024) | 63 bands | +5-7% | Medium |
| **Quarterly** | 14 (Q2+Q3 per year) | 126 bands | +7-10% | Slow |

### Hypotheses

**H1**: Performance improves monotonically with temporal density
**H2**: Quarterly sampling captures construction phases better than annual
**H3**: Diminishing returns after ~5-7 time steps due to temporal redundancy
**H4**: Computational cost grows linearly but accuracy saturates

### Controlled Experiment Design

```
Fixed: Model architecture (e.g., LSTM-UNet)
Fixed: Data source (Sentinel-2)
Fixed: Training hyperparameters

Vary: Number of input time steps (2, 4, 7, 14)

Measure:
  - IoU / F1-Score (accuracy)
  - Boundary F1 (delineation quality)
  - Training time (efficiency)
  - GPU memory usage (scalability)
```

### Expected Trade-offs

| Aspect | Quarterly (14) | Annual (7) | Bi-annual (4) | Bi-temporal (2) |
|--------|---------------|------------|---------------|-----------------|
| **Accuracy** | Highest | High | Medium | Baseline |
| **Boundary Quality** | Best | Good | Fair | Baseline |
| **Training Time** | Slowest (~60 min) | Medium (~30 min) | Fast (~20 min) | Fastest (~15 min) |
| **Memory Usage** | Highest | Medium | Low | Lowest |
| **Data Volume** | 126 bands | 63 bands | 36 bands | 18 bands |

### Key Analysis

1. **Accuracy vs Density Curve**: Plot IoU against number of time steps
2. **Saturation Point**: Identify elbow in performance curve
3. **Boundary Quality**: Does finer sampling improve edge pixels?
4. **Computational ROI**: Performance gain per unit of computational cost

---

## RESEARCH QUESTION 3: TEMPORAL MODELING PARADIGMS

### Two Competing Paradigms

#### Approach A: 1D Temporal Sequences (Per-Pixel)

**Concept**: Treat each pixel as an independent time series

```python
For each pixel (x, y):
    Extract time series: [t1, t2, ..., t14]
    Process with 1D operations (LSTM, 1D Conv, Transformer)
    Predict: change_label(x, y)
```

**Architectures**:
- **Pixel-LSTM**: LSTM over time → 2D CNN decoder
- **Temporal Transformer**: Self-attention over time → 2D decoder
- **1D-CNN-UNet**: 1D Conv over time → 2D U-Net decoder

**Pros**:
- Explicitly models temporal trajectory per pixel
- Efficient for long sequences (14+ time steps)
- Robust to spatial misalignment across time
- Lower memory usage

**Cons**:
- Ignores spatial context within each time step
- May miss spatially-coherent change patterns
- Sensitive to noisy individual pixels

---

#### Approach B: 2D Spatiotemporal Patches

**Concept**: Treat data as 3D volume (space + time)

```python
For each spatial patch (H×W):
    Stack over time: [H×W×t1, H×W×t2, ..., H×W×t14]
    Process with 3D operations (3D Conv, C3D)
    Predict: change_map[H×W]
```

**Architectures**:
- **3D U-Net**: 3D convolutions throughout
- **C3D (3D CNN)**: Video understanding adapted to satellite
- **(2+1)D Conv**: Factorized spatiotemporal convolutions
- **Video Transformer**: Vision Transformer for video

**Pros**:
- Joint spatiotemporal feature learning
- Captures spatial change patterns (e.g., building clusters)
- Better boundary delineation through spatial convolutions
- Spatially coherent predictions

**Cons**:
- Very memory-intensive (H×W×T)
- May not model long-term dependencies well
- Requires more training data
- Slower training

---

### Controlled Comparison

| Model | Temporal | Spatial | Input Shape | Expected Strength |
|-------|----------|---------|-------------|-------------------|
| **Pixel-LSTM** | LSTM (1D) | 2D CNN decoder | (B,T,H,W,C) | Temporal modeling |
| **Temporal Transformer** | Self-Attention | 2D CNN decoder | (B,T,H,W,C) | Long-range temporal |
| **3D U-Net** | 3D Conv | 3D Conv | (B,C,T,H,W) | Joint spatiotemporal |
| **(2+1)D U-Net** | 1D Conv | 2D Conv | (B,C,T,H,W) | Balanced |
| **Hybrid LSTM-3D** | LSTM then 3D | 3D decoder | Both | Best overall? |

### Evaluation Metrics (RQ3 Specific)

| Metric | 1D Temporal | 2D Spatiotemporal | Purpose |
|--------|-------------|-------------------|---------|
| **Boundary F1** | Lower? | **Higher?** | Edge delineation |
| **Small Object IoU** | Lower? | **Higher?** | Spatial coherence |
| **Temporal Noise Robustness** | **Higher?** | Lower? | Noisy pixel handling |
| **Training Time** | **Faster** | Slower | Efficiency |
| **Memory Usage** | **Lower** | Higher | Scalability |

### Hypothesis

**Prediction**:
- **1D models (LSTM/Transformer)**: Better for noisy/cloudy pixels, faster training, temporal trajectory learning
- **2D models (3D U-Net)**: Better for boundary delineation, spatial coherence, small object detection
- **Hybrid models**: Best overall performance by combining temporal modeling with spatial context

---

## IMPLEMENTATION REQUIREMENTS

### New Dataset Classes Needed

```python
# Current (bi-temporal VHR) - ALREADY IMPLEMENTED
class BiTemporalVHRDataset:
    """Loads 2018 RGB + 2025 RGB from VHR Google imagery"""
    returns: (6, H, W) tensor, mask

# NEW - For RQ1/RQ2
class MultiTemporalSentinel2Dataset:
    """Loads Sentinel-2 time series with configurable sampling"""
    returns: (T, C, H, W) tensor, mask
    # T = 2, 4, 7, or 14 time steps
    # C = 9 spectral bands

class MultiTemporalPlanetScopeDataset:
    """Loads PlanetScope RGB time series"""
    returns: (T, 3, H, W) tensor, mask
    # T = 2, 4, 7, or 14 time steps

# NEW - For RQ3 (different data organization)
class PixelTimeSeriesDataset:
    """Organizes data for per-pixel temporal processing (1D)"""
    returns: (B, H, W, T, C) tensor, mask
    # Pixels × Time × Channels

class SpatiotemporalPatchDataset:
    """Organizes data for 3D convolutions (2D)"""
    returns: (B, C, T, H, W) tensor, mask
    # Channels × Time × Height × Width
```

### Preprocessing Pipeline

```
RAW SENTINEL-2 (126 bands, GeoTIFF)
    ↓
1. Extract temporal slices
   ├─ Quarterly: T=14 (2018_Q2, 2018_Q3, ..., 2024_Q3)
   ├─ Annual: T=7 (average Q2+Q3 per year)
   ├─ Bi-annual: T=4 (2018, 2020, 2022, 2024)
   └─ Bi-temporal: T=2 (2018, 2024)
    ↓
2. Normalize per band
   ├─ Min-max scaling to [0, 1]
   └─ Or Z-score normalization (μ=0, σ=1)
    ↓
3. Cloud masking (optional, if needed)
   └─ Interpolate cloudy pixels temporally
    ↓
4. Spatial resampling
   ├─ Option A: Keep 10m (faster, less memory)
   └─ Option B: Upsample to 1m (align with VHR masks)
    ↓
5. Data augmentation
   ├─ Spatial: rotation, flip, crop (consistent across T)
   ├─ Temporal: random time reversal, temporal dropout
   └─ Photometric: brightness, contrast (per time step)
    ↓
OUTPUT: (T, C, H, W) tensor ready for model
```

### New Model Architectures Needed

#### 1. LSTM-UNet (Priority 1 - RQ1/RQ2 baseline)
```python
class LSTMUNet:
    """1D temporal processing with LSTM, 2D spatial decoder"""

    encoder:
        - ConvLSTM or LSTM over time dimension
        - Extract temporal features per pixel

    decoder:
        - Standard U-Net 2D decoder
        - Skip connections from LSTM hidden states

    forward:
        (B, T, C, H, W) → LSTM → (B, F, H, W) → UNet → (B, 1, H, W)
```

#### 2. 3D U-Net (Priority 2 - RQ3)
```python
class UNet3D:
    """3D convolutions for joint spatiotemporal processing"""

    encoder:
        - 3D convolutions (Conv3d)
        - 3D pooling (MaxPool3d)
        - Processes space+time jointly

    decoder:
        - 3D transposed convolutions
        - 3D skip connections

    forward:
        (B, C, T, H, W) → 3D Conv → ... → (B, 1, H, W)
```

#### 3. Temporal Transformer (Priority 3 - RQ3)
```python
class TemporalTransformerUNet:
    """Self-attention over time, 2D spatial decoder"""

    temporal_encoder:
        - Multi-head self-attention over time dimension
        - Positional encoding for time steps

    spatial_decoder:
        - 2D U-Net decoder
        - Skip connections from attention outputs

    forward:
        (B, T, C, H, W) → Attention → (B, F, H, W) → UNet → (B, 1, H, W)
```

#### 4. (2+1)D U-Net (Priority 4 - RQ3)
```python
class UNet2Plus1D:
    """Factorized spatiotemporal convolutions"""

    encoder:
        - Separate 2D spatial + 1D temporal convolutions
        - More efficient than full 3D

    decoder:
        - (2+1)D transposed convolutions

    forward:
        (B, C, T, H, W) → (2+1)D Conv → ... → (B, 1, H, W)
```

### Boundary Quality Metrics

```python
def compute_boundary_f1(pred, target, boundary_width=5):
    """
    F1-Score computed only on boundary pixels

    Args:
        pred: Predicted mask (H, W)
        target: Ground truth mask (H, W)
        boundary_width: Pixel width of boundary zone

    Returns:
        Boundary F1-Score
    """
    # Extract boundary pixels using morphological operations
    # Compute F1 only on these pixels

def compute_hausdorff_distance(pred, target):
    """
    Maximum boundary deviation (worst-case edge accuracy)
    """
    # Compute directed Hausdorff distance
    # Returns max distance in pixels

def compute_size_stratified_iou(pred, target, size_bins=[0, 100, 1000, inf]):
    """
    IoU stratified by object size

    Returns:
        {
            'small_objects_iou': 0.xx,
            'medium_objects_iou': 0.xx,
            'large_objects_iou': 0.xx
        }
    """
```

---

## PHASED EXECUTION PLAN

### Phase 0: Setup and Preparation (1 week)

**Tasks**:
- [ ] Create `MultiTemporalSentinel2Dataset` class
- [ ] Implement temporal sampling strategies (quarterly/annual/bi-annual/bi-temporal)
- [ ] Verify Sentinel-2 data quality and completeness for all 53 tiles
- [ ] Implement preprocessing pipeline (normalization, resampling)
- [ ] Test data loading with small batch

**Deliverables**:
- Working data loader for multi-temporal Sentinel-2
- Data quality report
- Example visualizations of temporal sequences

---

### Phase 1: Proof of Concept (2-3 weeks) - RQ1

**Goal**: Confirm that multi-temporal data improves over bi-temporal baseline

**Experiments**:

| Experiment | Data | Model | Time Steps | Purpose |
|------------|------|-------|------------|---------|
| **Baseline** | VHR 2018/2025 | SiamConc-ResNet50 | 2 | Current performance |
| **S2 Bi-temporal** | Sentinel-2 2018/2024 | LSTM-UNet | 2 | Fair comparison at 10m |
| **S2 Annual** | Sentinel-2 annual | LSTM-UNet | 7 | Multi-temporal benefit |
| **S2 Quarterly** | Sentinel-2 quarterly | LSTM-UNet | 14 | Full temporal density |

**Success Criteria**: Multi-temporal (7 or 14 steps) achieves ≥5% IoU improvement over bi-temporal

**Deliverables**:
- Trained LSTM-UNet models (4 variants)
- Comparative performance table
- Decision: Continue to Phase 2 if successful

---

### Phase 2: Temporal Sampling Study (2 weeks) - RQ2

**Goal**: Determine optimal temporal sampling density

**Experiments**:

| Sampling | Time Steps | Expected IoU | Training Time |
|----------|-----------|--------------|---------------|
| Bi-temporal | 2 | Baseline | ~15 min |
| Bi-annual | 4 | Baseline + 2-4% | ~20 min |
| Annual | 7 | Baseline + 5-7% | ~30 min |
| Quarterly | 14 | Baseline + 7-10% | ~60 min |

**Analysis**:
1. Plot accuracy vs number of time steps (learning curve)
2. Plot training time vs number of time steps
3. Compute performance gain per unit time (efficiency metric)
4. Identify saturation point (diminishing returns)

**Deliverables**:
- Learning curves (IoU vs temporal density)
- Computational cost analysis
- Recommended sampling strategy for production use

---

### Phase 3: Architecture Comparison (3-4 weeks) - RQ3

**Goal**: Compare 1D temporal vs 2D spatiotemporal modeling

**Using best temporal sampling from Phase 2 (likely quarterly or annual)**

**Experiments**:

| Model Family | Architecture | Expected IoU | Training Time | Memory |
|--------------|-------------|--------------|---------------|--------|
| **1D Temporal** | LSTM-UNet | 75-78% | ~30 min | 8GB |
| | Temporal Transformer | 76-79% | ~45 min | 12GB |
| | 1D-CNN-UNet | 74-77% | ~25 min | 6GB |
| **2D Spatiotemporal** | 3D U-Net | 77-80% | ~90 min | 16GB |
| | (2+1)D U-Net | 78-81% | ~60 min | 12GB |
| **Hybrid** | LSTM + 3D decoder | 79-82%? | ~90 min | 16GB |

**Detailed Evaluation** (all models):
- Overall metrics: IoU, F1, Precision, Recall
- Boundary F1 (edge quality)
- Hausdorff distance (worst-case boundary)
- IoU by object size (small/medium/large)
- Temporal consistency (prediction stability)
- Computational efficiency (FLOPs, memory, time)

**Deliverables**:
- Trained models (6 architectures)
- Comprehensive comparison table
- Best architecture identification
- Qualitative analysis (where does each model fail?)

---

### Phase 4: Boundary Quality Analysis (1-2 weeks)

**Goal**: Deep dive into boundary delineation quality

**Analysis**:
1. **Boundary F1 computation** for all models
2. **Hausdorff distance** analysis (worst-case errors)
3. **Visual comparison** of boundary predictions
4. **Error analysis**: Where do boundaries fail?
   - High-contrast edges (buildings)
   - Low-contrast edges (roads, fields)
   - Small objects
   - Complex shapes

**Deliverables**:
- Boundary quality metrics table
- Visualization gallery (best vs worst boundaries)
- Error pattern analysis
- Recommendations for boundary improvement

---

### Phase 5: Results Analysis and Thesis Writing (2-3 weeks)

**Goal**: Synthesize results and answer research questions

**Tasks**:
- [ ] Create comprehensive results tables and figures
- [ ] Statistical significance testing (multi-seed evaluation)
- [ ] Write Methods section (datasets, architectures, training)
- [ ] Write Results section (answer each RQ systematically)
- [ ] Write Discussion section (implications, limitations)
- [ ] Create visualization gallery for thesis

**Deliverables**:
- Complete thesis chapter draft
- Publication-ready figures and tables
- Supplementary materials

---

## TIMELINE SUMMARY

| Phase | Duration | Key Deliverable |
|-------|----------|----------------|
| **Phase 0: Setup** | 1 week | Working multi-temporal data loader |
| **Phase 1: Proof of Concept (RQ1)** | 2-3 weeks | Multi-temporal benefit confirmed |
| **Phase 2: Sampling Study (RQ2)** | 2 weeks | Optimal sampling identified |
| **Phase 3: Architecture Comparison (RQ3)** | 3-4 weeks | Best model architecture |
| **Phase 4: Boundary Analysis** | 1-2 weeks | Boundary quality metrics |
| **Phase 5: Analysis & Writing** | 2-3 weeks | Complete thesis chapter |
| **TOTAL** | **11-15 weeks** | **Three research questions answered** |

---

## MINIMAL VIABLE EXPERIMENT (RECOMMENDED START)

If time is limited or to validate approach quickly:

### Week 1-2: Data Pipeline
- Build `MultiTemporalSentinel2Dataset` for annual sampling (7 time steps)
- Verify data quality for all 53 tiles
- Test data loading and preprocessing

### Week 3: Model Implementation
- Implement LSTM-UNet architecture
- Port training script to work with temporal data
- Verify model can train on small subset

### Week 4: Initial Training
- Train LSTM-UNet with annual Sentinel-2 (7 time steps)
- Compare to bi-temporal baseline
- Evaluate on validation set

### Week 5: Decision Point
**If promising** (5-10% IoU improvement):
→ Continue with full experimental matrix (Phases 2-4)

**If not promising** (<5% improvement):
→ Debug and investigate:
  - Data quality issues?
  - Model architecture problems?
  - Preprocessing errors?
  - Temporal alignment issues?

---

## EXPECTED CONTRIBUTIONS

### 1. Scientific Novelty

- **First systematic comparison** of 1D vs 2D temporal modeling for land-take detection
- **Empirical analysis** of temporal sampling trade-offs in change detection
- **Multi-source evaluation** (VHR, Sentinel-2, PlanetScope) under controlled conditions
- **Boundary quality focus** - explicit analysis of edge delineation vs overall accuracy

### 2. Practical Impact

**Operational Guidance**:
- How much temporal data is "enough" for land-take detection?
- When to use LSTM vs 3D CNN architectures?
- Cost-benefit analysis: Performance vs computational requirements
- Recommendations for production deployment

**Scalability Insights**:
- Memory requirements for different approaches
- Training time trade-offs
- Inference speed comparison
- Data volume requirements

### 3. Publication Potential

**Main Paper** (Target: Remote Sensing journal):
> "Temporal Modeling Strategies for Land-Take Detection from Multi-Temporal Satellite Imagery: A Systematic Comparison of 1D and 2D Approaches"

**Workshop Paper** (Target: IGARSS, ECML-PKDD workshops):
> "How Much Temporal Information Do We Need for Change Detection? An Empirical Study"

**Dataset Paper** (Target: Scientific Data):
> "HABLOSS: A Multi-Temporal Multi-Source Benchmark for Land-Take Detection Across Europe"

---

## SUCCESS METRICS

### Minimum Success Criteria

- [ ] Answer all three research questions with empirical evidence
- [ ] Achieve ≥5% IoU improvement over bi-temporal baseline
- [ ] Complete training and evaluation for at least 2 architectures per RQ
- [ ] Demonstrate statistical significance (multi-seed evaluation)
- [ ] Complete thesis chapter draft

### Excellent Success Criteria

- [ ] Achieve ≥10% IoU improvement over baseline
- [ ] Complete full experimental matrix (~40-50 experiments)
- [ ] Identify clear winner for each research question
- [ ] Publish 1-2 papers in conferences/journals
- [ ] Release code and models publicly

---

## RISKS AND MITIGATION

### Risk 1: Multi-temporal doesn't improve performance

**Mitigation**:
- Start with Phase 1 proof-of-concept before committing to full plan
- If multi-temporal fails, pivot to alternative research questions:
  - Alternative fusion strategies
  - Self-supervised pre-training
  - Semi-supervised learning with unlabeled tiles

### Risk 2: Computational resources insufficient

**Symptoms**: 3D U-Net doesn't fit in GPU memory, training takes too long

**Mitigation**:
- Use gradient checkpointing to reduce memory
- Train on smaller spatial patches (256×256 instead of 512×512)
- Focus on efficient architectures (2+1)D, LSTM-UNet)
- Use mixed precision training (FP16)

### Risk 3: Data quality issues

**Symptoms**: Sentinel-2 has too many clouds, missing time steps

**Mitigation**:
- Use PlanetScope as alternative (higher resolution, 3-5m)
- Implement cloud masking and temporal interpolation
- Focus on clean tiles only (may reduce from 53 to ~40)
- Use AlphaEarth embeddings (pre-processed, cloud-free)

### Risk 4: Timeline too ambitious

**Mitigation**:
- Execute minimal viable experiment first (5 weeks)
- Prioritize RQ1 (most important) over RQ2/RQ3
- Parallelize experiments using SLURM job arrays
- Reduce number of model variants if time-constrained

---

## RESOURCES REQUIRED

### Computational Resources

**Current (available)**:
- Tesla P100 GPU (16GB VRAM)
- 32-48GB RAM
- SLURM job scheduler

**Requirements**:
- GPU: Sufficient for LSTM-UNet, may need optimization for 3D U-Net
- RAM: Sufficient for current tasks
- Storage: ~50-100GB for models, predictions, visualizations

**Optimization strategies if needed**:
- Mixed precision training (reduces memory by ~40%)
- Gradient checkpointing
- Smaller batch sizes
- Distributed training (if multiple GPUs available)

### Software Dependencies

**Current**:
- PyTorch (deep learning framework)
- Albumentations (data augmentation)
- GDAL/Rasterio (geospatial data)
- Weights & Biases (experiment tracking)

**Additional needed**:
- PyTorch3D or torch-geometric (for 3D convolutions)
- scikit-image (for boundary metrics)
- scipy.spatial (for Hausdorff distance)

### Data Dependencies

**Status**: All required data is already available

- Sentinel-2: 126 bands, 53 tiles
- PlanetScope: 42 bands, 53 tiles
- VHR Google: 6 bands, 53 tiles
- Masks: 1 band, 53 tiles

**No additional data collection needed**

---

## NEXT STEPS

### Immediate Actions (This Week)

1. **Review and approve this plan** with thesis advisor
2. **Verify Sentinel-2 data** quality and completeness
3. **Start implementing** `MultiTemporalSentinel2Dataset` class
4. **Set up project structure** for new experiments

### Short-term Actions (Next 2 Weeks)

1. **Complete Phase 0** (data pipeline setup)
2. **Implement LSTM-UNet** architecture
3. **Run first proof-of-concept** experiment
4. **Evaluate initial results** and decide on continuation

### Medium-term Actions (Next 1-2 Months)

1. **Execute Phases 1-3** (main experiments)
2. **Collect results** systematically
3. **Generate visualizations** for thesis
4. **Begin thesis writing**

---

## QUESTIONS FOR DISCUSSION

Before proceeding, consider:

1. **Scope**: Is this plan too ambitious for available time? Should we focus on RQ1 only?
2. **Resources**: Do we have sufficient GPU access for 3D models?
3. **Priorities**: Which research question is most important? (RQ1, RQ2, or RQ3?)
4. **Timeline**: What is the hard deadline for thesis submission?
5. **Publications**: Is the goal thesis-only, or also conference/journal papers?
6. **Collaboration**: Are there collaborators or co-authors to coordinate with?

---

## CONCLUSION

This tentative plan outlines a comprehensive research agenda to answer three important questions about multi-temporal deep learning for land-take detection. The plan is ambitious but feasible given the available data and computational resources.

**Key Strengths**:
- All required data is already available
- Clear, answerable research questions
- Systematic experimental design
- Phased approach allows early stopping if needed
- Strong baseline (68.37% IoU) to build upon

**Recommended Approach**:
Start with the minimal viable experiment (5 weeks) to validate that multi-temporal data provides benefit. If successful, proceed with the full experimental matrix. If not, pivot to alternative research directions.

**Expected Outcome**:
A complete thesis chapter with empirical answers to all three research questions, supported by comprehensive experiments and analysis. Potential for 1-2 publications in conferences or journals.

---

**Status**: TENTATIVE - Awaiting approval and feedback
**Next Review Date**: TBD
**Contact**: tmstorma@stud.ntnu.no
