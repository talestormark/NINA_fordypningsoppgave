# Baseline Model Architectures - Visual Guide

This document provides visual diagrams and detailed explanations of the baseline architectures for land-take detection.

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [U-Net Early Fusion](#1-u-net-early-fusion)
3. [U-Net SiamDiff](#2-u-net-siamdiff)
4. [U-Net SiamConc](#3-u-net-siamconc)
5. [Input/Output Specifications](#inputoutput-specifications)
6. [Architecture Comparison](#architecture-comparison)

---

## Architecture Overview

All baseline models follow an encoder-decoder architecture with skip connections, based on the U-Net framework. The key differences lie in how bi-temporal images are processed and fused.

**Common Components:**
- **Encoder**: Extracts hierarchical features (ResNet-50 or EfficientNet-B4 backbone)
- **Decoder**: Upsamples features to original resolution
- **Skip Connections**: Preserve spatial information across encoder-decoder
- **Output Head**: Final 1×1 convolution to binary segmentation

---

## 1. U-Net Early Fusion

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                          INPUT PREPARATION                          │
└─────────────────────────────────────────────────────────────────────┘

    Image t1 (2018)              Image t2 (2025)
    [H × W × 3]                  [H × W × 3]
         │                            │
         └────────── Concatenate ─────┘
                        │
                   [H × W × 6]
                        │
                        ▼

┌─────────────────────────────────────────────────────────────────────┐
│                        ENCODER (ResNet-50)                          │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │  Conv1: [H×W×6] → [H/2×W/2×64]                           │      │
│  │  + BatchNorm + ReLU + MaxPool                            │      │
│  └──────────────────────────────────────────────────────────┘      │
│                         │                                           │
│                         ├──────────────► Skip Connection 1         │
│                         │                                           │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │  Layer 1: [H/2×W/2×64] → [H/4×W/4×256]                   │      │
│  │  (3 Bottleneck blocks)                                   │      │
│  └──────────────────────────────────────────────────────────┘      │
│                         │                                           │
│                         ├──────────────► Skip Connection 2         │
│                         │                                           │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │  Layer 2: [H/4×W/4×256] → [H/8×W/8×512]                  │      │
│  │  (4 Bottleneck blocks)                                   │      │
│  └──────────────────────────────────────────────────────────┘      │
│                         │                                           │
│                         ├──────────────► Skip Connection 3         │
│                         │                                           │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │  Layer 3: [H/8×W/8×512] → [H/16×W/16×1024]               │      │
│  │  (6 Bottleneck blocks)                                   │      │
│  └──────────────────────────────────────────────────────────┘      │
│                         │                                           │
│                         ├──────────────► Skip Connection 4         │
│                         │                                           │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │  Layer 4: [H/16×W/16×1024] → [H/32×W/32×2048]            │      │
│  │  (3 Bottleneck blocks)                                   │      │
│  └──────────────────────────────────────────────────────────┘      │
│                         │                                           │
│                    [H/32×W/32×2048]                                 │
│                    (Bottleneck)                                     │
└─────────────────────────────────────────────────────────────────────┘
                          │
                          ▼

┌─────────────────────────────────────────────────────────────────────┐
│                            DECODER                                   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │  Upsample 1: [H/32×W/32×2048] → [H/16×W/16×1024]         │      │
│  │  + Skip Connection 4 → [H/16×W/16×2048]                  │      │
│  │  + Conv + BatchNorm + ReLU                               │      │
│  └──────────────────────────────────────────────────────────┘      │
│                         │                                           │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │  Upsample 2: [H/16×W/16×1024] → [H/8×W/8×512]            │      │
│  │  + Skip Connection 3 → [H/8×W/8×1024]                    │      │
│  │  + Conv + BatchNorm + ReLU                               │      │
│  └──────────────────────────────────────────────────────────┘      │
│                         │                                           │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │  Upsample 3: [H/8×W/8×512] → [H/4×W/4×256]               │      │
│  │  + Skip Connection 2 → [H/4×W/4×512]                     │      │
│  │  + Conv + BatchNorm + ReLU                               │      │
│  └──────────────────────────────────────────────────────────┘      │
│                         │                                           │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │  Upsample 4: [H/4×W/4×256] → [H/2×W/2×64]                │      │
│  │  + Skip Connection 1 → [H/2×W/2×128]                     │      │
│  │  + Conv + BatchNorm + ReLU                               │      │
│  └──────────────────────────────────────────────────────────┘      │
│                         │                                           │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │  Final Upsample: [H/2×W/2×64] → [H×W×64]                 │      │
│  │  + Conv + BatchNorm + ReLU                               │      │
│  └──────────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        OUTPUT HEAD                                   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │  Conv 1×1: [H×W×64] → [H×W×2]                            │      │
│  │  (2 classes: change / no-change)                         │      │
│  └──────────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
                  Binary Segmentation Mask
                     [H × W × 1]
                  (0 = no change, 1 = change)
```

### Key Characteristics

**Input Processing:**
- Concatenates bi-temporal images along channel dimension
- Single pass through encoder (6 input channels)
- All temporal information processed together

**Advantages:**
- Simple, straightforward architecture
- Leverages pretrained ImageNet weights (first conv adapted for 6 channels)
- Proven top performer on benchmarks

**Disadvantages:**
- No explicit temporal difference modeling
- First convolutional layer loses pretraining (6 channels vs. 3 channels)

**Parameters:** ~23.5M (ResNet-50 backbone)

---

## 2. U-Net SiamDiff

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                          INPUT PREPARATION                          │
└─────────────────────────────────────────────────────────────────────┘

    Image t1 (2018)              Image t2 (2025)
    [H × W × 3]                  [H × W × 3]
         │                            │
         │                            │
         ▼                            ▼

┌─────────────────────────────────────────────────────────────────────┐
│               SHARED ENCODER (ResNet-50) - Weight Sharing           │
└─────────────────────────────────────────────────────────────────────┘

    Encoder Branch 1              Encoder Branch 2
    (processes t1)               (processes t2)
         │                            │
         │  ┌─────────────┐           │  ┌─────────────┐
         └─►│  Conv1      │           └─►│  Conv1      │
            │  [H/2×W/2×64]│              │  [H/2×W/2×64]│
            └─────────────┘              └─────────────┘
         │                            │
         │  ┌─────────────┐           │  ┌─────────────┐
         └─►│  Layer 1    │           └─►│  Layer 1    │
            │[H/4×W/4×256]│              │[H/4×W/4×256]│
            └─────────────┘              └─────────────┘
         │                            │
         │  ┌─────────────┐           │  ┌─────────────┐
         └─►│  Layer 2    │           └─►│  Layer 2    │
            │[H/8×W/8×512]│              │[H/8×W/8×512]│
            └─────────────┘              └─────────────┘
         │                            │
         │  ┌─────────────┐           │  ┌─────────────┐
         └─►│  Layer 3    │           └─►│  Layer 3    │
            │[H/16×W/16×1024]          │[H/16×W/16×1024]
            └─────────────┘              └─────────────┘
         │                            │
         │  ┌─────────────┐           │  ┌─────────────┐
         └─►│  Layer 4    │           └─►│  Layer 4    │
            │[H/32×W/32×2048]          │[H/32×W/32×2048]
            └─────────────┘              └─────────────┘
         │                            │
         │                            │
    Features_t1                  Features_t2
    [H/32×W/32×2048]             [H/32×W/32×2048]
         │                            │
         └──────────┬─────────────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │  DIFFERENCE OPERATION │
         │                       │
         │  diff = |f_t2 - f_t1| │
         └──────────────────────┘
                    │
                    ▼
            [H/32×W/32×2048]
         (Temporal difference features)
                    │
                    ▼

┌─────────────────────────────────────────────────────────────────────┐
│                            DECODER                                   │
│                  (Same as Early Fusion)                              │
│                                                                      │
│  Skip connections use difference features:                           │
│    skip_i = |features_t2[i] - features_t1[i]|                       │
└─────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
            Binary Segmentation Mask
                [H × W × 1]
```

### Key Characteristics

**Input Processing:**
- Separate encoding of each temporal image
- Weight sharing between Siamese branches
- Feature-level difference computation

**Difference Operation:**
```python
# At each encoder level
diff_features = torch.abs(features_t2 - features_t1)
```

**Advantages:**
- Explicit temporal difference modeling
- Full pretrained weights preserved (3-channel input)
- Focuses on changed regions through feature difference

**Disadvantages:**
- Absolute difference may lose directional information
- Slightly more complex than Early Fusion

**Parameters:** ~23.5M (ResNet-50 backbone, shared across time)

---

## 3. U-Net SiamConc

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                          INPUT PREPARATION                          │
└─────────────────────────────────────────────────────────────────────┘

    Image t1 (2018)              Image t2 (2025)
    [H × W × 3]                  [H × W × 3]
         │                            │
         │                            │
         ▼                            ▼

┌─────────────────────────────────────────────────────────────────────┐
│               SHARED ENCODER (ResNet-50) - Weight Sharing           │
└─────────────────────────────────────────────────────────────────────┘

    Encoder Branch 1              Encoder Branch 2
    (processes t1)               (processes t2)
         │                            │
         │  ┌─────────────┐           │  ┌─────────────┐
         └─►│  Conv1      │           └─►│  Conv1      │
            │  [H/2×W/2×64]│              │  [H/2×W/2×64]│
            └─────────────┘              └─────────────┘
         │                            │
         │  ┌─────────────┐           │  ┌─────────────┐
         └─►│  Layer 1    │           └─►│  Layer 1    │
            │[H/4×W/4×256]│              │[H/4×W/4×256]│
            └─────────────┘              └─────────────┘
         │                            │
         │  ┌─────────────┐           │  ┌─────────────┐
         └─►│  Layer 2    │           └─►│  Layer 2    │
            │[H/8×W/8×512]│              │[H/8×W/8×512]│
            └─────────────┘              └─────────────┘
         │                            │
         │  ┌─────────────┐           │  ┌─────────────┐
         └─►│  Layer 3    │           └─►│  Layer 3    │
            │[H/16×W/16×1024]          │[H/16×W/16×1024]
            └─────────────┘              └─────────────┘
         │                            │
         │  ┌─────────────┐           │  ┌─────────────┐
         └─►│  Layer 4    │           └─►│  Layer 4    │
            │[H/32×W/32×2048]          │[H/32×W/32×2048]
            └─────────────┘              └─────────────┘
         │                            │
         │                            │
    Features_t1                  Features_t2
    [H/32×W/32×2048]             [H/32×W/32×2048]
         │                            │
         └──────────┬─────────────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │ CONCATENATION         │
         │                       │
         │  concat = [f_t1, f_t2]│
         └──────────────────────┘
                    │
                    ▼
            [H/32×W/32×4096]
         (Concatenated features: 2×2048)
                    │
                    ▼

┌─────────────────────────────────────────────────────────────────────┐
│                            DECODER                                   │
│                  (Modified for double channels)                      │
│                                                                      │
│  Skip connections use concatenated features:                         │
│    skip_i = concat(features_t1[i], features_t2[i])                  │
│                                                                      │
│  First decoder conv adapted to handle 4096 channels                  │
└─────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
            Binary Segmentation Mask
                [H × W × 1]
```

### Key Characteristics

**Input Processing:**
- Separate encoding of each temporal image
- Weight sharing between Siamese branches
- Feature-level concatenation

**Concatenation Operation:**
```python
# At each encoder level
concat_features = torch.cat([features_t1, features_t2], dim=1)
```

**Advantages:**
- Preserves all temporal information (no information loss)
- Decoder learns optimal fusion strategy
- Full pretrained weights preserved

**Disadvantages:**
- Doubled channel dimensions (higher memory usage)
- Decoder must be modified to handle concatenated features
- More parameters in decoder

**Parameters:** ~24.5M (slightly more due to wider decoder)

---

## Input/Output Specifications

### Input Formats by Sensor

**VHR Google (1m resolution):**
```
Input shape: [B, 6, H, W]
  - Channels 0-2: 2018 RGB (R, G, B)
  - Channels 3-5: 2025 RGB (R, G, B)
  - Typical H×W: 650-1600 × 654-662 pixels
  - Value range: [0, 255] → normalized to [-1, 1]
```

**Sentinel-2 (10m resolution):**
```
Input shape: [B, 8, H, W]
  - Channels 0-3: 2018 RGB+NIR (R, G, B, NIR)
  - Channels 4-7: 2025 RGB+NIR (R, G, B, NIR)
  - Typical H×W: ~65 × 65 pixels
  - Value range: [0, 10000] → normalized to [-1, 1]
```

**PlanetScope (3-5m resolution):**
```
Input shape: [B, 6, H, W]
  - Channels 0-2: Start quarter RGB
  - Channels 3-5: End quarter RGB
  - Typical H×W: ~200 × 160 pixels
  - Value range: [0, 255] → normalized to [-1, 1]
```

### Output Format

```
Output shape: [B, 2, H, W]
  - Channel 0: No-change probability
  - Channel 1: Change probability

After softmax/argmax:
  Final mask: [B, 1, H, W]
  - 0 = No change
  - 1 = Land-take change
```

---

## Architecture Comparison

### Parameter Counts

| Architecture       | Backbone    | Parameters | Memory (FP32) |
|-------------------|-------------|------------|---------------|
| U-Net Early Fusion | ResNet-50   | 23.5M      | ~94 MB        |
| U-Net SiamDiff     | ResNet-50   | 23.5M      | ~94 MB        |
| U-Net SiamConc     | ResNet-50   | 24.5M      | ~98 MB        |
| U-Net Early Fusion | EfficientNet-B4 | 17.8M | ~71 MB        |

### Computational Cost (FLOPs)

**Input: 256×256 RGB images**

| Architecture       | FLOPs      | Relative Speed |
|-------------------|------------|----------------|
| U-Net Early Fusion | 55.3 GFLOPs | 1.0× (baseline) |
| U-Net SiamDiff     | 65.8 GFLOPs | 0.84× (slower)  |
| U-Net SiamConc     | 68.2 GFLOPs | 0.81× (slower)  |

Siamese variants are slower because they encode each image separately (2× encoder passes).

### Performance Comparison (from Reality Check paper)

**LEVIR-CD Dataset (0.5m resolution, urban):**

| Architecture       | Backbone    | F1    | IoU   | Precision | Recall |
|-------------------|-------------|-------|-------|-----------|--------|
| U-Net Early Fusion | ResNet-50   | 90.38 | -     | 91.97     | 89.78  |
| U-Net SiamDiff     | ResNet-50   | 90.46 | -     | 93.21     | 89.50  |
| U-Net SiamConc     | ResNet-50   | 90.41 | -     | 92.87     | 89.48  |
| U-Net Early Fusion | EfficientNet-B4 | 89.25 | - | 92.69     | 87.16  |

**WHU-CD Dataset (0.075m resolution, urban):**

| Architecture       | Backbone    | F1    | IoU   | Precision | Recall |
|-------------------|-------------|-------|-------|-----------|--------|
| U-Net Early Fusion | ResNet-50   | 84.17 | 73.23 | 88.65     | 83.08  |
| U-Net SiamDiff     | ResNet-50   | 84.01 | 73.02 | 88.56     | 85.63  |
| U-Net SiamConc     | ResNet-50   | 82.75 | 71.15 | 83.69     | 86.56  |

**Key Observations:**
- All three architectures perform similarly (within 1-2% F1)
- SiamDiff shows slight precision improvement
- SiamConc shows slight recall improvement
- Early Fusion is fastest while maintaining competitive performance

### Trade-offs Summary

| Aspect              | Early Fusion | SiamDiff | SiamConc |
|---------------------|-------------|----------|----------|
| **Simplicity**      | ✓✓✓         | ✓✓       | ✓✓       |
| **Training Speed**  | ✓✓✓         | ✓✓       | ✓✓       |
| **Memory Usage**    | ✓✓✓         | ✓✓✓      | ✓✓       |
| **Pretrained Weights** | ✓✓       | ✓✓✓      | ✓✓✓      |
| **Temporal Modeling** | ✓         | ✓✓✓      | ✓✓✓      |
| **Performance**     | ✓✓✓         | ✓✓✓      | ✓✓✓      |

**Recommendation:** Start with **U-Net Early Fusion** for baseline, then test **SiamDiff** if temporal modeling benefits are suspected.

---

## Implementation Notes

### Backbone Selection

**ResNet-50 (Recommended):**
- Well-established baseline
- Good balance of performance and speed
- Extensive pretrained weights
- ~23M parameters

**EfficientNet-B4:**
- More efficient architecture
- Potentially better feature extraction
- ~17M parameters
- Slightly slower per-layer operations

### Pretrained Weight Initialization

**Early Fusion:**
```python
# First conv layer adapted for 6 channels
# Initialize by averaging RGB weights twice
conv1_weight = pretrained_model.conv1.weight  # [64, 3, 7, 7]
conv1_weight_6ch = torch.cat([conv1_weight, conv1_weight], dim=1) / 2
model.conv1.weight = conv1_weight_6ch  # [64, 6, 7, 7]
```

**Siamese (SiamDiff/SiamConc):**
```python
# Pretrained weights used directly (3-channel input preserved)
encoder = models.resnet50(pretrained=True)
# No modification needed
```

### Skip Connection Details

**Early Fusion & SiamDiff:**
```python
# Standard skip connections
decoder_input = upsample(bottleneck_features)
decoder_input = concat(decoder_input, skip_features)
```

**SiamConc:**
```python
# Concatenated skip connections (doubled channels)
skip_features_concat = concat(skip_t1, skip_t2)  # 2× channels
decoder_input = upsample(bottleneck_features)
decoder_input = concat(decoder_input, skip_features_concat)
```

---

## Summary

All three baseline architectures are proven performers with minimal performance differences (<2% F1). The choice depends on:

1. **Simplicity priority** → U-Net Early Fusion
2. **Explicit temporal modeling** → U-Net SiamDiff
3. **Maximal information preservation** → U-Net SiamConc

For our land-take detection project, we'll implement and compare all three to determine which best handles multi-scale patches and class imbalance.
