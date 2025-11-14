# NINA Land-Take Detection Project - Summary

**Author:** tmstorma@stud.ntnu.no
**Project:** NINA Fordypningsoppgave (Deep Learning for Land-Take Detection)
**Data Source:** HABLOSS Project (https://habloss.eu/)
**Date:** October 2025

---

## Project Overview

This project focuses on detecting and analyzing land-take (the conversion of natural or agricultural land to artificial surfaces) across diverse European locations using high-resolution satellite imagery and change detection masks. The work represents a comprehensive data validation and exploration pipeline, preparing a multi-source remote sensing dataset for deep learning-based land-take detection.

### Main Objectives
1. Validate and understand a complex multi-source satellite imagery dataset
2. Assess data quality across 5 different remote sensing sources
3. Analyze land-take patterns and class distribution
4. Prepare a clean, well-documented dataset for deep learning model development

### Key Dataset Statistics
- **Total validated tiles:** 53 complete tiles (from 55 in original dataset)
- **Geographic coverage:** 20 European countries across diverse landscapes
- **Tile size:** ~650m × 650m geographic extent
- **Time span:** 2018-2025 (7 years of observations)
- **Resolution range:** 1m to 10m across different sensors

---

## What We've Done

### Task 1: Environment Setup and Configuration
**What:** Set up the project structure, Python environment, and configuration system to handle multiple data sources and processing workflows.

**Files:**
- `config.py` - Central configuration with data paths and constants
- `environment/landtake_env.yml` - Conda environment specification
- `environment/setup_env.sh` - Environment setup script

**Why:** A well-organized project structure is essential for managing complex multi-source datasets and ensuring reproducibility. The configuration system allows flexible path management for both local and HPC environments.

**Results:**
- Created organized directory structure with separate folders for scripts, outputs, and documentation
- Defined standardized file naming patterns for all 5 data sources
- Set up modular configuration that can be easily adapted to different computing environments

---

### Task 2: Filesystem Exploration and REFID Extraction
**What:** Systematically explored the data directory structure, counted available files, and extracted unique tile identifiers (REFIDs) to understand what data is actually available.

**Files:**
- `scripts/data_validation/01_inspect_filesystem.py`
- `scripts/data_validation/02_extract_refids.py`
- `outputs/reports/folder_structure.txt`
- `outputs/reports/refid_list.txt`
- `outputs/reports/refid_presence.csv`

**Why:** Before processing any satellite data, we need to understand what files exist, how they're named, and which tiles have complete coverage across all data sources. This prevents errors during later processing stages.

**Results:**
- Identified 53 tiles with complete data across all main sources (Sentinel-2, VHR Google, AlphaEarth, Masks, PlanetScope)
- Discovered 2 tiles missing from the expected 55 due to incomplete data
- Created comprehensive REFID list with metadata (country, loss type, change type) from GeoJSON
- Geographic distribution: UK (8 tiles), France (8), Belgium (7), Netherlands (6), plus 14 other countries
- Loss type balance: 28 cropland loss tiles, 27 nature loss tiles (good balance for modeling)

---

### Task 3: Metadata Validation and Spatial Alignment
**What:** Verified that all raster files have correct band counts, coordinate systems (CRS), and spatial alignment across different data sources.

**Files:**
- `scripts/data_validation/03_inspect_metadata.py`
- `scripts/data_validation/04_check_spatial_alignment.py`
- `outputs/reports/metadata_validation.csv`
- `outputs/reports/spatial_alignment.txt`
- `scripts/slurm/step3_spatial_alignment.sh`

**Why:** Multi-source analysis requires that all data sources are properly georeferenced and aligned. Misalignments could cause the model to learn incorrect spatial relationships.

**Results:**
- **Band count validation:** 53/53 tiles passed
  - Sentinel-2: 126 bands ✓ (7 years × 2 quarters × 9 spectral bands)
  - PlanetScope: 42 bands ✓ (7 years × 2 quarters × 3 RGB bands)
  - VHR Google: 6 bands ✓ (2 time points × 3 RGB bands)
  - AlphaEarth: 448 bands (7 years × 64 embeddings per year)
  - Masks: 1 band ✓ (binary labels)
- **CRS validation:** All 53 tiles use EPSG:4326 (WGS84) consistently
- **Spatial alignment:** Perfect alignment achieved
  - 10m sources (Sentinel ↔ Mask, Sentinel ↔ AlphaEarth): Exact match in CRS, bounds, transform, and dimensions
  - High-res sources (VHR ↔ PlanetScope): Geographic bounds aligned within 0.000031° (well within tolerance)
- All tiles are ready for multi-source modeling without alignment issues

---

### Task 4: Comprehensive Data Quality Assessment
**What:** Performed detailed quality checks on all 53 tiles across all 5 data sources, analyzing NoData values, value ranges, brightness levels, and detecting any anomalies.

**Files:**
- `scripts/data_validation/05_check_data_quality.py`
- `scripts/slurm/step4_quality_checks.sh`
- `outputs/reports/data_quality.csv` (39 columns covering all metrics)
- `outputs/reports/data_quality_summary.txt`

**Why:** Low-quality satellite data (missing values, corrupted pixels, incorrect value ranges) can significantly degrade model performance. Early detection allows us to exclude problematic tiles or apply appropriate corrections.

**Results:** **All 53 tiles passed all quality checks! ✓**

**Sentinel-2 (10m resolution):**
- NoData: 0.31% (2018), 0.31% (2024) - minimal data gaps
- Value range: [120-7964] (2018), [73-7396] (2024) - within expected 0-10,000 reflectance range
- Status: ✓ PASS - No tiles with excessive NoData or out-of-range values

**PlanetScope (3-5m resolution):**
- NoData: 0.00% (both years) - perfect coverage
- Value range: [0-255] RGB - standard byte values
- Status: ✓ PASS - Excellent quality across all tiles

**VHR Google (1m resolution):**
- NoData: 0.04% (start), 0.10% (end) - negligible gaps
- Average brightness: 85.7/255 (start), 90.1/255 (end) - well-exposed imagery
- Status: ✓ PASS - High-quality imagery suitable for detailed analysis

**AlphaEarth (10m embeddings):**
- NoData: 0.00% - complete coverage
- Embedding range: [-0.509, 0.466] - normalized feature space
- Status: ✓ PASS - Pre-trained embeddings ready for use

**Land-Take Masks (10m binary):**
- All masks contain only {0, 1} ✓ - proper binary format
- Average change: 14.30% (median: 10.56%)
- Change range: [0.12%, 63.97%] - good diversity
- Distribution: 0 zero-change, 17 low (<5%), 30 moderate (5-30%), 6 high (≥30%)
- Status: ✓ PASS - Clean labels with diverse change levels

**Key Finding:** Dataset has exceptional quality with no problematic tiles. Ready for direct use in modeling.

---

### Task 5: Land-Take Mask Analysis and Class Balance Assessment
**What:** Analyzed the distribution of land-take changes across all tiles, computed class balance statistics, and identified spatial characteristics of change patches.

**Files:**
- `scripts/analysis/06_analyze_masks.py`
- `scripts/analysis/07_identify_edge_cases.py`
- `scripts/slurm/step5_mask_analysis.sh`
- `outputs/reports/mask_analysis.csv`
- `outputs/figures/mask_statistics.png`
- `outputs/reports/refids_zero_change.txt`
- `outputs/reports/refids_low_change.txt`
- `outputs/reports/refids_moderate_change.txt`
- `outputs/reports/refids_high_change.txt`

**Why:** Understanding class balance is critical for deep learning. Highly imbalanced datasets require special loss functions (Focal Loss, Dice Loss) and evaluation metrics (F1-score, IoU instead of accuracy). Knowing the spatial scale of changes helps determine appropriate model architectures and receptive field sizes.

**Results:**

**Class Balance:**
- Overall change ratio: 14.65% (all tiles combined)
- Median per-tile change: 10.56%
- Imbalance ratio: 1:6 (change:no-change)
- **Conclusion:** Significant class imbalance requires weighted loss functions

**Spatial Characteristics:**
- Total change patches: 239 across all tiles
- Average patches per tile: 4.5 patches
- **Median patch size: 24 pixels (~0.24 hectares at 10m resolution)**
- Mean patch size: 435 pixels (~4.35 hectares)
- Largest patch: 4,779 pixels (~48 hectares)
- **Distribution:** Heavy-tailed (most patches are tiny, few are very large - spans 4 orders of magnitude)

**Tile Categories:**
- Zero change (0%): 0 tiles (excellent - all tiles have signal)
- Low change (0-5%): 17 tiles (32%)
- Moderate change (5-30%): 30 tiles (57%)
- High change (≥30%): 6 tiles (11%)
- **Conclusion:** Good diversity across change levels for robust model training

**Critical Insights for Modeling:**
1. **Multi-scale challenge:** Model must detect patches from 24 pixels (tiny) to 4,779 pixels (large)
2. **Class imbalance strategy:** Use Focal Loss or Dice Loss, not standard cross-entropy
3. **Evaluation metrics:** F1-score and IoU are essential; accuracy alone will be misleading
4. **Train/val/test split:** Must stratify by change_ratio to ensure balanced representation
5. **No trivial tiles:** All 53 tiles contain some land-take, making the dataset valuable for training

---

### Task 6: Visual Inspection and Quality Verification
**What:** Generated high-quality visualizations of individual tiles and created a summary grid showing the diversity of land-take patterns across different change levels.

**Files:**
- `scripts/analysis/08_visualize_tiles.py`
- `scripts/analysis/09_create_summary_grid.py`
- `scripts/slurm/step6_visualizations.sh`
- `outputs/figures/tile_viz_*.png` (3 individual tiles, 4-5MB each)
- `outputs/figures/summary_grid.png` (27MB, 6 tiles × 3 views)

**Why:** Automated quality checks can miss subtle issues visible to human inspection. Visual verification ensures that masks align correctly with imagery, RGB colors are realistic, and temporal changes are visible. The summary grid provides a quick overview of dataset diversity for presentations and reports.

**Results:**

**Individual Tile Visualizations (2×2 layout):**
- **Panel [0,0]:** VHR 2018 RGB (1m resolution baseline)
- **Panel [0,1]:** VHR 2025 RGB (1m resolution current state)
- **Panel [1,0]:** Binary mask showing land-take areas
- **Panel [1,1]:** VHR 2025 with mask overlay (red highlights changes)

**Summary Grid:**
- Shows 6 representative tiles spanning low to high change levels
- Each tile: VHR 2018 | VHR 2025 | Mask overlay
- Provides one-page overview of dataset diversity

**Verification Results:**
- RGB colors are realistic and well-exposed (no saturation issues)
- Mask alignment is pixel-perfect (no spatial shifts detected)
- Temporal changes are clearly visible between 2018-2025
- Land-take patterns include: urban expansion, road construction, agricultural conversion, industrial development
- No data loading errors or corrupted visualizations

**Key Finding:** Dataset passes visual inspection with excellent image quality and proper mask alignment.

---

### Task 7: SLURM Integration for HPC Processing
**What:** Created SLURM batch scripts to efficiently run compute-intensive tasks on the NTNU Idun HPC cluster, with proper resource allocation and error handling.

**Files:**
- `scripts/slurm/step3_spatial_alignment.sh`
- `scripts/slurm/step4_quality_checks.sh`
- `scripts/slurm/step5_mask_analysis.sh`
- `scripts/slurm/step6_visualizations.sh`
- `docs/SLURM_INSTRUCTIONS.md`

**Why:** Loading and processing 53 tiles with multiple high-resolution bands (126+ bands per tile) requires significant RAM and compute time. SLURM allows us to leverage HPC resources with appropriate memory allocation and enables parallel processing.

**Results:**
- **Step 3 (Spatial Alignment):** 16GB RAM, 30 min - Validated geometric alignment
- **Step 4 (Quality Checks):** 16GB RAM, 30 min - Processed all sources and temporal endpoints
- **Step 5 (Mask Analysis):** 32GB RAM, 1 hour - Analyzed 239 patches across 53 tiles
- **Step 6 (Visualizations):** 48GB RAM, 45 min - Generated high-resolution PNGs

All jobs completed successfully with proper error logging and output capture. SLURM integration enables reproducible processing and efficient resource usage.

---

### Task 8: Comprehensive Documentation
**What:** Created detailed documentation covering datasets, workflows, HPC usage, and step-by-step instructions for reproducing the analysis.

**Files:**
- `README.md` - Project overview and quick start guide
- `docs/INSTRUCTIONS.md` - Step-by-step workflow with detailed requirements
- `docs/DATASETS.md` - Data source descriptions and technical specifications
- `docs/SLURM_INSTRUCTIONS.md` - HPC usage guide with examples

**Why:** Thorough documentation ensures that the project is reproducible, maintainable, and accessible to collaborators. It serves as a reference for future work and thesis writing.

**Results:**
- **README.md:** 211 lines covering project structure, quick start, dataset statistics, and workflow steps
- **INSTRUCTIONS.md:** 1,081 lines with detailed requirements, expected outputs, success criteria for each step
- **DATASETS.md:** 131 lines documenting band structures, resolutions, and file formats
- **SLURM_INSTRUCTIONS.md:** 78 lines with SLURM usage examples and resource requirements

Documentation includes:
- Dataset inventory and statistics
- File naming conventions and band structures
- Expected outputs and success criteria for each step
- Troubleshooting guidance and common issues
- HPC/SLURM usage with proper resource allocation
- Git commit history showing iterative development

---

## Project Statistics Summary

### Data Sources (5 sources validated)
| Source | Resolution | Bands | Coverage | Quality |
|--------|------------|-------|----------|---------|
| **Sentinel-2** | 10m | 126 | 53/53 tiles | ✓ PASS (0.31% NoData) |
| **PlanetScope** | 3-5m | 42 | 53/53 tiles | ✓ PASS (0% NoData) |
| **VHR Google** | 1m | 6 | 53/53 tiles | ✓ PASS (0.04% NoData) |
| **AlphaEarth** | 10m | 448 | 53/53 tiles | ✓ PASS (0% NoData) |
| **Land-Take Masks** | 10m | 1 | 53/53 tiles | ✓ PASS (binary valid) |

### Geographic Coverage
- **Countries:** 20 European nations
- **Top regions:** UK (8), France (8), Belgium (7), Netherlands (6)
- **Loss types:** Cropland (28 tiles), Nature (27 tiles) - balanced
- **Change types:** Residential (15), Uncertain (12), Agriculture (8), Transport (8), Industry (5)

### Quality Metrics
- **Overall data quality:** 53/53 tiles passed all checks ✓
- **Spatial alignment:** Perfect alignment achieved across all sources ✓
- **Band count validation:** 100% correct (53/53 tiles) ✓
- **Value range validation:** 100% within expected ranges ✓

### Land-Take Characteristics
- **Average change per tile:** 14.65%
- **Class imbalance ratio:** 1:6 (change:no-change)
- **Total change patches:** 239 across all tiles
- **Patch size range:** 1 pixel to 4,779 pixels (multi-scale challenge)
- **Median patch size:** 24 pixels (~0.24 hectares)
- **Tile diversity:** 17 low-change, 30 moderate-change, 6 high-change

---

## Key Findings and Insights

### 1. Dataset Quality
The dataset has **exceptional quality** with no tiles requiring exclusion:
- NoData < 0.5% across all sources (industry standard is < 5%)
- All values within expected ranges (no corrupted pixels)
- Perfect spatial alignment (no geometric distortions)
- Consistent CRS (EPSG:4326) across all sources

### 2. Class Imbalance Challenge
Land-take represents only 14.65% of pixels, creating a **significant imbalance**:
- Standard accuracy metrics will be misleading (predicting all "no-change" gives 85% accuracy)
- Requires specialized loss functions: Focal Loss (recommended) or Dice Loss
- Evaluation must use F1-score, IoU, precision-recall curves
- Stratified train/val/test split essential to maintain change distribution

### 3. Multi-Scale Spatial Challenge
Change patches span **4 orders of magnitude** (1 to 4,779 pixels):
- Model must detect both tiny changes (24 pixel median) and large developments
- U-Net architecture with skip connections is well-suited for multi-scale features
- May benefit from multi-scale input or pyramid approaches
- Small patches require high-resolution features (1m VHR imagery valuable)

### 4. Temporal Signal Strength
All 53 tiles show measurable land-take (**0 zero-change tiles**):
- Strong temporal signal between 2018-2025
- Multi-temporal features likely to improve performance
- 7 years of quarterly Sentinel-2 data available for temporal modeling
- Change detection between start/end years is well-posed problem

### 5. Geographic Diversity
20 countries with diverse landscape types:
- Good generalization potential across Europe
- Balanced loss types (cropland vs. nature)
- Multiple change types (residential, transport, industry, agriculture)
- Model trained on this dataset should generalize to new European locations

---

## Recommendations for Next Steps

### 1. Modeling Strategy

**Baseline Approach (Recommended for initial experiments):**
- **Resolution:** 10m (Sentinel-2/Mask native resolution)
- **Input data:** Sentinel-2 RGB + NIR (8 bands: 2018 Q2 + 2025 Q3)
- **Architecture:** U-Net with ResNet34 encoder (pre-trained on ImageNet)
- **Loss function:** Focal Loss (α=0.25, γ=2.0) or Dice Loss
- **Evaluation metrics:** F1-score, IoU, precision-recall curves (NOT accuracy alone)

**Advanced Experiments:**
- Multi-source fusion: Combine Sentinel-2 (10m) with VHR (1m, resampled)
- Temporal modeling: Use all 14 quarters instead of just start/end
- AlphaEarth embeddings: Leverage pre-trained features for improved performance
- Multi-scale architecture: Pyramid pooling or nested U-Net for patch size variation

### 2. Train/Val/Test Split
- **Split method:** Tile-level (not pixel-level) to avoid spatial leakage
- **Ratios:** 70/15/15 → 38 train / 8 validation / 7 test tiles
- **Stratification:** By change_ratio to maintain distribution across splits
- **Geographic consideration:** Optional stratification by country for spatial generalization testing

### 3. Class Imbalance Handling
Given 14.65% change pixels:
- **Loss function:** Focal Loss (recommended) or Dice Loss
- **Weighting:** Consider inverse frequency weighting (weight_change = 5.8×)
- **Augmentation:** Oversampling tiles with moderate/high change
- **Evaluation:** Report F1, IoU, precision, recall (accuracy is misleading)

### 4. Compute Requirements
For baseline U-Net model:
- **GPU:** 1× 80GB GPU (A100 or H100) recommended
- **RAM:** 64GB for data loading and preprocessing
- **Training time estimate:** 2-4 hours for 100 epochs (53 tiles, 10m resolution)
- **Storage:** ~20GB for processed tensors + checkpoints

### 5. Technical Implementation
**Recommended tools:**
- **Framework:** PyTorch + segmentation_models_pytorch
- **Data loading:** Custom Dataset class with rasterio
- **Augmentation:** albumentations (geometric + photometric)
- **Logging:** Weights & Biases (wandb) or TensorBoard
- **Checkpointing:** Save best model by validation IoU

---

## Files and Outputs Generated

### Configuration and Setup
- `config.py` - Central configuration file
- `environment/landtake_env.yml` - Conda environment
- `.gitignore` - Git ignore rules for data/outputs

### Data Validation Scripts (7 scripts)
1. `scripts/data_validation/01_inspect_filesystem.py` - File counting
2. `scripts/data_validation/02_extract_refids.py` - REFID extraction with metadata
3. `scripts/data_validation/03_inspect_metadata.py` - Band counts and CRS validation
4. `scripts/data_validation/04_check_spatial_alignment.py` - Geometric alignment checks
5. `scripts/data_validation/05_check_data_quality.py` - Comprehensive quality assessment

### Analysis Scripts (4 scripts)
6. `scripts/analysis/06_analyze_masks.py` - Mask statistics and patch analysis
7. `scripts/analysis/07_identify_edge_cases.py` - Change level categorization
8. `scripts/analysis/08_visualize_tiles.py` - Individual tile visualizations
9. `scripts/analysis/09_create_summary_grid.py` - Multi-tile summary grid

### SLURM Scripts (4 scripts)
- `scripts/slurm/step3_spatial_alignment.sh`
- `scripts/slurm/step4_quality_checks.sh`
- `scripts/slurm/step5_mask_analysis.sh`
- `scripts/slurm/step6_visualizations.sh`

### Reports (12 files)
- `outputs/reports/folder_structure.txt` - File inventory
- `outputs/reports/refid_list.txt` - Enhanced REFID list with metadata
- `outputs/reports/refid_presence.csv` - Data availability matrix
- `outputs/reports/metadata_validation.csv` - Band counts and dimensions
- `outputs/reports/spatial_alignment.txt` - Alignment validation results
- `outputs/reports/data_quality.csv` - Comprehensive quality metrics (39 columns)
- `outputs/reports/data_quality_summary.txt` - Human-readable quality report
- `outputs/reports/mask_analysis.csv` - Patch statistics
- `outputs/reports/refids_zero_change.txt` - Empty (no zero-change tiles)
- `outputs/reports/refids_low_change.txt` - 17 tiles
- `outputs/reports/refids_moderate_change.txt` - 30 tiles
- `outputs/reports/refids_high_change.txt` - 6 tiles

### Visualizations (5 files)
- `outputs/figures/mask_statistics.png` - Change distribution histograms
- `outputs/figures/summary_grid.png` - 6-tile overview (27MB)
- `outputs/figures/tile_viz_*.png` - 3 individual tiles (4-5MB each)

### Documentation (4 files)
- `README.md` - Project overview (211 lines)
- `docs/INSTRUCTIONS.md` - Detailed workflow (1,081 lines)
- `docs/DATASETS.md` - Data descriptions (131 lines)
- `docs/SLURM_INSTRUCTIONS.md` - HPC guide (78 lines)

---

## Technical Achievements

### 1. Robust Data Pipeline
Created a reproducible, well-documented pipeline for multi-source satellite data validation:
- Modular script design (each step is independent and reusable)
- Comprehensive error handling and logging
- HPC integration for scalability
- Configuration-driven approach (easy to adapt to new datasets)

### 2. Quality Assurance
Implemented multi-level validation ensuring dataset integrity:
- File system validation (availability)
- Metadata validation (band counts, CRS, dimensions)
- Spatial alignment validation (geometric consistency)
- Quality validation (NoData, value ranges, anomalies)
- Visual validation (human inspection of RGB composites)

### 3. Statistical Analysis
Performed comprehensive statistical characterization:
- Class balance quantification (14.65% change)
- Spatial pattern analysis (239 patches, multi-scale distribution)
- Temporal coverage assessment (7 years, quarterly)
- Geographic diversity documentation (20 countries)

### 4. Reproducibility
Ensured full reproducibility through:
- Version-controlled codebase
- Documented dependencies (conda environment)
- SLURM scripts with resource specifications
- Step-by-step instructions with success criteria
- Git commit history showing development process

---

## Conclusion

This project has successfully validated and characterized a high-quality multi-source satellite imagery dataset for land-take detection. All 53 tiles passed comprehensive quality checks, demonstrating excellent data integrity and readiness for deep learning model development.

**Key accomplishments:**
1. ✓ Validated 53 tiles across 5 data sources (100% pass rate)
2. ✓ Documented class imbalance (14.65%) and spatial characteristics (multi-scale patches)
3. ✓ Created reproducible pipeline with HPC integration
4. ✓ Generated comprehensive visualizations and reports
5. ✓ Identified critical modeling challenges (imbalance, multi-scale, stratification)

**Dataset is ready for modeling phase** with clear recommendations for:
- Baseline architecture (U-Net with ResNet34 encoder)
- Loss function (Focal Loss for class imbalance)
- Evaluation metrics (F1-score, IoU, not accuracy)
- Train/val/test split strategy (tile-level, stratified)

This work provides a solid foundation for developing robust deep learning models for automated land-take detection, with potential applications in environmental monitoring, urban planning, and habitat loss assessment across Europe.

---

## Contact and References

**Project Contact:** tmstorma@stud.ntnu.no
**Data Provider:** Zander Venter (zander.venter@nina.no)
**Institution:** NINA (Norwegian Institute for Nature Research)
**Data Source:** HABLOSS Project - https://habloss.eu/
**Repository:** /cluster/home/tmstorma/NINA_fordypningsoppgave

**Git History:**
- `78bf97e` - refactor: Enhance data validation and visualizations
- `cafc729` - docs: Reorganize documentation and add comprehensive README
- `3e42d21` - feat: Add mask analysis and VHR visualization pipeline
- `43a0839` - feat: Implement data validation pipeline with VHR quality checks
- `9f03d9c` - feat: Add project configuration and environment setup