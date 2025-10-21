# START_HERE: Data Understanding & Sanity Checks
## Land-Take Detection/Segmentation Dataset

**Project:** HABLOSS – European Land Take Monitoring  
**Author:** Zander Venter (NINA)  
**Your Role:** Master's student – land-take detection/segmentation  
**Last Updated:** October 2025

---

## 📋 Instructions for AI Assistant

This document guides a **step-by-step data exploration workflow**. When the user asks you to proceed with a step:

1. **Read the step's requirements carefully**
2. **Implement the code as specified** (create scripts, functions, analysis)
3. **Execute the code** and show outputs
4. **Wait for user confirmation** before proceeding to next step
5. **Never skip steps** - each builds on the previous

The user will say things like:
- "Proceed with Step 1"
- "Execute Step 3.2"
- "Continue to next step"

Always confirm what you're about to do before executing.

### ⚠️ IMPORTANT: Compute Resources
**The user does NOT have GPU or extensive local storage.** Before executing any compute-intensive or storage-heavy operations:
1. **Warn the user** about resource requirements
2. **Suggest creating a SLURM script** for HPC execution if needed
3. **Wait for confirmation** before proceeding

**Steps that may require HPC:**
- Step 4: Loading all bands for quality checks (memory intensive)
- Step 5: Processing ALL 55 masks (I/O intensive)
- Step 6: Creating visualizations with large rasters (memory + storage)
- Step 8+: Any model training or large-scale data processing

---

## Quick Dataset Overview

Read the [Dataset file](DATASETS.md) for information about the dataset.


### What the User Has
- **55 annotated tiles** (650m × 650m) across Europe
- **4 remote sensing sources** at different resolutions
- **Binary masks**: Land-take change (1) vs. no change (0)
- **Time span**: 2018–2024 (7 years, quarterly)

### Data Sources

| Source | Resolution | Bands | Key Info |
|--------|------------|-------|----------|
| **Sentinel-2** | 10m | 126 | Multi-spectral time-series (7yr × 2q × 9bands) |
| **PlanetScope** | 3-5m | 42 | RGB time-series (7yr × 2q × 3bands) |
| **Google VHR** | 1m | 6 | Start/end year RGB only |
| **AlphaEarth** | 10m | varies | Pre-trained embeddings (7 years) |
| **Masks** | 10m | 1 | Ground truth labels |

### Critical Data Notes
⚠️ **PlanetScope anomaly:** Folder contains 1,967 images, but only 55 match annotated REFIDs  
⚠️ **Resolution mismatch:** Data ranges from 1m to 10m – requires resampling strategy  
⚠️ **Projection:** All data in EPSG:4326 (WGS84 lat/lon)  
⚠️ **Reflectance units:** Sentinel-2 values are TOP-OF-ATMOSPHERE × 10,000

---

## STEP 1: Environment Setup

### Objective
Set up Python environment and project structure.

### Requirements

**1.1 - Create conda environment file:**
- Filename: `landtake_env.yml`
- Python version: 3.10
- Required packages:
  - rasterio, gdal
  - numpy, pandas, geopandas
  - matplotlib, seaborn
  - scikit-learn
  - tqdm
  - scipy

**1.2 - Create project directory structure:**
```
project_root/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
├── outputs/
│   ├── figures/
│   └── reports/
├── config.py
└── README.md
```

**1.3 - Create config.py:**
```python
# User will fill in actual path
DATA_ROOT = "/path/to/data"  # PLACEHOLDER - user must update

# Folder names
FOLDERS = {
    'sentinel': 'Sentinel',
    'planetscope': 'PlanetScope',
    'vhr': 'VHR_google',
    'alphaearth': 'AlphaEarth',
    'masks': 'Land_take_masks'
}
```

**1.4 - Test imports:**
Create a simple test script that imports all critical libraries.

### Deliverables
- [ ] `landtake_env.yml` created
- [ ] Directory structure created
- [ ] `config.py` created (user will update DATA_ROOT)
- [ ] Import test passes

### Success Criteria
All imports successful, directory structure exists.

---

## STEP 2: File System Exploration

### Objective
Discover what files exist and extract the 55 REFID identifiers.

### Requirements

**2.1 - Inspect folder structure:**

Create `src/01_inspect_filesystem.py` that:
- Imports: `pathlib`, `os`, `collections`, `DATA_ROOT` and `FOLDERS` from config
- Defines `list_data_folders(data_root)`:
  - Lists all subdirectories matching FOLDERS values
  - For each folder, counts `.tif` files
  - Returns dict: `{folder_name: file_count}`
  - Prints formatted table with folder names and counts
- In `if __name__ == "__main__"`:
  - Calls function with DATA_ROOT
  - Saves results to `outputs/reports/folder_structure.txt`
  - Includes error handling for missing folders

**2.2 - Extract REFIDs:**

Create `src/02_extract_refids.py` that:
- Imports: `pathlib`, `re`, `pandas`, config
- Defines `extract_refid(filename)`:
  - Uses regex pattern: `r'(REFID_\d+)'`
  - Returns REFID string or None
- Defines `get_refids_by_folder(data_root)`:
  - For each folder in FOLDERS (except PlanetScope)
  - Extracts unique REFIDs from all `.tif` filenames
  - Returns: `{folder_name: set_of_refids}`
- Defines `find_common_refids(refid_dict)`:
  - Finds intersection of REFIDs across all folders
  - Returns sorted list
- In main:
  - Gets REFIDs from: Sentinel, VHR_google, AlphaEarth, Land_take_masks
  - Finds common REFIDs
  - Prints: count, first 5 REFIDs
  - Saves full list to `outputs/reports/refid_list.txt` (one per line)
  - Also creates CSV: `outputs/reports/refid_presence.csv` with columns:
    - refid, in_sentinel, in_vhr, in_alphaearth, in_masks (boolean flags)
  - **Validation**: Prints warning if common REFID count ≠ 55

### Expected Outputs
```
✓ Sentinel: 55 REFIDs
✓ VHR_google: 55 REFIDs
✓ AlphaEarth: 55 REFIDs
✓ Land_take_masks: 55 REFIDs
✓ Common REFIDs: 55
```

### Deliverables
- [ ] `src/01_inspect_filesystem.py` created and executed
- [ ] `src/02_extract_refids.py` created and executed
- [ ] `outputs/reports/folder_structure.txt` exists
- [ ] `outputs/reports/refid_list.txt` contains exactly 55 lines
- [ ] `outputs/reports/refid_presence.csv` exists

### Success Criteria
Exactly 55 common REFIDs identified across all data sources.

---

## STEP 3: Raster Metadata Validation

### Objective
Verify band counts, spatial dimensions, CRS, and alignment.

### Expected Band Counts

| Source | Expected Bands | Calculation |
|--------|----------------|-------------|
| Sentinel-2 | 126 | 7 years × 2 quarters × 9 bands |
| PlanetScope | 42 | 7 years × 2 quarters × 3 bands |
| VHR Google | 6 | 2 time points × 3 RGB bands |
| AlphaEarth | varies | 7 years × N embeddings (document actual) |
| Masks | 1 | Binary label |

### Requirements

**3.1 - Metadata inspector:**

Create `src/03_inspect_metadata.py` that:
- Imports: `rasterio`, `pandas`, `pathlib`, `numpy`, config
- Loads `outputs/reports/refid_list.txt`
- Defines `inspect_raster_metadata(filepath)`:
  - Opens raster with rasterio
  - Extracts: `count` (bands), `width`, `height`, `crs`, `bounds`, `dtype`
  - Returns as dictionary
- Defines `validate_tile_metadata(data_root, refid)`:
  - Constructs file paths for all 5 sources:
    - Sentinel: `{refid}_RGBNIRRSWIRQ_Mosaic.tif`
    - PlanetScope: `{refid}_RGBQ_Mosaic.tif`
    - VHR: `{refid}_RGBY_Mosaic.tif`
    - AlphaEarth: `{refid}_VEY_Mosaic.tif`
    - Mask: `{refid}_mask.tif`
  - Calls `inspect_raster_metadata()` for each existing file
  - Returns dict with metadata per source
- In main:
  - Process first 10 REFIDs from list
  - For each REFID:
    - Get metadata
    - Check band counts against expected values
    - Print formatted summary table
    - Flag mismatches with ⚠️
  - Create DataFrame with columns:
    - refid, sentinel_bands, sentinel_dims, vhr_bands, vhr_dims, mask_bands, mask_dims, all_crs_match, all_checks_passed
  - Save to `outputs/reports/metadata_validation.csv`
- Use tqdm progress bar

**3.2 - Spatial alignment check:**

Create `src/04_check_spatial_alignment.py` that:
- For first 5 REFIDs:
  - Load Sentinel and mask metadata
  - Compare:
    - Bounds (within 0.0001 degree tolerance)
    - Transform (affine transformation matrix)
    - CRS (must match exactly)
  - Print ✓ if aligned, ⚠️ with details if misaligned
- Save alignment report to `outputs/reports/spatial_alignment.txt`
- Include summary: X/5 tiles properly aligned

### Deliverables
- [ ] `src/03_inspect_metadata.py` executed
- [ ] `src/04_check_spatial_alignment.py` executed
- [ ] `outputs/reports/metadata_validation.csv` exists
- [ ] `outputs/reports/spatial_alignment.txt` exists
- [ ] Band count validation passed
- [ ] Spatial alignment confirmed

### Success Criteria
All tiles have correct band counts, all use EPSG:4326, Sentinel/masks are spatially aligned.

---

## STEP 4: Data Quality Checks

### Objective
Detect NoData values, validate value ranges, identify outliers.

### Requirements

**4.1 - Quality assessment:**

Create `src/05_check_data_quality.py` that:
- For first 10 REFIDs:
  
  **Sentinel-2 checks:**
  - Load first 9 bands only (2018 Q2: bands 1-9)
  - Count NaN/NoData pixels → compute percentage
  - For valid (non-NaN) pixels:
    - Compute: min, max, mean, std
    - Expected range: 0-10,000 (reflectance × 10k)
    - Flag if: min < 0 or max > 15,000
  
  **Mask checks:**
  - Load mask (single band)
  - Get unique values with `np.unique()`
  - Verify only {0, 1} present
  - Compute: change_pixels / total_pixels → change percentage
  - Flag if any values other than 0 or 1 found

- Create DataFrame with columns:
  - refid
  - sentinel_nodata_pct
  - sentinel_min, sentinel_max, sentinel_mean, sentinel_std
  - mask_unique_values (as string)
  - mask_change_pct
  - quality_issues (list of any problems found)

- Save to `outputs/reports/data_quality.csv`

- Print summary:
  - Average NoData percentage across tiles
  - Number of tiles with >5% NoData
  - Number of tiles with invalid mask values
  - Average change percentage

- Use tqdm for progress

### Deliverables
- [ ] `src/05_check_data_quality.py` executed
- [ ] `outputs/reports/data_quality.csv` exists
- [ ] Quality issues identified (if any)

### Success Criteria
Data quality assessed, any issues documented.

---

## STEP 5: Mask Analysis & Class Balance

### Objective
Understand label distribution and prepare for class imbalance handling.

### Requirements

**5.1 - Comprehensive mask analysis:**

Create `src/06_analyze_masks.py` that:
- For ALL 55 REFIDs:
  - Load mask
  - Compute:
    - total_pixels
    - change_pixels (mask == 1)
    - no_change_pixels (mask == 0)
    - change_ratio (as percentage)
  - Use `scipy.ndimage.label()` to find connected change regions
  - For connected components:
    - Count number of patches
    - Get size of each patch (in pixels)
    - Find largest patch size
    - Compute mean patch size

- Create DataFrame with columns:
  - refid, total_pixels, change_pixels, change_ratio, num_change_patches, max_patch_size, mean_patch_size

- Compute global statistics:
  - Overall change ratio (all tiles combined)
  - Median change ratio per tile
  - Count tiles with: 0% change, >50% change

- Save to `outputs/reports/mask_analysis.csv`

- Create visualization (`matplotlib` figure with 3 subplots):
  - Subplot 1: Histogram of change_ratio (30 bins)
  - Subplot 2: Histogram of num_change_patches (30 bins)
  - Subplot 3: Boxplot of change_ratio
  - Add axis labels, titles
  - Save to `outputs/figures/mask_statistics.png` at 150 DPI

**5.2 - Identify edge cases:**

Create `src/07_identify_edge_cases.py` that:
- Loads `outputs/reports/mask_analysis.csv`
- Categorizes tiles:
  - zero_change: change_ratio == 0
  - low_change: 0 < change_ratio < 5%
  - moderate_change: 5% ≤ change_ratio < 30%
  - high_change: change_ratio ≥ 30%
- For each category:
  - Count tiles
  - List REFIDs
  - Print summary statistics
- Save REFIDs to separate files:
  - `outputs/reports/refids_zero_change.txt`
  - `outputs/reports/refids_low_change.txt`
  - `outputs/reports/refids_moderate_change.txt`
  - `outputs/reports/refids_high_change.txt`
- Print modeling recommendations based on distribution

### Deliverables
- [ ] `src/06_analyze_masks.py` executed
- [ ] `src/07_identify_edge_cases.py` executed
- [ ] `outputs/reports/mask_analysis.csv` exists
- [ ] `outputs/figures/mask_statistics.png` created
- [ ] Edge case files created (4 files)

### Success Criteria
Class balance understood, edge cases identified, imbalance strategy clear.

---

## STEP 6: Visual Inspection

### Objective
Visually verify data quality and mask alignment through imagery.

### Sentinel-2 Band Structure Reference

126 bands organized as:
- Bands 1-18: 2018 (Q2 + Q3)
- Bands 19-36: 2019 (Q2 + Q3)
- ...
- Bands 109-126: 2024 (Q2 + Q3)

Each quarter has 9 bands in order:
1. blue, 2. green, 3. red, 4. R1, 5. R2, 6. R3, 7. nir, 8. swir1, 9. swir2

**Key band indices (1-indexed for rasterio):**
- 2018 Q2 RGB: bands [3, 2, 1]
- 2024 Q3 RGB: bands [111, 110, 109]
- 2024 Q3 NIR: band 115

### Requirements

**6.1 - Tile visualization:**

Create `src/08_visualize_tiles.py` that:
- Defines `visualize_tile(data_root, refid, save_path)`:
  
  Creates 2×3 subplot figure (figsize=(18, 12)):
  
  **Row 1 - RGB composites:**
  - [0,0]: 2018 Q2 RGB (Sentinel bands [3,2,1])
    - Read with rasterio: `src.read([3, 2, 1])`
    - Normalize: divide by 10000, clip to [0, 1]
    - Transpose to (H, W, C) for display
  - [0,1]: 2024 Q3 RGB (Sentinel bands [111,110,109])
  - [0,2]: 2024 Q3 False Color NIR-R-G (bands [115,111,110])
  
  **Row 2 - Masks & change:**
  - [1,0]: Mask alone
    - Colormap: 'RdYlGn_r' (red=change, green=no-change)
  - [1,1]: 2024 RGB with mask overlay
    - Show RGB as base
    - Use `np.ma.masked_where(mask == 0, mask)` 
    - Overlay with 'Reds' colormap at alpha=0.5
  - [1,2]: RGB difference (2024 - 2018)
    - Compute: mean across RGB channels
    - Colormap: 'RdBu', vmin=-0.2, vmax=0.2
  
  - Add titles to each subplot
  - Set figure suptitle to REFID
  - Turn off axes
  - Save to save_path at 150 DPI
  - Use tight_layout()

- In main:
  - Load refid_list.txt
  - Visualize first 3 REFIDs
  - Save to `outputs/figures/tile_viz_{refid}.png`
  - Print confirmation for each

**6.2 - Summary grid:**

Create `src/09_create_summary_grid.py` that:
- Loads edge case files from Step 5.2
- Selects 6 diverse REFIDs:
  - 2 from zero_change (if available)
  - 2 from moderate_change
  - 2 from high_change
- For each REFID, creates compact visualization:
  - 1×3 layout: [2018 RGB | 2024 RGB | Mask overlay]
  - Small subplot size: ~4×3 inches
- Arranges all 6 into 6×3 grid
- Adds row labels:
  - REFID name
  - Change percentage (from mask_analysis.csv)
- Saves to `outputs/figures/summary_grid.png` at 300 DPI
- This creates a one-page dataset overview

### Deliverables
- [ ] `src/08_visualize_tiles.py` executed
- [ ] `src/09_create_summary_grid.py` executed
- [ ] 3 individual tile visualizations created
- [ ] `outputs/figures/summary_grid.png` created

### Success Criteria
Visualizations show:
- Realistic RGB colors
- Mask alignment with visible changes
- No obvious data loading errors
- Clear temporal changes between 2018-2024

**User should manually review images before proceeding.**

---

## STEP 7: Generate Comprehensive Report

### Objective
Compile all findings into a final markdown report.

### Requirements

**7.1 - Compile data report:**

Create `src/10_generate_report.py` that:
- Loads all CSV/TXT reports from `outputs/reports/`:
  - folder_structure.txt
  - refid_list.txt
  - metadata_validation.csv
  - data_quality.csv
  - mask_analysis.csv
  - Edge case files

- Generates markdown report: `outputs/reports/DATASET_REPORT.md`

**Report structure:**

```markdown
# Land-Take Detection Dataset Report
*Generated: [timestamp]*

## Executive Summary
- Total annotated tiles: X
- Date range: 2018-2024 (7 years, bi-quarterly)
- Overall change ratio: X.X%
- Data quality status: [PASS/ISSUES FOUND]

## Dataset Inventory
[Table showing file counts per folder]

## REFID Coverage
- Common REFIDs across all sources: 55
- Total PlanetScope images: 1,967 (only 55 match REFIDs)

## Metadata Validation
### Band Counts
[Table: Source | Expected | Actual | Status]

### Spatial Properties
- Tile size: ~650m × 650m
- Sentinel/Mask resolution: 10m (~65×65 pixels)
- VHR resolution: 1m (~650×650 pixels)
- PlanetScope resolution: 3-5m (~130-217 pixels)
- Projection: EPSG:4326 (all sources)

### Spatial Alignment
[Summary of alignment check results]

## Data Quality Assessment
### NoData Analysis
- Average NoData percentage: X.X%
- Tiles with >5% NoData: X tiles
- Tiles with >10% NoData: X tiles

### Value Range Validation
- Sentinel-2 range: [min, max] (expected: 0-10,000)
- Tiles with out-of-range values: X

### Mask Validation
- All masks contain only {0, 1}: [YES/NO]
- Invalid masks: [list if any]

## Label Analysis
### Class Balance
- Overall change pixels: X.X% (highly imbalanced)
- Median tile change ratio: X.X%
- Tiles with 0% change: X
- Tiles with >50% change: X

### Edge Case Distribution
- Zero change: X tiles
- Low change (<5%): X tiles
- Moderate change (5-30%): X tiles
- High change (>30%): X tiles

### Spatial Characteristics
- Average change patches per tile: X.X
- Average patch size: X pixels
- Largest single change patch: X pixels

## Key Findings
1. [Most important observation]
2. [Second key finding]
3. [Third key finding]
...

## Recommendations for Modeling

### Resolution Strategy
**Recommendation:** Use 10m (Sentinel-2 native resolution)
- Pros: Matches mask resolution, manageable size
- Cons: Loses fine detail from VHR
- Alternative: Multi-scale fusion (advanced)

### Class Imbalance Handling
Given X.X% change pixels:
- Use weighted loss functions (Focal Loss recommended)
- Consider oversampling change patches
- Evaluate with F1-score, IoU (not accuracy)

### Train/Val/Test Split
- Split by REFID (tile-level), not pixels
- Recommended: 70/15/15 (38/8/9 tiles)
- Stratify by change_ratio distribution

### Input Data Options
1. **Baseline:** Sentinel-2 only (RGB + NIR, start + end)
2. **Intermediate:** Sentinel-2 + VHR (resampled to 10m)
3. **Advanced:** Multi-source fusion + temporal features

## Data Quality Issues
[List any tiles with problems, recommendations to exclude]

## Next Steps
1. Create train/validation/test split (stratified by change ratio)
2. Implement PyTorch DataLoader
3. Develop baseline U-Net model
4. Define evaluation metrics (F1, IoU, precision-recall)

## Appendix
### Contact
Dataset: Zander Venter (zander.venter@nina.no)
Analysis: [Student name]
Date: [Date]
```

- Print report to console
- Save to file

**7.2 - Create project README:**

Create comprehensive `README.md` in project root:
- Project description
- Dataset overview
- Directory structure
- Script execution order (01-10)
- How to reproduce analysis
- References

### Deliverables
- [ ] `src/10_generate_report.py` executed
- [ ] `outputs/reports/DATASET_REPORT.md` created
- [ ] `README.md` created in project root

### Success Criteria
Report is comprehensive, actionable, and suitable for thesis appendix.

---

## STEP 8: Prepare Modeling Checklist

### Objective
Create a decision document for the modeling phase.

### Requirements

Create `outputs/reports/MODELING_DECISIONS.md` that:

```markdown
# Modeling Phase Decisions

## Status: Data Understanding Complete ✓

## Critical Decisions Required

### 1. Resolution Choice
- [ ] Use 10m (Sentinel native) - **RECOMMENDED FOR BASELINE**
- [ ] Use 1m (VHR native) - requires upsampling, high compute cost
- [ ] Multi-scale approach - advanced, for later experiments

**Decision:** _________________
**Rationale:** _________________

### 2. Input Data Selection
- [ ] Sentinel-2 only (simplest, fastest baseline)
- [ ] Sentinel-2 + VHR (bi-temporal, resampled)
- [ ] All sources (complex, highest potential)

**Decision:** _________________
**Bands to use:** _________________

### 3. Temporal Features
- [ ] Start + end only (2018 Q2 + 2024 Q3)
- [ ] All quarterly mosaics (14 time points)
- [ ] Derived temporal indices (NDVI change, etc.)

**Decision:** _________________

### 4. Train/Val/Test Split Strategy
- [ ] Random 70/15/15 split
- [ ] Stratified by change_ratio
- [ ] Geographic/spatial stratification

**Decision:** _________________
**Rationale:** _________________

### 5. Class Imbalance Strategy
Given X.X% change pixels:
- [ ] Weighted Binary Cross-Entropy
- [ ] Focal Loss (recommended)
- [ ] Dice Loss
- [ ] Combined loss

**Decision:** _________________

### 6. Baseline Model
- [ ] U-Net with ResNet34 encoder (recommended)
- [ ] DeepLabV3+
- [ ] Custom architecture

**Framework:** PyTorch + segmentation_models_pytorch
**Input shape:** [________________]

## Compute Resources
- GPU available: [YES/NO]
- GPU type: [________________]
- RAM: [________________]
- Training time budget: [________________]

## Timeline
- Baseline model: [________ days]
- Experiments: [________ days]
- Thesis writing: [________ days]

## Next Scripts to Create
1. `11_create_train_val_test_split.py`
2. `12_dataset_class.py` (PyTorch Dataset)
3. `13_test_dataloader.py` (verify loading)
4. `14_baseline_model.py` (U-Net)
5. `15_train.py` (training loop)
6. `16_evaluate.py` (metrics, visualization)

## Sign-off
Data understanding phase completed on: [DATE]
Ready to proceed to modeling: [YES/NO]
Issues to address first: [LIST IF ANY]
```

### Deliverables
- [ ] `outputs/reports/MODELING_DECISIONS.md` created
- [ ] User has made all key decisions
- [ ] Compute resources documented
- [ ] Timeline established

### Success Criteria
All decisions documented, ready for modeling phase.

---

## Appendix: Quick Reference

### File Naming Patterns

**Sentinel-2:** `REFID_XXX_RGBNIRRSWIRQ_Mosaic.tif`  
**PlanetScope:** `REFID_XXX_RGBQ_Mosaic.tif`  
**VHR Google:** `REFID_XXX_RGBY_Mosaic.tif`  
**AlphaEarth:** `REFID_XXX_VEY_Mosaic.tif`  
**Masks:** `REFID_XXX_mask.tif`

### Sentinel-2 Band Reference

| Quarter | Blue | Green | Red | R1 | R2 | R3 | NIR | SWIR1 | SWIR2 |
|---------|------|-------|-----|----|----|----|-----|-------|-------|
| 2018 Q2 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| 2018 Q3 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| 2024 Q2 | 109 | 110 | 111 | 112 | 113 | 114 | 115 | 116 | 117 |
| 2024 Q3 | 118 | 119 | 120 | 121 | 122 | 123 | 124 | 125 | 126 |

### Common Spectral Indices

**NDVI (Vegetation):**
```python
ndvi = (nir - red) / (nir + red + 1e-8)
```

**NDBI (Urban):**
```python
ndbi = (swir1 - nir) / (swir1 + nir + 1e-8)
```

**NDWI (Water):**
```python
ndwi = (green - nir) / (green + nir + 1e-8)
```

### Troubleshooting

| Issue | Solution |
|-------|----------|
| "No such file or directory" | Verify DATA_ROOT in config.py |
| "Rasterio can't open file" | Check file: `gdalinfo filename.tif` |
| "Unexpected band count" | Document discrepancy, may exclude tile |
| "Memory error" | Read bands in chunks, not all at once |
| "Mask values not 0 or 1" | Check with `np.unique()`, may need threshold |

---

## Progress Tracking

### Phase 1: Setup
- [ ] Step 1: Environment setup complete

### Phase 2: Discovery
- [ ] Step 2: File system explored, 55 REFIDs extracted
- [ ] Step 3: Metadata validated
- [ ] Step 4: Data quality assessed

### Phase 3: Analysis
- [ ] Step 5: Mask analysis & class balance computed
- [ ] Step 6: Visual inspection complete

### Phase 4: Documentation
- [ ] Step 7: Comprehensive report generated
- [ ] Step 8: Modeling decisions documented

**✅ Data Understanding Phase Complete**

---

**Document Version:** 1.0  
**Last Updated:** October 2025  