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
├── scripts/
│   ├── data_validation/
│   ├── analysis/
│   ├── utils/
│   └── slurm/
├── outputs/
│   ├── figures/
│   ├── reports/
│   └── slurm_outputs/
├── environment/
├── docs/
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

Create `scripts/data_validation/01_inspect_filesystem.py` that:
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

Create `scripts/data_validation/02_extract_refids.py` that:
- Imports: `pathlib`, `re`, `pandas`, `json`, config
- Defines `extract_refid(filename)`:
  - Uses regex patterns for both `REFID_XXX` and coordinate-based IDs
  - Returns REFID string or None
- Defines `get_refids_by_folder(data_root, include_planetscope=False)`:
  - For each folder in FOLDERS (excludes PlanetScope by default for common calculation)
  - Extracts unique REFIDs from all `.tif` filenames
  - Returns: `{folder_name: set_of_refids}`
- Defines `get_planetscope_refids(data_root)`:
  - Separately extracts REFIDs from PlanetScope folder
  - Returns: set of PlanetScope REFIDs
- Defines `find_common_refids(refid_dict)`:
  - Finds intersection of REFIDs across main folders (excluding PlanetScope)
  - Returns sorted list
- Defines `load_geojson_metadata(geojson_path)`:
  - Loads `land_take_bboxes_650m_v1.geojson`
  - Extracts: country, loss type (r), change_type for each PLOTID
  - Returns: `{plotid: {country, loss_type, change_type}}`
- Defines `save_refid_list(refids, output_path, geojson_metadata=None)`:
  - Creates enhanced text file with:
    - Descriptive header with dataset info
    - Metadata table with columns: REFID, Country, Loss Type, Change Type
    - Professional formatting suitable for documentation
- Defines `save_refid_presence_csv(refid_dict, common_refids, planetscope_refids, output_path)`:
  - Creates CSV with columns: refid, in_sentinel, in_vhrgoogle, in_alphaearth, in_landtakemasks, in_planetscope
  - Returns DataFrame
- In main:
  - Gets REFIDs from: Sentinel, VHR_google, AlphaEarth, Land_take_masks (excludes PlanetScope)
  - Finds common REFIDs across main sources
  - Separately checks PlanetScope availability for common REFIDs
  - Loads GeoJSON metadata
  - Prints: count, first 5 REFIDs, PlanetScope coverage, metadata summary
  - Identifies missing REFIDs from each source
  - Saves enhanced `refid_list.txt` with metadata table
  - Creates `refid_presence.csv` with 6 columns (including PlanetScope)
  - **Validation**: Prints warning if common REFID count ≠ 55

### Expected Outputs
```
✓ Sentinel: 54 REFIDs (1 missing)
✓ VHR_google: 54 REFIDs (1 missing)
✓ AlphaEarth: 55 REFIDs
✓ Land_take_masks: 55 REFIDs
✓ Common REFIDs: 53
⚠️ WARNING: Expected 55, found 53
  - Missing from Sentinel: a4-62484266638608_51-98215379896622
  - Missing from VHR_google: a33-32266776718259_47-77995870945647
✓ PlanetScope: All 53 common REFIDs have PlanetScope data
✓ GeoJSON metadata: Loaded for all 53 tiles
```

### Actual Results
**53 validated REFIDs** with complete multi-source coverage:
- **Geographic**: 20 European countries (GBR: 8, FRA: 8, BEL: 7, NLD: 6, others: 1-4)
- **Loss types**: Cropland loss (28), Nature loss (27) - balanced
- **Change types**: Residential (15), Uncertain (12), Agriculture (8), Transport (8), Industry (5)

### Deliverables
- [x] `scripts/data_validation/01_inspect_filesystem.py` created and executed
- [x] `scripts/data_validation/02_extract_refids.py` created and executed
- [x] `outputs/reports/folder_structure.txt` exists
- [x] `outputs/reports/refid_list.txt` created with enhanced format:
  - Descriptive header with dataset information
  - Metadata table (REFID, Country, Loss Type, Change Type)
  - Contains 53 validated tiles (not 55 due to missing data)
- [x] `outputs/reports/refid_presence.csv` exists with 6 columns:
  - `refid`, `in_sentinel`, `in_vhrgoogle`, `in_alphaearth`, `in_landtakemasks`, `in_planetscope`

### Success Criteria
Common REFIDs identified across all main data sources. **Note:** Found 53 instead of expected 55 due to 2 tiles with incomplete data (acceptable for modeling with validated subset).

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

Create `scripts/data_validation/03_inspect_metadata.py` that:
- Imports: `rasterio`, `pandas`, `pathlib`, `tqdm`, config
- Loads `outputs/reports/refid_list.txt`
  - Parses enhanced format (skips header, extracts REFIDs from metadata table)
  - Filters lines to get only valid REFID entries
- Defines `inspect_raster_metadata(filepath)`:
  - Opens raster with rasterio
  - Extracts: `count` (bands), `width`, `height`, `crs`, `bounds`, `dtype`, `transform`
  - Returns as dictionary
  - Handles errors gracefully (returns None if file can't be read)
- Defines `construct_filepath(data_root, refid, data_type)`:
  - Constructs filepath for given REFID and data type
  - Handles different filename patterns for each source
- Defines `validate_tile_metadata(data_root, refid)`:
  - Constructs file paths for all 5 main sources:
    - Sentinel: `{refid}_RGBNIRRSWIRQ_Mosaic.tif`
    - PlanetScope: `{refid}_RGBQ_Mosaic.tif`
    - VHR: `{refid}_RGBY_Mosaic.tif`
    - AlphaEarth: `{refid}_VEY_Mosaic.tif`
    - Mask: `{refid}_mask.tif`
  - Calls `inspect_raster_metadata()` for each existing file
  - Returns dict with metadata per source
- Defines `check_band_count(actual, expected, data_type)`:
  - Validates band count against expected value
  - Returns (is_valid, formatted_message)
  - Handles None values for non-validated sources (AlphaEarth)
- In main:
  - Process **ALL 53 REFIDs** from list (not just 10)
  - For each REFID with tqdm progress bar:
    - Get metadata for all sources
    - Extract band counts, dimensions, CRS
    - Check if all CRS values match
    - Validate band counts against expected values
    - Flag mismatches with ⚠️
  - Create DataFrame with columns:
    - refid, sentinel_bands, sentinel_dims, sentinel_crs
    - planetscope_bands, planetscope_dims, planetscope_crs
    - vhr_bands, vhr_dims, vhr_crs
    - alphaearth_bands, alphaearth_dims, alphaearth_crs
    - mask_bands, mask_dims, mask_crs
    - all_crs_match, all_checks_passed
  - Print detailed summary table for each tile
  - Save to `outputs/reports/metadata_validation.csv`
  - Print overall statistics:
    - Total tiles processed
    - Number passing all checks
    - Number with CRS matches

**3.2 - Comprehensive spatial alignment check:**

Create `scripts/data_validation/04_check_spatial_alignment.py` that:
- Loads enhanced REFID list (parses format from Step 2)
- For first 5 REFIDs, performs three alignment checks:

  **Check 1: 10m resolution sources (Sentinel ↔ Mask, Sentinel ↔ AlphaEarth)**
  - Compare CRS (must match exactly)
  - Compare bounds (within 0.0001° tolerance)
  - Compare transforms (affine transformation matrix, 1e-10 tolerance)
  - Compare dimensions (must match exactly for same resolution)
  - All criteria must pass for "ALIGNED" status

  **Check 2: High-resolution sources (VHR ↔ PlanetScope)**
  - Compare CRS (must match exactly)
  - Compare bounds coverage (within 0.001° tolerance, more lenient for different resolutions)
  - Dimensions expected to differ due to resolution differences

  **Check 3: Overall CRS consistency**
  - Verify all 5 sources use the same CRS

- Print detailed results with three sections per tile:
  - 10m resolution group status
  - High-resolution group status
  - Overall dimensions for all sources
- Save comprehensive alignment report to `outputs/reports/spatial_alignment.txt`
- Include summary: X/5 tiles passed all checks, detailed breakdown per check type

### Deliverables
- [x] `scripts/data_validation/03_inspect_metadata.py` executed on all 53 REFIDs
- [x] `scripts/data_validation/04_check_spatial_alignment.py` created with comprehensive checks
- [x] `scripts/slurm/step3_spatial_alignment.sh` created and executed (Job ID: 23144257)
- [x] `scripts/data_validation/04_check_spatial_alignment.py` executed successfully
- [x] `outputs/reports/metadata_validation.csv` exists with 53 tiles and 5 sources
- [x] `outputs/reports/spatial_alignment.txt` exists with comprehensive validation results
- [x] Band count validation passed:
  - Sentinel-2: 126 bands ✓ (all 53 tiles)
  - PlanetScope: 42 bands ✓ (all 53 tiles)
  - VHR Google: 6 bands ✓ (all 53 tiles)
  - AlphaEarth: **448 bands** (documented, 7 years × 64 embeddings)
  - Masks: 1 band ✓ (all 53 tiles)
- [x] All 53 tiles use EPSG:4326
- [x] Comprehensive spatial alignment validated (5/5 sampled tiles):
  - 10m sources (Sentinel ↔ Mask, Sentinel ↔ AlphaEarth): exact alignment ✓
  - High-res sources (VHR ↔ PlanetScope): bounds coverage ✓ (max diff: 0.000031°)
  - Overall CRS consistency across all 5 sources ✓

### Actual Results
**Metadata validation completed for all 53 tiles:**
- **Band counts:** 53/53 correct across all 5 sources
  - Sentinel-2: 126 bands (7 years × 2 quarters × 9 bands)
  - PlanetScope: 42 bands (7 years × 2 quarters × 3 bands)
  - VHR: 6 bands (2 time points × 3 RGB bands)
  - AlphaEarth: 448 bands (fixed embedding size)
  - Masks: 1 band (binary)
- **CRS:** 53/53 tiles use EPSG:4326 across all sources
- **Dimensions:** Vary by tile and resolution:
  - 10m sources (Sentinel/AlphaEarth/Masks): ~65-108 × 66-67 pixels
  - PlanetScope (3-5m): ~179-336 × 178-336 pixels
  - VHR (1m): ~820-1603 × 655-662 pixels
- **Spatial alignment script:** Created with comprehensive validation across all source pairs

**Spatial alignment validation completed (5 tiles sampled):**
- **10m resolution sources:** Perfect alignment
  - Sentinel ↔ Mask: 5/5 aligned (exact CRS, bounds, transform, dimensions)
  - Sentinel ↔ AlphaEarth: 5/5 aligned (exact CRS, bounds, transform, dimensions)
- **High-resolution sources:** Geographic bounds aligned
  - VHR ↔ PlanetScope: 5/5 bounds coverage (max diff: 0.000031°, well within 0.001° tolerance)
- **Overall CRS:** 5/5 tiles have all sources using EPSG:4326

### Success Criteria
All tiles have correct band counts, all use EPSG:4326, comprehensive alignment validation passed. ✅ **All Step 3 criteria met!**

---

## STEP 4: Data Quality Checks

### Objective
Detect NoData values, validate value ranges, identify outliers across all data sources.

### Requirements

**4.1 - Comprehensive quality assessment:**

⚠️ **Resource Note:** Processing ALL 53 REFIDs × 5 sources requires ~2,650 band reads. Recommended to run via SLURM.

Create `scripts/data_validation/05_check_data_quality.py` that:
- For **ALL 53 REFIDs** (not just 10):

  **Sentinel-2 checks:**
  - Load first 9 bands (2018 Q2: bands 1-9) for start year validation
  - Load last 9 bands (2024 Q3: bands 118-126) for end year validation
  - For each set of 9 bands:
    - Count NaN/NoData pixels → compute percentage
    - For valid (non-NaN) pixels: compute min, max, mean, std
    - Expected range: 0-10,000 (reflectance × 10k)
    - Flag if: min < 0 or max > 15,000
  - Report: `s2_start_nodata_pct`, `s2_start_min/max/mean/std`, `s2_end_nodata_pct`, `s2_end_min/max/mean/std`

  **PlanetScope checks:**
  - Load first 3 bands (2018 Q2 RGB: bands 1-3)
  - Load last 3 bands (2024 Q3 RGB: bands 40-42)
  - For each set:
    - Count NaN/NoData pixels → compute percentage
    - Compute: min, max, mean, std
    - Expected range: 0-10,000 (same as Sentinel-2 reflectance scaling)
    - Flag if: min < 0 or max > 15,000
  - Report: `ps_start_nodata_pct`, `ps_start_min/max/mean/std`, `ps_end_nodata_pct`, `ps_end_min/max/mean/std`

  **VHR Google checks:**
  - Load first 3 bands (start year RGB: bands 1-3)
  - Load last 3 bands (end year RGB: bands 4-6)
  - For each set:
    - Count NaN/NoData pixels → compute percentage
    - Compute: min, max, mean, std
    - Expected range: 0-255 (typical RGB byte values) OR 0-10,000 (if scaled like others)
    - Document actual range found
  - Report: `vhr_start_nodata_pct`, `vhr_start_min/max/mean/std`, `vhr_end_nodata_pct`, `vhr_end_min/max/mean/std`

  **AlphaEarth checks (optional):**
  - Load first 64 bands (2018 embeddings)
  - Count NaN/NoData pixels → compute percentage
  - Compute: min, max, mean, std (embeddings may have negative values)
  - Document actual range (no expected range for embeddings)
  - Report: `ae_nodata_pct`, `ae_min/max/mean/std`

  **Mask checks:**
  - Load mask (single band)
  - Get unique values with `np.unique()`
  - Verify only {0, 1} present
  - Compute: change_pixels / total_pixels → change percentage
  - Flag if any values other than 0 or 1 found
  - Report: `mask_unique_values`, `mask_change_pct`

- Create comprehensive DataFrame with columns:
  - `refid`
  - **Sentinel-2:** `s2_start_nodata_pct`, `s2_start_min`, `s2_start_max`, `s2_start_mean`, `s2_start_std`, `s2_end_nodata_pct`, `s2_end_min`, `s2_end_max`, `s2_end_mean`, `s2_end_std`
  - **PlanetScope:** `ps_start_nodata_pct`, `ps_start_min`, `ps_start_max`, `ps_start_mean`, `ps_start_std`, `ps_end_nodata_pct`, `ps_end_min`, `ps_end_max`, `ps_end_mean`, `ps_end_std`
  - **VHR:** `vhr_start_nodata_pct`, `vhr_start_min`, `vhr_start_max`, `vhr_start_mean`, `vhr_start_std`, `vhr_end_nodata_pct`, `vhr_end_min`, `vhr_end_max`, `vhr_end_mean`, `vhr_end_std`
  - **AlphaEarth (optional):** `ae_nodata_pct`, `ae_min`, `ae_max`, `ae_mean`, `ae_std`
  - **Mask:** `mask_unique_values`, `mask_change_pct`
  - **Quality:** `quality_issues` (list of any problems found)

- Save to `outputs/reports/data_quality.csv`

- **Generate human-readable summary report:**
  - Create `outputs/reports/data_quality_summary.txt`
  - Include per-source statistics with clear sections:
    - Overall status (X/53 tiles passing)
    - Sentinel-2: NoData analysis, value ranges, quality status (✓ PASS/⚠ ISSUES)
    - PlanetScope: NoData analysis, value ranges, quality status
    - VHR: NoData analysis, brightness analysis, value ranges, quality status
    - AlphaEarth: NoData analysis, embedding statistics, quality status
    - Masks: Binary validation, change statistics (avg/median/range), change distribution breakdown
  - If issues exist: detailed list of problematic tiles with specific issues

- Print comprehensive summary to console:
  - **Per source statistics:**
    - Sentinel-2: avg NoData %, tiles with >5% NoData, value range issues
    - PlanetScope: avg NoData %, tiles with >5% NoData, value range issues
    - VHR: avg NoData %, tiles with >5% NoData, value range issues
    - AlphaEarth: avg NoData %, documented value ranges
    - Masks: tiles with invalid values (not {0,1}), avg change %, change distribution
  - **Overall:**
    - Total tiles processed: 53
    - Tiles with ANY quality issues
    - Tiles passing ALL checks

- Use tqdm progress bar for tile processing
- Include error handling for missing files

### Deliverables
- [x] `scripts/data_validation/05_check_data_quality.py` created/updated
- [x] `scripts/slurm/step4_quality_checks.sh` created (SLURM script)
- [x] Script executed via SLURM (Job ID: 23144669)
- [x] `outputs/reports/data_quality.csv` exists with all 53 tiles (39 columns)
- [x] `outputs/reports/data_quality_summary.txt` created (human-readable summary)
- [x] Quality issues identified and documented

### Actual Results
**All 53 tiles assessed across all 5 data sources - 53/53 passed all checks! ✅**

**Data Quality Summary:**
- **Sentinel-2 (10m):**
  - NoData: 0.31% (2018), 0.31% (2024)
  - Value range: [120-7964] (2018), [73-7396] (2024) - within expected 0-10,000 ✓
  - Status: ✓ PASS

- **PlanetScope (3-5m):**
  - NoData: 0.00% (both years)
  - Value range: [0-255] RGB ✓
  - Status: ✓ PASS

- **VHR Google (1m):**
  - NoData: 0.04% (start), 0.10% (end)
  - Brightness: 85.7/255 (start), 90.1/255 (end)
  - Status: ✓ PASS

- **AlphaEarth (10m embeddings):**
  - NoData: 0.00%
  - Embedding range: [-0.509, 0.466]
  - Status: ✓ PASS

- **Masks (10m binary):**
  - All masks contain only {0, 1} ✓
  - Average change: 14.30% (median: 10.56%)
  - Change range: [0.12%, 63.97%]
  - Distribution: 0 zero-change, 17 low (<5%), 30 moderate (5-30%), 6 high (≥30%)
  - Status: ✓ PASS

### Success Criteria
✅ **All criteria met!**
- All 53 tiles assessed across all 5 data sources
- NoData patterns documented (minimal across all sources)
- Value ranges validated (all within expected ranges)
- No problematic tiles identified - dataset ready for modeling

---

## STEP 5: Mask Analysis & Class Balance

### Objective
Understand label distribution and prepare for class imbalance handling.

### Requirements

**5.1 - Comprehensive mask analysis:**

Create `scripts/analysis/06_analyze_masks.py` that:
- For ALL 53 REFIDs:
  - Load mask
  - Compute:
    - total_pixels
    - change_pixels (mask == 1)
    - no_change_pixels (mask == 0)
    - change_ratio (as percentage)
  - Use `scipy.ndimage.label()` to find connected change regions
  - For connected components:
    - Count number of patches
    - **Collect all individual patch sizes** (for distribution analysis)
    - Find largest patch size
    - Compute mean patch size

- Create DataFrame with columns:
  - refid, total_pixels, change_pixels, change_ratio, num_change_patches, max_patch_size, mean_patch_size, patch_sizes

- Compute global statistics:
  - Overall change ratio (all tiles combined)
  - Median change ratio per tile
  - Count tiles with: 0% change, >50% change
  - Average patches per tile
  - Total patches across all tiles

- Save to `outputs/reports/mask_analysis.csv`

- Create **enhanced** visualization (`matplotlib` figure with 3 subplots):
  - **Subplot 1:** Histogram of change_ratio (30 bins)
  - **Subplot 2 (ENHANCED):** Histogram of **patch sizes in pixels** (50 bins)
    - Use logarithmic Y-axis for wide range visualization
    - Show median patch size with vertical red dashed line
    - Aggregates ALL patches from ALL tiles
    - Reveals spatial scale distribution (tiny to large patches)
  - **Subplot 3:** Boxplot of change_ratio
  - Add axis labels, titles, grid
  - Save to `outputs/figures/mask_statistics.png` at 150 DPI

**5.2 - Identify edge cases:**

Create `scripts/analysis/07_identify_edge_cases.py` that:
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
- [x] `scripts/analysis/06_analyze_masks.py` executed (enhanced with patch size distribution)
- [x] `scripts/analysis/07_identify_edge_cases.py` executed
- [x] `scripts/slurm/step5_mask_analysis.sh` created and executed (Job ID: 23144710)
- [x] `outputs/reports/mask_analysis.csv` exists (53 tiles)
- [x] `outputs/figures/mask_statistics.png` created (enhanced visualization)
- [x] Edge case files created (4 files)

### Actual Results
**All 53 tiles analyzed successfully! ✅**

**Class Balance Statistics:**
- **Overall change ratio:** 14.65% (all tiles combined)
- **Median change per tile:** 10.56%
- **Mean change per tile:** 14.30%
- **Range:** 0.12% to 63.97%
- **Imbalance ratio:** 1:6 (change:no-change)

**Spatial Characteristics:**
- **Total patches across all tiles:** 239 patches
- **Average patches per tile:** 4.5 patches/tile
- **Median patch size:** 24 pixels (~0.24 hectares at 10m resolution)
- **Mean patch size:** 435 pixels (~4.35 hectares)
- **Largest patch:** 4,779 pixels (~48 hectares)
- **Distribution:** Heavy-tailed (most patches tiny, few very large)

**Tile Categories:**
- **Zero change (0%):** 0 tiles
- **Low change (0-5%):** 17 tiles (32%)
- **Moderate change (5-30%):** 30 tiles (57%)
- **High change (≥30%):** 6 tiles (11%)
- **Very high change (>50%):** 2 tiles (4%)

**Key Findings:**
1. **Multi-scale challenge:** Patches range from 1 pixel to 4,779 pixels (4 orders of magnitude)
2. **Small patches dominate:** Median = 24 pixels, requires high-resolution feature detection
3. **Significant class imbalance:** Only 14.65% of pixels are land-take
4. **Good tile diversity:** 32% low, 57% moderate, 11% high change
5. **No zero-change tiles:** All tiles contain some land-take (good for training)

### Success Criteria
✅ **All criteria met!**
- Class balance fully understood (14.65% change, 1:6 imbalance ratio)
- Edge cases identified and categorized (4 categories, 53 tiles)
- Imbalance strategy clear: Use weighted loss functions (Focal Loss, Dice Loss), stratified splitting, evaluate with F1/IoU
- Spatial scale challenge documented: Model must detect patches from 24px (tiny) to 4779px (large)

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

Create `scripts/analysis/08_visualize_tiles.py` that:
- Defines `visualize_tile(data_root, refid, save_path)`:

  Creates 2×2 subplot figure (figsize=(14, 14)):

  **Layout:**
  - [0,0]: VHR 2018 RGB (bands 1, 2, 3)
    - Read with rasterio: `src.read([1, 2, 3])`
    - Normalize: divide by 255 (uint8 to 0-1)
    - Transpose to (H, W, C) for display
  - [0,1]: VHR 2025 RGB (bands 4, 5, 6)
    - Same normalization as 2018
  - [1,0]: Binary mask (land-take areas)
    - Resample mask from 10m to 1m resolution to match VHR
    - Colormap: 'RdYlGn_r' (red=change, green=no-change)
  - [1,1]: VHR 2025 with mask overlay
    - Show VHR 2025 RGB as base
    - Use `np.ma.masked_where(mask == 0, mask)`
    - Overlay with 'Reds' colormap at alpha=0.5

  - Add titles to each subplot
  - Set figure suptitle to REFID
  - Turn off axes
  - Save to save_path at 150 DPI
  - Use tight_layout()

- Enhanced REFID parsing:
  - Parse enhanced refid_list.txt format (with header and metadata table)
  - Skip header lines and separator lines
  - Extract REFIDs from metadata table (lines starting with 'a')

- In main:
  - Load refid_list.txt with proper parsing
  - Visualize first 3 REFIDs
  - Save to `outputs/figures/tile_viz_{refid}.png`
  - Print confirmation for each

**6.2 - Summary grid:**

Create `scripts/analysis/09_create_summary_grid.py` that:
- Loads edge case files from Step 5.2
- Loads mask_analysis.csv for change percentages
- Selects 6 diverse REFIDs:
  - 2 from zero_change (if available, otherwise low_change)
  - 2 from moderate_change
  - 2 from high_change
- For each REFID, creates compact visualization using VHR imagery:
  - 1×3 layout: [VHR 2018 RGB | VHR 2025 RGB | VHR 2025 + Mask overlay]
  - Uses VHR Google imagery (1m resolution) for high-quality visualization
  - Resamples mask from 10m to 1m to match VHR resolution
- Arranges all 6 into 6×3 grid (6 rows × 3 columns)
- Adds row labels:
  - REFID name (truncated if long)
  - Change category and percentage (from mask_analysis.csv)
- Saves to `outputs/figures/summary_grid.png` at 300 DPI
- Creates a one-page dataset overview showing diversity of land-take patterns

### Deliverables
- [x] `scripts/analysis/08_visualize_tiles.py` created and executed
- [x] `scripts/analysis/09_create_summary_grid.py` created and executed
- [x] `scripts/slurm/step6_visualizations.sh` created and executed (Job ID: 23144739)
- [x] 3 individual tile visualizations created with streamlined 2×2 layout
- [x] `outputs/figures/summary_grid.png` created (6 tiles × 3 views)

### Actual Results
**All visualizations successfully created! ✅**

**Individual Tile Visualizations (3 files):**
- Uses VHR Google imagery (1m resolution) for maximum clarity
- 2×2 layout: [VHR 2018] [VHR 2025] [Mask] [Overlay]
- File sizes: 4-5MB per tile
- Removed: RGB Change (Enhanced) and Grayscale Difference (not informative enough)

**Summary Grid (1 file, 27MB):**
- Shows 6 diverse tiles spanning low to high change levels
- Each tile shows: VHR 2018, VHR 2025, and mask overlay
- Provides one-page overview of dataset diversity

### Success Criteria
✅ **All criteria met!**
Visualizations show:
- Realistic RGB colors with excellent detail (1m resolution)
- Perfect mask alignment with visible changes
- No data loading errors
- Clear temporal changes between 2018-2025
- Streamlined layout focuses on most informative views

**User should manually review images before proceeding.**

---

## STEP 7: Generate Comprehensive Report

### Objective
Compile all findings into a final markdown report.

### Requirements

**7.1 - Compile data report:**

Create `scripts/analysis/10_generate_report.py` that:
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
- [ ] `scripts/analysis/10_generate_report.py` executed
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