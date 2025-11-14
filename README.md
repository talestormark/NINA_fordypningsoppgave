# NINA Land-Take Detection Project

Deep learning-based land-take detection using multi-temporal remote sensing data from the HABLOSS project.

## Project Overview

This project focuses on detecting and analyzing land-take (conversion of natural/agricultural land to artificial surfaces) across 55 diverse European locations using high-resolution satellite imagery and change detection masks.

**Primary Data Source**: VHR (Very High Resolution) Google imagery at 1m resolution
- 2018 RGB baseline
- 2025 RGB current state
- Binary land-take change masks

**Secondary Data Sources** (for future multi-temporal modeling):
- Sentinel-2 multi-temporal imagery (7 years, 126 bands)
- AlphaEarth embeddings (448 bands)
- PlanetScope imagery

## Whats done so far?

Take a look at the [Project Summary](docs/PROJECT_SUMMARY.md) to understand whats done so far. 

## Repository Structure

```
NINA_fordypningsoppgave/
├── README.md                          # This file
├── config.py                          # Central configuration (data paths, constants)
├── land_take_bboxes_650m_v1.geojson  # Geographic boundaries for tiles
│
├── docs/                              # Project documentation
│   ├── INSTRUCTIONS.md               # Step-by-step workflow guide
│   ├── DATASETS.md                   # Data source descriptions
│   └── SLURM_INSTRUCTIONS.md         # HPC/SLURM usage guide
│
├── scripts/
│   ├── slurm/                        # SLURM batch scripts for HPC
│   │   ├── step4_quality_checks.sh
│   │   ├── step5_mask_analysis.sh
│   │   └── step6_visualizations.sh
│   │
│   ├── data_validation/              # Steps 1-4: Data understanding
│   │   ├── 01_inspect_filesystem.py
│   │   ├── 02_extract_refids.py
│   │   ├── 03_inspect_metadata.py
│   │   ├── 04_check_spatial_alignment.py
│   │   └── 05_check_data_quality.py
│   │
│   ├── analysis/                     # Steps 5-6: Mask analysis & visualization
│   │   ├── 06_analyze_masks.py
│   │   ├── 07_identify_edge_cases.py
│   │   ├── 08_visualize_tiles.py
│   │   └── 09_create_summary_grid.py
│   │
│   └── utils/                        # Shared utilities
│       └── test_imports.py
│
├── environment/                       # Environment setup
│   ├── landtake_env.yml              # Conda environment specification
│   └── setup_env.sh                  # Environment setup script
│
├── data/
│   ├── raw/                          # Original datasets (gitignored)
│   │   ├── Sentinel-2/
│   │   ├── VHR-google/
│   │   ├── AlphaEarth/
│   │   ├── PlanetScope/
│   │   └── masks/
│   └── processed/                    # Processed outputs (gitignored)
│
└── outputs/
    ├── reports/                      # CSV/TXT analysis results
    ├── figures/                      # PNG/PDF visualizations
    └── slurm_outputs/                # SLURM job logs
```

## Quick Start

### 1. Environment Setup

The modeling environment is now installed! To use it:

```bash
# Load Python module
module load Python/3.11.3-GCCcore-12.3.0

# Verify installation
python3 scripts/utils/test_imports.py
```

For detailed setup info, see `docs/ENVIRONMENT_QUICK_START.md`

### 2. Configuration

Edit `config.py` to set your data paths:

```python
DATA_ROOT = "data/raw"  # Adjust to your data location
```

### 3. Run Data Validation

For local testing (small scripts):
```bash
python scripts/data_validation/01_inspect_filesystem.py
python scripts/data_validation/02_extract_refids.py
```

For compute-intensive tasks (use SLURM):
```bash
# From project root directory
sbatch scripts/slurm/step4_quality_checks.sh
sbatch scripts/slurm/step5_mask_analysis.sh
sbatch scripts/slurm/step6_visualizations.sh
```

## Dataset Statistics

- **Total tiles**: 53 complete tiles (55 in original dataset)
- **Tile size**: ~650m × 650m geographic extent
- **VHR resolution**: 1m (650-1600 × 654-662 pixels)
- **Sentinel-2 resolution**: 10m
- **Mask resolution**: 10m
- **Coordinate system**: EPSG:4326 (WGS84)

### Data Quality Summary

**VHR Google Imagery**:
- NoData: 0.02% (excellent coverage)
- Brightness (2018): 90.0/255
- Brightness (2025): 95.6/255
- No quality issues detected

**Land-Take Change**:
- Average change: 14.65% of pixels per tile
- Zero-change tiles: 0
- Low-change (<5%): 17 tiles
- Moderate-change (5-30%): 30 tiles
- High-change (≥30%): 6 tiles

## Key Technical Details

### VHR Google Imagery
- **Bands**: 6 (2018_R, 2018_G, 2018_B, 2025_R, 2025_G, 2025_B)
- **Dtype**: uint8 (0-255)
- **Resolution**: 1m
- **File pattern**: `{refid}_RGBY_Mosaic.tif`

### Sentinel-2 Multi-temporal
- **Bands**: 126 (7 years × 2 quarters × 9 spectral bands)
- **Dtype**: int16 (reflectance × 10000)
- **Resolution**: 10m
- **File pattern**: `{refid}_VEY_Mosaic.tif`

### Land-Take Masks
- **Bands**: 1 (binary: 0 = no change, 1 = land-take)
- **Dtype**: uint8
- **Resolution**: 10m
- **File pattern**: `{refid}_mask.tif`

## HPC/SLURM Usage

**Account**: `share-ie-idi`

**Resource Requirements**:
- Step 4 (Quality Checks): 16GB RAM, 30 min
- Step 5 (Mask Analysis): 32GB RAM, 1 hour
- Step 6 (Visualizations): 48GB RAM, 45 min

**Monitoring Jobs**:
```bash
squeue -u $USER                    # Check job status
cat outputs/slurm_outputs/*.txt    # View output logs
```

See `docs/SLURM_INSTRUCTIONS.md` for detailed HPC usage.

## Outputs

### Reports (`outputs/reports/`)
- `data_quality.csv` - Quality metrics for all tiles
- `mask_analysis.csv` - Land-take statistics
- `refids_*.txt` - Categorized tile lists by change level

### Figures (`outputs/figures/`)
- `tile_{refid}.png` - Individual tile visualizations (2018, 2025, change, mask)
- `summary_grid.png` - 6-tile overview showing diverse change levels

## Experiment Tracking

**Weights & Biases Integration**: All training runs are tracked with wandb for visualization and comparison.

- **Project Dashboard**: https://wandb.ai/NINA_Fordypningsoppgave/landtake-detection
- **Enable logging**: Add `--wandb` flag to training commands
- **View past runs**: 8 baseline models already uploaded with full metrics
- **Features**: Training curves, model comparison, checkpoint management

```bash
# Train with wandb logging
sbatch scripts/slurm/train_baseline.sh siam_conc 200 4 resnet50 42 true
```

## Baseline models

Read the file [Baseline Models](docs/BASELINE_MODELS.md) and its linked references.

--- 

## Code Style Guidelines
- Keep the code **minimal** — only include what is **absolutely necessary**.  
- The code must be **clean, readable, and well-structured**.  
- Use **clear variable names** and **concise comments**.  
- Do **not** use emoticons
- **Do not** generate or modify any code **without my explicit approval**.

---

## Contact

**Author**: tmstorma@stud.ntnu.no
**Project**: NINA Fordypningsoppgave (Deep Learning for Land-Take Detection)
**Data Source**: HABLOSS Project (https://habloss.eu/)

## License

Research project - contact author for data access and usage permissions.
