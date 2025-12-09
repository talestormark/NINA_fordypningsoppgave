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

## Repository Structure

```
NINA_fordypningsoppgave/
├── README.md                          # This file
├── .gitignore                         # Git ignore rules
├── config.py                          # Central configuration (data paths, constants)
├── land_take_bboxes_650m_v1.geojson  # Geographic boundaries for tiles
│
└── scripts/
    ├── slurm/                         # SLURM batch scripts for HPC
    ├── data_validation/               # Data understanding and quality checks
    ├── analysis/                      # Mask analysis and visualizations
    ├── modeling/                      # Model training and evaluation
    │   ├── train.py                   # Training script
    │   ├── evaluate.py                # Evaluation script
    │   ├── models.py                  # Model architectures
    │   └── dataset.py                 # Dataset loaders
    ├── visualization/                 # Map generation and plotting
    └── utils/                         # Shared utilities
```

**Note**: `data/`, `outputs/`, `docs/`, and other large files are gitignored.


## Contact

**Author**: tmstorma@stud.ntnu.no
**Project**: NINA Fordypningsoppgave (Deep Learning for Land-Take Detection)
