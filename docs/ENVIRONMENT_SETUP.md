# Environment Setup Guide

## Installation (One-Time Setup)

Run the installation script:

```bash
bash scripts/setup_modeling_environment.sh
```

This installs all required packages using `pip install --user` (no conda/mamba required).

Installation takes approximately 10-15 minutes.

## Using the Environment

### Load Required Modules

**For CPU-only:**
```bash
module load Python/3.11.3-GCCcore-12.3.0
```

**For GPU:**
```bash
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.0.0
```

### Verify Installation

```bash
python3 scripts/utils/test_imports.py
```

### Check GPU Availability (if using GPU version)

```bash
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

## Troubleshooting

### Package Installation Fails

If a specific package fails to install:
```bash
# Try installing it individually
python3 -m pip install --user <package-name>

# Check error messages
python3 -m pip install --user <package-name> --verbose
```

### Import Errors

If imports fail after installation:
```bash
# Make sure you've loaded the Python module
module load Python/3.11.3-GCCcore-12.3.0

# Check installed packages
python3 -m pip list --user | grep <package-name>

# Verify Python path
python3 -c "import sys; print('\n'.join(sys.path))"
```

### CUDA/GPU Issues

```bash
# Check GPU availability
nvidia-smi

# Verify CUDA module is loaded
module list

# Test PyTorch CUDA
python3 -c "import torch; print(torch.cuda.is_available())"
```

## Package List

### Core Scientific
- numpy, pandas, scipy, scikit-learn
- matplotlib, seaborn
- tqdm

### Geospatial
- rasterio, geopandas, shapely, pyproj, fiona

### Deep Learning
- torch==2.2.0
- torchvision==0.17.0
- torchgeo
- segmentation-models-pytorch
- albumentations

### Experiment Tracking
- wandb

## Alternative: Minimal Installation

If you only need specific packages for testing:
```bash
module load Python/3.11.3-GCCcore-12.3.0
python3 -m pip install --user torch torchvision rasterio geopandas segmentation-models-pytorch
```

## Notes

- All packages are installed to `~/.local/lib/python3.11/site-packages`
- No conda/mamba required
- Compatible with SLURM batch jobs (just add `module load` commands to job script)
- Installation takes approximately 10-15 minutes depending on network speed
