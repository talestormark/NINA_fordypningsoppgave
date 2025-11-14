# Environment Quick Start

## ✅ Environment Successfully Installed!

The modeling environment has been set up using `pip install --user`.

## Usage

### Load Environment (Required for Every Session)

```bash
module load Python/3.11.3-GCCcore-12.3.0
```

### Verify Installation

```bash
python3 scripts/utils/test_imports.py
```

Expected output: All 14 packages should show "OK"

## Installed Packages

### Core Deep Learning
- **PyTorch 2.2.0** (CPU version)
- **TorchVision 0.17.0**
- **TorchGeo 0.7.2** - Geospatial ML library
- **Segmentation Models PyTorch 0.5.0** - U-Net implementations
- **Albumentations 2.0.8** - Image augmentation

### Geospatial
- **Rasterio 1.4.3** - GeoTIFF I/O
- **GeoPandas 1.1.1** - Geospatial data frames
- **Shapely 2.1.2** - Geometric operations
- **PyProj 3.7.2** - Coordinate transformations
- **Fiona 1.10.1** - Vector data I/O

### Scientific Computing
- **NumPy 1.26.4** (pinned to 1.x for PyTorch compatibility)
- **Pandas 2.3.3**
- **SciPy 1.16.3**
- **Scikit-learn 1.7.2**
- **Matplotlib 3.10.7**
- **Seaborn 0.13.2**

### Utilities
- **wandb 0.23.0** - Experiment tracking
- **tqdm 4.67.1** - Progress bars

## Using in SLURM Jobs

Add to your SLURM script:

```bash
#!/bin/bash
#SBATCH --job-name=landtake_training
#SBATCH --account=share-ie-idi
#SBATCH --time=02:00:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=4

# Load environment
module load Python/3.11.3-GCCcore-12.3.0

# Run your script
python3 scripts/modeling/train.py
```

## GPU Support

The current installation is **CPU-only**. For GPU support:

```bash
# Load CUDA module
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.0.0

# Reinstall PyTorch with CUDA support
python3 -m pip install --user torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
```

## Troubleshooting

### Import Errors

If you get import errors:
1. Verify module is loaded: `module list`
2. Check installed packages: `python3 -m pip list --user | grep <package>`
3. Re-run test script: `python3 scripts/utils/test_imports.py`

### NumPy Version Issues

If you see NumPy 2.x errors:
```bash
python3 -m pip install --user "numpy<2.0" --force-reinstall
```

### Dill Compatibility Issues

If you see dill errors:
```bash
python3 -m pip install --user "dill<0.3.9" --force-reinstall
```

## Next Steps

Now that your environment is ready, you can:

1. **Test dataset loading**:
   ```bash
   python3 scripts/modeling/test_dataset.py
   ```

2. **Test model architectures**:
   ```bash
   python3 scripts/modeling/test_models.py
   ```

3. **Start baseline training** (see docs/BASELINE_MODELS.md):
   ```bash
   sbatch scripts/slurm/train_unet_baseline.sh
   ```

## Installation Location

All packages are installed in:
```
~/.local/lib/python3.11/site-packages
```

This persists across login sessions.

## Reinstalling from Scratch

If you need to start fresh:

```bash
# Remove all user packages
rm -rf ~/.local/lib/python3.11/site-packages

# Reinstall
bash scripts/setup_environment_pipuser.sh
```
