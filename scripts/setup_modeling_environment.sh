#!/bin/bash
#
# Install modeling environment using pip --user
# Run: bash scripts/setup_environment_pipuser.sh
#

set -e  # Exit on error

echo "=== NINA Land-Take Modeling Environment Setup ==="
echo "Using pip install --user approach"
echo ""

# Load Python module
echo "[1/4] Loading Python module..."
module load Python/3.11.3-GCCcore-12.3.0
python3 --version
python3 -m pip --version
echo ""

# Upgrade pip
echo "[2/4] Upgrading pip..."
python3 -m pip install --user --upgrade pip
echo ""

# Install PyTorch and torchvision with CUDA 11.8 support
echo "[3/4] Installing PyTorch ecosystem with CUDA support..."
python3 -m pip install --user torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu118
echo ""

# Install remaining packages
echo "[4/5] Installing remaining packages..."
python3 -m pip install --user \
    "numpy<2.0" \
    pandas \
    scipy \
    scikit-learn \
    matplotlib \
    seaborn \
    tqdm \
    rasterio \
    geopandas \
    shapely \
    pyproj \
    fiona \
    albumentations \
    segmentation-models-pytorch \
    wandb \
    torchgeo

echo ""
echo "[5/5] Fixing compatibility issues..."
python3 -m pip install --user "dill<0.3.9" --force-reinstall

echo ""
echo "=== Installation complete! ==="
echo ""
echo "To use this environment in future sessions:"
echo "  module load Python/3.11.3-GCCcore-12.3.0"
echo ""
echo "To verify installation:"
echo "  python3 scripts/utils/test_imports.py"
