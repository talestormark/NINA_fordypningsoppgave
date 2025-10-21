#!/bin/bash
# Quick environment setup using HPC modules instead of conda

# Load GDAL module (includes rasterio bindings)
module purge
module load GDAL/3.9.0-foss-2023b

# Install missing Python packages via pip (much faster than conda)
pip install --user rasterio geopandas

echo "Environment setup complete!"
echo "To use this environment, run: source setup_env.sh"
