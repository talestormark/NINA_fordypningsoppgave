"""
Test script to verify all required libraries are installed
"""

import sys

def test_imports():
    """Test importing all critical libraries"""

    print("Testing imports for Land-Take Detection project...")
    print("=" * 60)

    failed_imports = []

    # Test each import
    imports_to_test = [
        ('rasterio', 'Rasterio (geospatial raster I/O)'),
        ('numpy', 'NumPy (numerical computing)'),
        ('pandas', 'Pandas (data analysis)'),
        ('geopandas', 'GeoPandas (geospatial data)'),
        ('matplotlib', 'Matplotlib (plotting)'),
        ('seaborn', 'Seaborn (statistical visualization)'),
        ('sklearn', 'Scikit-learn (machine learning)'),
        ('tqdm', 'TQDM (progress bars)'),
        ('scipy', 'SciPy (scientific computing)'),
        ('torch', 'PyTorch (deep learning)'),
        ('torchvision', 'TorchVision (computer vision)'),
        ('torchgeo', 'TorchGeo (geospatial ML)'),
        ('albumentations', 'Albumentations (image augmentation)'),
        ('segmentation_models_pytorch', 'SMP (segmentation models)'),
    ]

    for module_name, description in imports_to_test:
        try:
            __import__(module_name)
            print(f"✓ {description:50s} OK")
        except ImportError as e:
            print(f"✗ {description:50s} FAILED")
            failed_imports.append((module_name, str(e)))

    print("=" * 60)

    if failed_imports:
        print(f"\n❌ {len(failed_imports)} import(s) failed:")
        for module, error in failed_imports:
            print(f"   - {module}: {error}")
        print("\nPlease install missing packages with:")
        print("   conda env create -f landtake_env.yml")
        print("   conda activate landtake")
        return False
    else:
        print("\n✅ All imports successful!")
        print("\nEnvironment ready for land-take detection analysis.")
        return True

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
