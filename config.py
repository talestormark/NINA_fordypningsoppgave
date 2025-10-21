"""
Configuration file for Land-Take Detection project
"""

# ============================================================================
# DATA PATHS
# ============================================================================

# IMPORTANT: User must update this path to point to the actual data location
DATA_ROOT = "data/raw"  # PLACEHOLDER - UPDATE THIS!

# Example:
# DATA_ROOT = "/cluster/home/tmstorma/data/landtake"
# or
# DATA_ROOT = "/mnt/storage/HABLOSS_data"

# ============================================================================
# FOLDER NAMES (matching the dataset structure)
# ============================================================================

FOLDERS = {
    'sentinel': 'Sentinel',
    'planetscope': 'PlanetScope',
    'vhr': 'VHR_google',
    'alphaearth': 'AlphaEarth',
    'masks': 'Land_take_masks'
}

# ============================================================================
# FILE NAMING PATTERNS
# ============================================================================

FILE_PATTERNS = {
    'sentinel': '{refid}_RGBNIRRSWIRQ_Mosaic.tif',
    'planetscope': '{refid}_RGBQ_Mosaic.tif',
    'vhr': '{refid}_RGBY_Mosaic.tif',
    'alphaearth': '{refid}_VEY_Mosaic.tif',
    'mask': '{refid}_mask.tif'
}

# ============================================================================
# EXPECTED METADATA
# ============================================================================

EXPECTED_BANDS = {
    'sentinel': 126,      # 7 years × 2 quarters × 9 bands
    'planetscope': 42,    # 7 years × 2 quarters × 3 bands
    'vhr': 6,             # 2 time points × 3 RGB bands
    'alphaearth': None,   # Varies - will be documented
    'mask': 1             # Binary label
}

EXPECTED_CRS = 'EPSG:4326'  # WGS84

# ============================================================================
# OUTPUT PATHS
# ============================================================================

REPORTS_DIR = 'outputs/reports'
FIGURES_DIR = 'outputs/figures'
