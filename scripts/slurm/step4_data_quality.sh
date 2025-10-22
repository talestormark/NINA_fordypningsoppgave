#!/bin/sh
#SBATCH --account=share-ie-idi
#SBATCH --job-name=landtake_step4_quality
#SBATCH --time=0-01:00:00
#SBATCH --partition=CPUQ
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --output=outputs/slurm_outputs/%j_step4_quality_output.txt
#SBATCH --error=outputs/slurm_outputs/%j_step4_quality_error.err
#SBATCH --mail-user=tmstorma@stud.ntnu.no
#SBATCH --mail-type=ALL

# Step 4: Comprehensive Data Quality Check
# Checks ALL 53 REFIDs across ALL 5 data sources:
# - Sentinel-2 (start + end quarters)
# - PlanetScope (start + end quarters)
# - VHR Google (start + end years)
# - AlphaEarth (2018 embeddings)
# - Masks (binary labels)

echo "=========================================="
echo "Step 4: Data Quality Check"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "=========================================="
echo ""

# Load conda module
module purge
module load Anaconda3/2023.09-0

echo "Python version:"
python --version
echo ""

echo "Activating conda environment..."
source activate landtake_env
echo ""

# Print resource allocation
echo "Resources:"
echo "  Memory: 16GB"
echo "  Time limit: 1 hour"
echo ""

# Set working directory
WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of job: $SLURM_JOB_NAME"
echo "ID of job: $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo ""

# Run script
echo "Starting comprehensive quality check script..."
python scripts/data_validation/05_check_data_quality.py

# Check exit status
exit_code=$?
echo ""
echo "=========================================="
echo "Script exit code: $exit_code"
echo "End time: $(date)"
echo "=========================================="

exit $exit_code
