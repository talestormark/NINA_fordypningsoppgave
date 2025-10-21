#!/bin/sh
#SBATCH --account=share-ie-idi
#SBATCH --job-name=landtake_step4
#SBATCH --time=0-00:30:00         # 30 minutes should be enough

#SBATCH --partition=CPUQ          # CPU only (no GPU needed for this step)
#SBATCH --mem=16G                 # 16GB RAM for loading raster data
#SBATCH --nodes=1
#SBATCH --output=outputs/slurm_outputs/%j_output.txt
#SBATCH --error=outputs/slurm_outputs/%j_error.err

#SBATCH --mail-user=tmstorma@stud.ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of job: $SLURM_JOB_NAME"
echo "ID of job: $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "========================================="
echo ""

module purge
# Use the same Anaconda we used for local testing
module load Anaconda3/2023.09-0

echo "Python version:"
python --version
echo ""

echo "Installing/checking required packages..."
pip install --user rasterio geopandas > /dev/null 2>&1 || true
echo ""

echo "Running Step 4: Data Quality Checks"
echo "========================================="
python /cluster/home/tmstorma/NINA_fordypningsoppgave/scripts/data_validation/05_check_data_quality.py

exit_code=$?

echo ""
echo "========================================="
echo "Job completed at: $(date)"
echo "Exit code: $exit_code"
echo "========================================="

exit $exit_code
