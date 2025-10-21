#!/bin/sh
#SBATCH --account=share-ie-idi
#SBATCH --job-name=landtake_step5
#SBATCH --time=0-01:00:00         # 1 hour for processing all 53 masks

#SBATCH --partition=CPUQ          # CPU only
#SBATCH --mem=32G                 # 32GB RAM for all masks + visualization
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
module load Anaconda3/2023.09-0

echo "Python version:"
python --version
echo ""

echo "Installing/checking required packages..."
pip install --user rasterio geopandas scipy matplotlib > /dev/null 2>&1 || true
echo ""

echo "Running Step 5.1: Mask Analysis (ALL 53 REFIDs)"
echo "========================================="
python /cluster/home/tmstorma/NINA_fordypningsoppgave/scripts/analysis/06_analyze_masks.py

exit_code=$?

echo ""
echo "========================================="
echo "Job completed at: $(date)"
echo "Exit code: $exit_code"
echo "========================================="

exit $exit_code
