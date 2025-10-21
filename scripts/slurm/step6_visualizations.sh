#!/bin/sh
#SBATCH --account=share-ie-idi
#SBATCH --job-name=landtake_step6
#SBATCH --time=0-00:45:00         # 45 minutes for loading and visualizing rasters

#SBATCH --partition=CPUQ          # CPU only
#SBATCH --mem=48G                 # 48GB RAM for loading large rasters
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
pip install --user rasterio matplotlib > /dev/null 2>&1 || true
echo ""

echo "Running Step 6.1: Individual Tile Visualizations"
echo "========================================="
python /cluster/home/tmstorma/NINA_fordypningsoppgave/scripts/analysis/08_visualize_tiles.py

step1_exit=$?
echo "Step 6.1 exit code: $step1_exit"
echo ""

echo "Running Step 6.2: Summary Grid Creation"
echo "========================================="
python /cluster/home/tmstorma/NINA_fordypningsoppgave/scripts/analysis/09_create_summary_grid.py

step2_exit=$?
echo "Step 6.2 exit code: $step2_exit"
echo ""

# Overall exit code (fail if either failed)
if [ $step1_exit -ne 0 ] || [ $step2_exit -ne 0 ]; then
    exit_code=1
else
    exit_code=0
fi

echo ""
echo "========================================="
echo "Job completed at: $(date)"
echo "Overall exit code: $exit_code"
echo "========================================="

exit $exit_code
