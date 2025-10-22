#!/bin/sh
#SBATCH --account=share-ie-idi
#SBATCH --job-name=landtake_step3_alignment
#SBATCH --time=0-00:15:00         # 15 minutes should be enough for 5 tiles
#SBATCH --partition=CPUQ          # CPU only (no GPU needed)
#SBATCH --mem=8G                  # 8GB RAM should be sufficient
#SBATCH --nodes=1
#SBATCH --output=outputs/slurm_outputs/%j_step3_alignment_output.txt
#SBATCH --error=outputs/slurm_outputs/%j_step3_alignment_error.err
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

echo "Activating conda environment..."
source activate landtake_env
echo ""

echo "Running comprehensive spatial alignment validation..."
python scripts/data_validation/04_check_spatial_alignment.py

echo ""
echo "========================================="
echo "Job completed!"
