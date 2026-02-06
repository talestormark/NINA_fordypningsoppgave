#!/bin/bash
#SBATCH --job-name=stratified_analysis
#SBATCH --output=slurm_outputs/stratified_analysis_%j.log
#SBATCH --error=slurm_outputs/stratified_analysis_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --account=ie-idi

echo "========================================"
echo "Stratified Analysis by Land-Take Type"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo ""

# Load modules
module purge
module load Anaconda3/2024.02-1

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate masterthesis

# Navigate to project directory
cd /cluster/home/tmstorma/NINA_fordypningsoppgave/multi_temporal_experiments

# Create output directory for SLURM logs if needed
mkdir -p slurm_outputs

# Print environment info
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# Run the stratified analysis
echo "Running stratified analysis..."
python scripts/analysis/stratified_by_change_type.py

echo ""
echo "========================================"
echo "Completed: $(date)"
echo "========================================"
