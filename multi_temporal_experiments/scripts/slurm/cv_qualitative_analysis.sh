#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=cv_qualitative
#SBATCH --time=0-01:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH --output=multi_temporal_experiments/outputs/logs/slurm_%x_%j.log
#SBATCH --error=multi_temporal_experiments/outputs/logs/slurm_%x_%j.err

# ============================================================================
# CV QUALITATIVE ANALYSIS (Sanity Check Before Test Evaluation)
# ============================================================================
#
# Generates systematic visualizations for out-of-fold predictions:
# - Good cases (IoU >= 75th percentile)
# - Hard cases (40-60th percentile)
# - Failure cases (IoU <= 25th percentile)
#
# Each visualization includes:
# - RGB composite (first vs last year)
# - Valid data mask
# - Ground truth mask
# - Probability map
# - Thresholded prediction (t=0.5)
# - TP/FP/FN overlay
#
# Run this BEFORE test evaluation to verify:
# - Correct spatial alignment
# - Expected failure modes
# - Consistency with reported metrics
#
# ============================================================================

echo "=========================================="
echo "CV QUALITATIVE ANALYSIS"
echo "=========================================="
echo "Job started at: $(date)"
echo "Host: $(hostname)"
echo "=========================================="

# Navigate to project root
cd /cluster/home/tmstorma/NINA_fordypningsoppgave

# Load environment
module --quiet purge
module load Anaconda3/2024.02-1
source activate masterthesis

# Check GPU
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Run qualitative analysis
# Generates 3 examples per category (good/hard/failure)
# Same tiles are shown across all modes (T=2, T=7, T=14) for apples-to-apples comparison
python multi_temporal_experiments/scripts/analysis/qualitative_cv_analysis.py \
    --num-examples 3

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
