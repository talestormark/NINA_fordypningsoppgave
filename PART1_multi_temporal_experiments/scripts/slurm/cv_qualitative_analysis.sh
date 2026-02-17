#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=cv_qualitative
#SBATCH --time=0-01:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH --output=PART1_multi_temporal_experiments/outputs/logs/slurm_%x_%j.log
#SBATCH --error=PART1_multi_temporal_experiments/outputs/logs/slurm_%x_%j.err

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

# Run qualitative analysis for 3 hand-picked report tiles
python PART1_multi_temporal_experiments/scripts/analysis/qualitative_cv_analysis.py \
    --refids \
    "a4-63523450914143_51-752288292" \
    "a-3-82567804883019_40-09254102374404" \
    "a6-66442386131923_52-507271861"

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
