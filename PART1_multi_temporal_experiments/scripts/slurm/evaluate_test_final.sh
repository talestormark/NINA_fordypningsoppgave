#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=test_eval
#SBATCH --time=0-00:30:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH --output=PART1_multi_temporal_experiments/outputs/logs/slurm_%x_%j.log
#SBATCH --error=PART1_multi_temporal_experiments/outputs/logs/slurm_%x_%j.err

# ============================================================================
# FINAL TEST SET EVALUATION
# ============================================================================
#
# This script runs the final, locked evaluation on the held-out test set.
#
# PROTOCOL (LOCKED - DO NOT MODIFY):
# - Ensemble: CV ensemble (5 fold models, mean probability)
# - Checkpoint: Best validation IoU
# - Threshold: 0.5 (fixed)
# - Normalization: Per-fold training stats
#
# ============================================================================

echo "=========================================="
echo "FINAL TEST SET EVALUATION"
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

# Run evaluation for all conditions
python PART1_multi_temporal_experiments/scripts/modeling/evaluate_test_final.py --all-conditions

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
