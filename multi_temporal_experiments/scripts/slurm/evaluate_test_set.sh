#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=test_eval
#SBATCH --time=0-00:30:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH --output=multi_temporal_experiments/outputs/logs/slurm_%x_%j.log
#SBATCH --error=multi_temporal_experiments/outputs/logs/slurm_%x_%j.err

# ============================================================================
# Test Set Evaluation Script
# ============================================================================
#
# Evaluates trained models on the held-out test set (8 samples).
#
# Usage:
#   # Evaluate single experiment (all folds + ensemble)
#   sbatch evaluate_test_set.sh exp001_v2
#
#   # Evaluate all v2 experiments
#   for exp in exp001_v2 exp002_v2 exp003_v2; do
#     sbatch evaluate_test_set.sh $exp
#   done
#
# ============================================================================

EXPERIMENT=${1:-"exp001_v2"}

echo "=========================================="
echo "TEST SET EVALUATION"
echo "=========================================="
echo "Experiment: $EXPERIMENT"
echo "Job started at: $(date)"
echo "Host: $(hostname)"
echo "=========================================="
echo ""

# Navigate to project root
cd /cluster/home/tmstorma/NINA_fordypningsoppgave

# Load environment
module --quiet purge
module load Anaconda3/2024.02-1
source activate masterthesis

# Check GPU
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Run evaluation (all folds + ensemble)
python multi_temporal_experiments/scripts/modeling/evaluate_test_set.py \
    --experiment $EXPERIMENT \
    --all-folds \
    --num-folds 5 \
    --batch-size 1

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
