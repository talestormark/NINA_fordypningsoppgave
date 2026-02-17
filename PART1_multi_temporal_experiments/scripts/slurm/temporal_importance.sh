#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=temporal_importance
#SBATCH --time=0-01:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH --output=PART1_multi_temporal_experiments/outputs/logs/slurm_%x_%j.log
#SBATCH --error=PART1_multi_temporal_experiments/outputs/logs/slurm_%x_%j.err

# ============================================================================
# TEMPORAL CONTRIBUTION ANALYSIS
# ============================================================================
#
# Runs three experiments to explain WHY LSTM-7 outperforms:
#   Exp 1: Temporal gradient attribution (GPU, ~30 min)
#   Exp 2: Input temporal autocorrelation (CPU, ~10 min)
#   Exp 3: Training dynamics comparison (CPU, ~1 min)
#
# ============================================================================

echo "=========================================="
echo "TEMPORAL CONTRIBUTION ANALYSIS"
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

# Run all three experiments (Exp 1 needs GPU)
python PART1_multi_temporal_experiments/scripts/analysis/temporal_importance_analysis.py

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
