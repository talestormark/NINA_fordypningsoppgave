#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80g
#SBATCH --time=0-04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=PART1_multi_temporal_experiments/outputs_v2/logs/%x_%j.log

module --quiet purge
module load Anaconda3/2024.02-1
conda activate masterthesis

cd /cluster/home/tmstorma/NINA_fordypningsoppgave

EXPS="annual,bi_temporal,bi_seasonal,early_fusion,late_fusion,late_fusion_pool,conv3d_fusion,lstm_lite,lstm7_lite"

echo "=== Starting $SLURM_JOB_NAME ==="
echo "Experiments: $EXPS"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo ""

if [ "$SLURM_JOB_NAME" = "test_eval" ]; then
    python PART1_multi_temporal_experiments/scripts/modeling/evaluate_test_final.py --experiments $EXPS
elif [ "$SLURM_JOB_NAME" = "persample_stats" ]; then
    python PART1_multi_temporal_experiments/scripts/modeling/statistical_analysis_persample.py --experiments $EXPS
fi

echo ""
echo "=== Done ==="
