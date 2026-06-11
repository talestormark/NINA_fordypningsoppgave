#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80g
#SBATCH --time=0-04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=experiments/exp1_temporal_sampling/outputs_v3/logs/%x_%j.log

module --quiet purge
module load Anaconda3/2024.02-1
conda activate masterthesis

cd /cluster/home/tmstorma/NINA_fordypningsoppgave

EXPS="annual,bi_temporal,bi_seasonal,early_fusion,late_fusion,late_fusion_pool,conv3d_fusion,lstm_lite,lstm7_lite"

echo "=== Starting $SLURM_JOB_NAME ==="
echo "Experiments: $EXPS"
echo ""

if [ "$SLURM_JOB_NAME" = "test_eval_v3" ]; then
    python experiments/exp1_temporal_sampling/scripts/modeling/evaluate_test_final.py --experiments $EXPS
elif [ "$SLURM_JOB_NAME" = "persample_v3" ]; then
    python experiments/exp1_temporal_sampling/scripts/modeling/statistical_analysis_persample.py --experiments $EXPS
fi

echo "=== Done ==="
