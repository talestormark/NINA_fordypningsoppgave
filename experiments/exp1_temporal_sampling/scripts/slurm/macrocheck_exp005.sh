#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --time=0-06:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=4
#SBATCH --output=experiments/exp1_temporal_sampling/outputs_v3/logs/slurm_%x_%j.log
#SBATCH --error=experiments/exp1_temporal_sampling/outputs_v3/logs/slurm_%x_%j.err
#
# Macro-vs-micro checkpoint-metric check.
# Re-runs exp005 (EarlyFusion T=2, full-window) identically to
# train_all_experiments_v3.sh, but with per-epoch validation MACRO IoU now also
# logged by train_multitemporal.py validate() as val['iou_macro'] (selection
# still on micro 'iou'). Writes to a FRESH output dir so the original
# exp005_fold* runs are untouched. WandB disabled (history.json is read locally).
#
# Usage: sbatch --job-name=macrochk_f0 experiments/exp1_temporal_sampling/scripts/slurm/macrocheck_exp005.sh 0

FOLD=${1:-"0"}

cd /cluster/home/tmstorma/NINA_fordypningsoppgave
mkdir -p experiments/exp1_temporal_sampling/outputs_v3/logs

module --quiet purge
module load Anaconda3/2024.02-1
source activate masterthesis

echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

OUTPUT_DIR="experiments/exp1_temporal_sampling/outputs_v3/macrocheck_exp005_fold${FOLD}"
echo "Output directory: $OUTPUT_DIR  (fold $FOLD)"
echo "Job started at: $(date)"

python experiments/exp1_temporal_sampling/scripts/modeling/train_multitemporal.py \
    --model-name early_fusion_unet \
    --temporal-sampling bi_temporal \
    --encoder-name resnet50 \
    --encoder-weights imagenet \
    --batch-size 8 \
    --accumulation-steps 1 \
    --image-size 64 \
    --num-workers 4 \
    --epochs 400 \
    --lr 0.01 \
    --optimizer adamw \
    --scheduler cosine \
    --weight-decay 5e-4 \
    --loss focal_dice \
    --focal-alpha 0.75 \
    --focal-gamma 2.0 \
    --lambda-focal 1.0 \
    --lambda-dice 1.0 \
    --output-dir $OUTPUT_DIR \
    --seed 42 \
    --fold $FOLD \
    --num-folds 5 \
    --data-dir data_v2 \
    --mask-subdir Land_take_masks_coarse \
    --splits-dir preprocessing/outputs/splits/unified_fullwindow \
    --change-level-path preprocessing/outputs/splits/unified_fullwindow/split_info.csv

echo ""
echo "Done fold $FOLD at $(date)"
