#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=pilot_imgsize
#SBATCH --time=0-02:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=4
#SBATCH --output=multi_temporal_experiments/outputs/logs/slurm_%x_%j.log
#SBATCH --error=multi_temporal_experiments/outputs/logs/slurm_%x_%j.err

# Pilot experiment for image size testing

IMG_SIZE=${1:-"64"}
FOLD=${2:-"3"}
LR=${3:-"0.01"}

echo "=========================================="
echo "PILOT: Image Size = $IMG_SIZE, Fold = $FOLD, LR = $LR"
echo "Job started at: $(date)"
echo "=========================================="

cd /cluster/home/tmstorma/NINA_fordypningsoppgave

module --quiet purge
module load Anaconda3/2024.02-1
source activate masterthesis

OUTPUT_DIR="multi_temporal_experiments/outputs/experiments/pilot_imgsize${IMG_SIZE}_fold${FOLD}"

python multi_temporal_experiments/scripts/modeling/train_multitemporal.py \
    --model-name lstm_unet \
    --temporal-sampling annual \
    --encoder-name resnet50 \
    --encoder-weights imagenet \
    --lstm-hidden-dim 512 \
    --lstm-num-layers 2 \
    --skip-aggregation max \
    --batch-size 4 \
    --image-size $IMG_SIZE \
    --num-workers 4 \
    --epochs 200 \
    --lr $LR \
    --momentum 0.9 \
    --weight-decay 5e-4 \
    --loss focal \
    --focal-alpha 0.25 \
    --focal-gamma 2.0 \
    --output-dir $OUTPUT_DIR \
    --seed 42 \
    --fold $FOLD \
    --num-folds 5

echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
