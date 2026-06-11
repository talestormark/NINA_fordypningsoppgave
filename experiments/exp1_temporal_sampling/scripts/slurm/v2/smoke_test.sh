#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=smoke_test
#SBATCH --time=0-00:15:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=4
#SBATCH --output=experiments/exp1_temporal_sampling/outputs_v2/logs/smoke_test_%j.log
#SBATCH --error=experiments/exp1_temporal_sampling/outputs_v2/logs/smoke_test_%j.err

cd /cluster/home/tmstorma/NINA_fordypningsoppgave
mkdir -p experiments/exp1_temporal_sampling/outputs_v2/logs

module --quiet purge
module load Anaconda3/2024.02-1
source activate masterthesis

echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

python experiments/exp1_temporal_sampling/scripts/modeling/train_multitemporal.py \
    --model-name lstm_unet --temporal-sampling annual \
    --encoder-name resnet50 --encoder-weights imagenet \
    --lstm-hidden-dim 256 --lstm-num-layers 2 \
    --batch-size 4 --image-size 64 --num-workers 2 \
    --epochs 2 --lr 0.01 --optimizer adamw --scheduler cosine \
    --loss focal --focal-alpha 0.25 --focal-gamma 2.0 \
    --output-dir experiments/exp1_temporal_sampling/outputs_v2/smoke_test \
    --seed 42 --fold 0 --num-folds 5 \
    --data-dir data_v2 --mask-subdir Land_take_masks_coarse \
    --splits-dir preprocessing/outputs/splits/part1 \
    --change-level-path preprocessing/outputs/splits/part1/split_info.csv

echo ""
echo "Exit code: $?"
echo "Finished at: $(date)"
