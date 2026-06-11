#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --time=0-06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4

# Usage:
#   sbatch --job-name=ae_E4s2_f0 train_masked_unet.slurm A3_s2_9band 0 sparse_labels_seed42.json
#   sbatch --job-name=ae_E4n5_f0 train_masked_unet.slurm D2_alphaearth 0 sparse_labels_n5_seed42.json

EXPERIMENT=${1:?"Usage: sbatch train_masked_unet.slurm EXPERIMENT FOLD LABELS_FILE [OUTPUT_DIR]"}
FOLD=${2:?"Usage: sbatch train_masked_unet.slurm EXPERIMENT FOLD LABELS_FILE [OUTPUT_DIR]"}
LABELS_FILE=${3:?"Usage: sbatch train_masked_unet.slurm EXPERIMENT FOLD LABELS_FILE [OUTPUT_DIR]"}
OUTPUT_DIR=${4:-""}

module --quiet purge
module load Anaconda3/2024.02-1
conda activate masterthesis

cd /cluster/home/tmstorma/NINA_fordypningsoppgave

LABELS="experiments/annotation_efficiency/outputs/sparse_labels/${LABELS_FILE}"

EXTRA_ARGS=""
if [ -n "$OUTPUT_DIR" ]; then
    EXTRA_ARGS="--output-dir $OUTPUT_DIR"
fi

echo "=== Masked-loss U-Net: ${EXPERIMENT} fold ${FOLD} ==="
echo "Labels: ${LABELS}"
echo ""

python experiments/annotation_efficiency/scripts/modeling/train_masked_unet.py \
    --experiment $EXPERIMENT \
    --fold $FOLD \
    --sparse-labels $LABELS \
    --epochs 400 \
    --lr 0.01 \
    --batch-size 8 \
    --seed 42 \
    $EXTRA_ARGS

echo "=== Done ==="
