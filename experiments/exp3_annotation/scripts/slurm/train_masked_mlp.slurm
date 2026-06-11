#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --time=0-06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4

# Usage:
#   sbatch --job-name=ae_E5_f0 train_masked_mlp.slurm D2_alphaearth 0 sparse_labels_seed42.json [OUTPUT_DIR]

EXPERIMENT=${1:?"Usage: sbatch train_masked_mlp.slurm EXPERIMENT FOLD LABELS_FILE [OUTPUT_DIR]"}
FOLD=${2:?"Usage: sbatch train_masked_mlp.slurm EXPERIMENT FOLD LABELS_FILE [OUTPUT_DIR]"}
LABELS_FILE=${3:?"Usage: sbatch train_masked_mlp.slurm EXPERIMENT FOLD LABELS_FILE [OUTPUT_DIR]"}
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

echo "=== Per-pixel MLP: ${EXPERIMENT} fold ${FOLD} ==="
echo "Labels: ${LABELS}"
echo ""

python experiments/annotation_efficiency/scripts/modeling/train_masked_mlp.py \
    --experiment $EXPERIMENT \
    --fold $FOLD \
    --sparse-labels $LABELS \
    --epochs 400 \
    --lr 0.01 \
    --batch-size 8 \
    --seed 42 \
    $EXTRA_ARGS

echo "=== Done ==="
