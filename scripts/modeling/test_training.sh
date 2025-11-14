#!/bin/bash
# Quick test of training script with 2 epochs

echo "Testing training script with 2 epochs..."
echo ""

module load Python/3.11.3-GCCcore-12.3.0

python3 scripts/modeling/train.py \
    --model-name early_fusion \
    --encoder-name resnet50 \
    --encoder-weights imagenet \
    --batch-size 2 \
    --image-size 512 \
    --num-workers 0 \
    --epochs 2 \
    --lr 0.01 \
    --momentum 0.9 \
    --weight-decay 5e-4 \
    --loss focal \
    --focal-alpha 0.25 \
    --focal-gamma 2.0 \
    --output-dir outputs/training/test_run \
    --seed 42

echo ""
echo "Training test complete!"
