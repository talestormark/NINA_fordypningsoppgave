#!/usr/bin/env python3
"""
Comprehensive model testing script.

Tests:
1. Model initialization with/without pretrained weights
2. Forward pass with random data
3. Forward pass with real data from dataset
4. Output shapes and value ranges
5. Parameter counts and memory usage
6. Gradient flow
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path

# Import models
from models import (
    UNetEarlyFusion,
    UNetSiamDiff,
    UNetSiamConc,
    create_model,
    count_parameters,
)

# Import dataset for real data testing
from dataset import LandTakeDataset, load_refids_from_file, get_validation_augmentation


def test_model_initialization():
    """Test that all models can be initialized with and without pretrained weights."""
    print("=" * 80)
    print("TEST 1: Model Initialization")
    print("=" * 80)

    models_config = [
        ('UNetEarlyFusion', UNetEarlyFusion, {}),
        ('UNetSiamDiff', UNetSiamDiff, {}),
        ('UNetSiamConc', UNetSiamConc, {}),
    ]

    for name, model_class, kwargs in models_config:
        print(f"\n{name}:")

        # Test without pretrained weights
        print("  Without pretrained weights...", end=" ")
        model_no_pretrain = model_class(encoder_weights=None, **kwargs)
        print(f"✓ (name: {model_no_pretrain.name})")

        # Test with pretrained weights
        print("  With ImageNet pretrained weights...", end=" ")
        model_pretrain = model_class(encoder_weights='imagenet', **kwargs)
        print("✓")

        # Clean up
        del model_no_pretrain, model_pretrain

    print("\n✓ All models initialized successfully")


def test_forward_pass_random():
    """Test forward pass with random tensors."""
    print("\n" + "=" * 80)
    print("TEST 2: Forward Pass with Random Data")
    print("=" * 80)

    batch_size = 2
    height, width = 512, 512

    # Test Early Fusion
    print("\nU-Net Early Fusion:")
    model_ef = UNetEarlyFusion(encoder_weights=None)
    model_ef.eval()

    x_concat = torch.randn(batch_size, 6, height, width)
    print(f"  Input shape: {x_concat.shape}")

    with torch.no_grad():
        output_ef = model_ef(x_concat)

    print(f"  Output shape: {output_ef.shape}")
    print(f"  Output range: [{output_ef.min():.3f}, {output_ef.max():.3f}]")

    # Check output shape
    assert output_ef.shape == (batch_size, 1, height, width), \
        f"Expected shape ({batch_size}, 1, {height}, {width}), got {output_ef.shape}"
    print("  ✓ Output shape correct")

    # Test SiamDiff
    print("\nU-Net SiamDiff:")
    model_sd = UNetSiamDiff(encoder_weights=None)
    model_sd.eval()

    x1 = torch.randn(batch_size, 3, height, width)
    x2 = torch.randn(batch_size, 3, height, width)
    print(f"  Input shape (per image): {x1.shape}")

    with torch.no_grad():
        output_sd = model_sd(x1, x2)

    print(f"  Output shape: {output_sd.shape}")
    print(f"  Output range: [{output_sd.min():.3f}, {output_sd.max():.3f}]")

    assert output_sd.shape == (batch_size, 1, height, width)
    print("  ✓ Output shape correct")

    # Test SiamConc
    print("\nU-Net SiamConc:")
    model_sc = UNetSiamConc(encoder_weights=None)
    model_sc.eval()

    with torch.no_grad():
        output_sc = model_sc(x1, x2)

    print(f"  Output shape: {output_sc.shape}")
    print(f"  Output range: [{output_sc.min():.3f}, {output_sc.max():.3f}]")

    assert output_sc.shape == (batch_size, 1, height, width)
    print("  ✓ Output shape correct")

    print("\n✓ All forward passes successful with random data")


def test_forward_pass_real_data():
    """Test forward pass with real data from dataset."""
    print("\n" + "=" * 80)
    print("TEST 3: Forward Pass with Real Data")
    print("=" * 80)

    # Load a few refids
    train_refids = load_refids_from_file("outputs/splits/train_refids.txt")[:2]
    print(f"\nUsing {len(train_refids)} tiles from training set")

    # Create datasets for both modes
    print("\nCreating datasets...")

    # Dataset for Early Fusion (concatenated images)
    dataset_concat = LandTakeDataset(
        refids=train_refids,
        transform=get_validation_augmentation(image_size=512),
        return_separate_images=False,
    )

    # Dataset for Siamese models (separate images)
    dataset_separate = LandTakeDataset(
        refids=train_refids,
        transform=get_validation_augmentation(image_size=512),
        return_separate_images=True,
    )

    print("✓ Datasets created")

    # Test Early Fusion with real data
    print("\nU-Net Early Fusion with real data:")
    model_ef = UNetEarlyFusion(encoder_weights='imagenet')
    model_ef.eval()

    sample_concat = dataset_concat[0]
    x_real = sample_concat['image'].unsqueeze(0)  # Add batch dimension
    print(f"  Input shape: {x_real.shape}")
    print(f"  Input range: [{x_real.min():.3f}, {x_real.max():.3f}]")
    print(f"  RefID: {sample_concat['refid']}")

    with torch.no_grad():
        output_real_ef = model_ef(x_real)

    print(f"  Output shape: {output_real_ef.shape}")
    print(f"  Output range: [{output_real_ef.min():.3f}, {output_real_ef.max():.3f}]")
    print("  ✓ Forward pass successful")

    # Test Siamese models with real data
    print("\nU-Net SiamDiff with real data:")
    model_sd = UNetSiamDiff(encoder_weights='imagenet')
    model_sd.eval()

    sample_separate = dataset_separate[0]
    x1_real = sample_separate['image_2018'].unsqueeze(0)
    x2_real = sample_separate['image_2025'].unsqueeze(0)
    print(f"  Input shape (per image): {x1_real.shape}")
    print(f"  Image 2018 range: [{x1_real.min():.3f}, {x1_real.max():.3f}]")
    print(f"  Image 2025 range: [{x2_real.min():.3f}, {x2_real.max():.3f}]")

    with torch.no_grad():
        output_real_sd = model_sd(x1_real, x2_real)

    print(f"  Output shape: {output_real_sd.shape}")
    print(f"  Output range: [{output_real_sd.min():.3f}, {output_real_sd.max():.3f}]")
    print("  ✓ Forward pass successful")

    print("\n✓ All models work with real data from dataset")


def test_parameter_counts():
    """Test parameter counting for all models."""
    print("\n" + "=" * 80)
    print("TEST 4: Parameter Counts")
    print("=" * 80)

    models_config = [
        ('Early Fusion', UNetEarlyFusion, {}),
        ('SiamDiff', UNetSiamDiff, {}),
        ('SiamConc', UNetSiamConc, {}),
    ]

    print(f"\n{'Model':<20} {'Trainable (M)':<15} {'Total (M)':<15}")
    print("-" * 50)

    for name, model_class, kwargs in models_config:
        model = model_class(encoder_weights=None, **kwargs)
        params = count_parameters(model)

        print(f"{name:<20} {params['trainable_millions']:>14.2f} {params['total_millions']:>14.2f}")

        # Check that parameters are reasonable
        assert 10 < params['trainable_millions'] < 100, \
            f"{name}: Unexpected parameter count {params['trainable_millions']:.2f}M"

        del model

    print("\n✓ All parameter counts are reasonable")


def test_gradient_flow():
    """Test that gradients flow through the models."""
    print("\n" + "=" * 80)
    print("TEST 5: Gradient Flow")
    print("=" * 80)

    batch_size = 2
    height, width = 512, 512

    # Test Early Fusion
    print("\nU-Net Early Fusion gradient flow:")
    model_ef = UNetEarlyFusion(encoder_weights=None)
    model_ef.train()

    x = torch.randn(batch_size, 6, height, width, requires_grad=True)
    target = torch.randint(0, 2, (batch_size, 1, height, width)).float()

    output = model_ef(x)
    loss = nn.functional.binary_cross_entropy_with_logits(output, target)

    print(f"  Loss: {loss.item():.4f}")

    loss.backward()

    # Check that gradients exist
    assert x.grad is not None, "No gradient for input"
    print(f"  Input gradient range: [{x.grad.min():.6f}, {x.grad.max():.6f}]")

    # Check that model parameters have gradients
    grad_count = sum(1 for p in model_ef.parameters() if p.grad is not None)
    total_count = sum(1 for _ in model_ef.parameters())
    print(f"  Parameters with gradients: {grad_count}/{total_count}")

    assert grad_count == total_count, "Not all parameters have gradients"
    print("  ✓ Gradients flow correctly")

    # Test SiamDiff
    print("\nU-Net SiamDiff gradient flow:")
    model_sd = UNetSiamDiff(encoder_weights=None)
    model_sd.train()

    x1 = torch.randn(batch_size, 3, height, width, requires_grad=True)
    x2 = torch.randn(batch_size, 3, height, width, requires_grad=True)

    output = model_sd(x1, x2)
    loss = nn.functional.binary_cross_entropy_with_logits(output, target)

    print(f"  Loss: {loss.item():.4f}")

    loss.backward()

    assert x1.grad is not None and x2.grad is not None, "No gradients for inputs"
    print(f"  Input 1 gradient range: [{x1.grad.min():.6f}, {x1.grad.max():.6f}]")
    print(f"  Input 2 gradient range: [{x2.grad.min():.6f}, {x2.grad.max():.6f}]")

    grad_count = sum(1 for p in model_sd.parameters() if p.grad is not None)
    total_count = sum(1 for _ in model_sd.parameters())
    print(f"  Parameters with gradients: {grad_count}/{total_count}")

    assert grad_count == total_count, "Not all parameters have gradients"
    print("  ✓ Gradients flow correctly")

    print("\n✓ Gradients flow through all models")


def test_factory_function():
    """Test the model factory function."""
    print("\n" + "=" * 80)
    print("TEST 6: Factory Function")
    print("=" * 80)

    model_names = ['early_fusion', 'siam_diff', 'siam_conc']

    print("\nCreating models via factory function:")
    for name in model_names:
        model = create_model(name, encoder_weights=None)
        print(f"  {name:20s} → {model.name}")
        del model

    print("\n✓ Factory function works correctly")

    # Test invalid name
    print("\nTesting invalid model name:")
    try:
        model = create_model('invalid_name')
        print("  ✗ Should have raised ValueError")
    except ValueError as e:
        print(f"  ✓ Correctly raised ValueError: {str(e)[:50]}...")


def test_memory_usage():
    """Estimate memory usage for models."""
    print("\n" + "=" * 80)
    print("TEST 7: Memory Usage Estimates")
    print("=" * 80)

    batch_size = 4
    height, width = 512, 512

    print(f"\nBatch size: {batch_size}, Image size: {height}×{width}")
    print("\nMemory estimates (approximate):")
    print(f"{'Component':<30} {'Memory (MB)':<15}")
    print("-" * 45)

    # Input tensors
    input_concat_size = batch_size * 6 * height * width * 4 / (1024**2)  # 4 bytes per float32
    input_separate_size = batch_size * 3 * height * width * 4 / (1024**2) * 2  # Two images

    print(f"{'Input (Early Fusion)':<30} {input_concat_size:>14.1f}")
    print(f"{'Input (Siamese, both images)':<30} {input_separate_size:>14.1f}")

    # Model parameters (approximate)
    model = UNetEarlyFusion(encoder_weights=None)
    params = count_parameters(model)
    model_memory = params['total'] * 4 / (1024**2)  # 4 bytes per parameter
    print(f"{'Model parameters':<30} {model_memory:>14.1f}")

    # Output tensor
    output_size = batch_size * 1 * height * width * 4 / (1024**2)
    print(f"{'Output':<30} {output_size:>14.1f}")

    # Total estimate
    total_estimate = input_concat_size + model_memory + output_size
    print(f"{'Total (approximate)':<30} {total_estimate:>14.1f}")

    print(f"\n✓ Estimated memory usage: ~{total_estimate:.0f} MB per forward pass")
    print(f"  (Note: Actual usage may be 2-3x higher due to gradients and optimizer states)")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("BASELINE MODELS TESTING")
    print("=" * 80)

    try:
        # Run all tests
        test_model_initialization()
        test_forward_pass_random()
        test_forward_pass_real_data()
        test_parameter_counts()
        test_gradient_flow()
        test_factory_function()
        test_memory_usage()

        # Summary
        print("\n" + "=" * 80)
        print("ALL TESTS PASSED ✓")
        print("=" * 80)
        print("\nModels are ready for training!")
        print("\nNext steps:")
        print("  1. Implement training script with Focal Loss")
        print("  2. Set up evaluation metrics (F1, IoU, precision, recall)")
        print("  3. Configure optimizer and learning rate schedule")
        print("  4. Start baseline training on VHR data")

        return 0

    except Exception as e:
        print("\n" + "=" * 80)
        print("TEST FAILED ✗")
        print("=" * 80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
