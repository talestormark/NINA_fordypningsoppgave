#!/usr/bin/env python3
"""
Profile GPU memory usage for LSTM-UNet with different configurations.

Tests different temporal sampling modes and batch sizes to determine
optimal settings for training.
"""

import torch
import torch.cuda as cuda

from models_multitemporal import LSTMUNet, count_parameters


def profile_config(model, T, batch_size, image_size=512):
    """
    Profile GPU memory for a specific configuration.

    Args:
        model: LSTM-UNet model
        T: Number of time steps
        batch_size: Batch size
        image_size: Image size (default: 512)

    Returns:
        dict: Memory statistics
    """
    # Reset memory stats
    cuda.reset_peak_memory_stats()
    cuda.empty_cache()

    try:
        # Create input
        x = torch.randn(batch_size, T, 9, image_size, image_size).cuda()

        # Forward pass
        output = model(x)

        # Compute loss (to simulate training)
        target = torch.randint(0, 2, (batch_size, image_size, image_size)).float().cuda()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(output.squeeze(1), target)

        # Backward pass
        loss.backward()

        # Get memory stats
        memory_allocated = cuda.max_memory_allocated() / 1e9  # GB
        memory_reserved = cuda.max_memory_reserved() / 1e9    # GB

        # Cleanup
        del x, output, target, loss
        cuda.empty_cache()

        return {
            'success': True,
            'memory_allocated_gb': memory_allocated,
            'memory_reserved_gb': memory_reserved,
        }

    except RuntimeError as e:
        if "out of memory" in str(e):
            cuda.empty_cache()
            return {
                'success': False,
                'error': 'OOM',
                'memory_allocated_gb': None,
                'memory_reserved_gb': None,
            }
        else:
            raise


def main():
    print("=" * 100)
    print("LSTM-UNET MEMORY PROFILING")
    print("=" * 100)

    # Check GPU availability
    if not cuda.is_available():
        print("\n✗ No GPU available! This script requires a GPU.")
        return

    device = torch.device('cuda')
    gpu_name = cuda.get_device_name(0)
    gpu_memory_total = cuda.get_device_properties(0).total_memory / 1e9  # GB

    print(f"\nGPU: {gpu_name}")
    print(f"Total GPU Memory: {gpu_memory_total:.1f} GB")

    # Create model
    print("\nCreating LSTM-UNet model...")
    model = LSTMUNet(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=9,
        lstm_hidden_dim=512,
        lstm_num_layers=2,
        skip_aggregation="max",
    ).to(device)

    param_stats = count_parameters(model)
    print(f"Model parameters: {param_stats['total_millions']:.2f}M")

    # Test configurations
    print("\n" + "=" * 100)
    print("PROFILING DIFFERENT CONFIGURATIONS")
    print("=" * 100)

    configs = [
        # (name, T, batch_size, image_size)
        ("Bi-temporal (T=2)", 2, 8, 512),
        ("Bi-temporal (T=2)", 2, 4, 512),
        ("Bi-temporal (T=2)", 2, 2, 512),
        ("Annual (T=7)", 7, 8, 512),
        ("Annual (T=7)", 7, 4, 512),
        ("Annual (T=7)", 7, 2, 512),
        ("Quarterly (T=14)", 14, 8, 512),
        ("Quarterly (T=14)", 14, 4, 512),
        ("Quarterly (T=14)", 14, 2, 512),
        ("Quarterly (T=14)", 14, 1, 512),
    ]

    results = []

    for name, T, batch_size, image_size in configs:
        print(f"\nTesting: {name}, batch_size={batch_size}, image_size={image_size}")

        result = profile_config(model, T, batch_size, image_size)

        if result['success']:
            print(f"  ✓ Success!")
            print(f"    Memory allocated: {result['memory_allocated_gb']:.2f} GB")
            print(f"    Memory reserved: {result['memory_reserved_gb']:.2f} GB")
        else:
            print(f"  ✗ {result['error']}")

        results.append({
            'name': name,
            'T': T,
            'batch_size': batch_size,
            'image_size': image_size,
            **result
        })

    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY - RECOMMENDED CONFIGURATIONS")
    print("=" * 100)

    print(f"\n{'Configuration':<25} {'Batch Size':<12} {'Memory (GB)':<15} {'Status':<10}")
    print("-" * 100)

    for result in results:
        if result['success']:
            memory_str = f"{result['memory_allocated_gb']:.2f}"
            status = "✓ OK"
        else:
            memory_str = "OOM"
            status = "✗ Failed"

        print(f"{result['name']:<25} {result['batch_size']:<12} {memory_str:<15} {status:<10}")

    # Recommendations
    print("\n" + "=" * 100)
    print("RECOMMENDATIONS FOR TRAINING")
    print("=" * 100)

    successful_configs = [r for r in results if r['success']]

    if successful_configs:
        # Group by temporal sampling
        bi_temporal = [r for r in successful_configs if r['T'] == 2]
        annual = [r for r in successful_configs if r['T'] == 7]
        quarterly = [r for r in successful_configs if r['T'] == 14]

        print("\nOptimal batch sizes for 512x512 images:")

        if bi_temporal:
            max_batch = max(bi_temporal, key=lambda x: x['batch_size'])
            print(f"  • Bi-temporal (T=2):   batch_size={max_batch['batch_size']} "
                  f"({max_batch['memory_allocated_gb']:.2f} GB)")

        if annual:
            max_batch = max(annual, key=lambda x: x['batch_size'])
            print(f"  • Annual (T=7):        batch_size={max_batch['batch_size']} "
                  f"({max_batch['memory_allocated_gb']:.2f} GB)")

        if quarterly:
            max_batch = max(quarterly, key=lambda x: x['batch_size'])
            print(f"  • Quarterly (T=14):    batch_size={max_batch['batch_size']} "
                  f"({max_batch['memory_allocated_gb']:.2f} GB)")

        print(f"\nGPU memory available: {gpu_memory_total:.1f} GB")

        # Check if we need 80GB GPU
        max_memory = max(r['memory_allocated_gb'] for r in successful_configs)
        if max_memory > 32:
            print(f"\n⚠ WARNING: Maximum memory usage ({max_memory:.1f} GB) exceeds 32GB.")
            print("  → Recommend requesting 80GB GPU for quarterly sampling")
            print("  → Use SLURM constraint: --constraint=gpu80g")
        else:
            print(f"\n✓ All configurations fit within 32GB GPU memory")

    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
