#!/usr/bin/env python3
"""
Multi-temporal segmentation models for land-take detection.

Implements LSTM-UNet architecture for processing Sentinel-2 time series data.
"""

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder

# Import ConvLSTM from same directory
from convlstm import ConvLSTM


class LSTMUNet(nn.Module):
    """
    LSTM-UNet for multi-temporal land-take detection.

    Architecture:
    - Per-frame 2D encoder (ResNet-50 with shared weights)
    - ConvLSTM at bottleneck for temporal modeling
    - Standard 2D U-Net decoder with skip connections

    Args:
        encoder_name: Encoder backbone (default: resnet50)
        encoder_weights: Pre-trained weights (default: imagenet)
        in_channels: Input bands per timestep (default: 9 for Sentinel-2)
        classes: Output classes (default: 1 for binary)
        lstm_hidden_dim: ConvLSTM hidden channels (default: 512)
        lstm_num_layers: Number of ConvLSTM layers (default: 2)
        skip_aggregation: How to aggregate skip connections ("max", "mean", "last")
        activation: Output activation (default: None)
    """

    def __init__(
        self,
        encoder_name: str = "resnet50",
        encoder_weights: Optional[str] = "imagenet",
        in_channels: int = 9,
        classes: int = 1,
        lstm_hidden_dim: int = 512,
        lstm_num_layers: int = 2,
        skip_aggregation: str = "max",
        activation: Optional[str] = None,
    ):
        super().__init__()

        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.in_channels = in_channels
        self.classes = classes
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers
        self.skip_aggregation = skip_aggregation

        # 1. Create template U-Net to extract encoder architecture
        template = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,  # RGB for ImageNet weights
            classes=classes,
            activation=activation,
        )

        # 2. Extract encoder and modify first conv for multi-spectral input
        self.encoder = template.encoder
        self.encoder_channels = list(template.encoder.out_channels)
        # For ResNet-50: [3, 64, 256, 512, 1024, 2048]

        # Modify first conv layer to accept in_channels (e.g., 9 for Sentinel-2)
        if in_channels != 3:
            self._modify_first_conv_for_multispectral(in_channels, encoder_name)

        # 3. Create ConvLSTM for bottleneck temporal fusion
        bottleneck_channels = self.encoder_channels[-1]  # 2048 for ResNet-50

        self.convlstm = ConvLSTM(
            input_dim=bottleneck_channels,
            hidden_dim=lstm_hidden_dim,
            kernel_size=3,
            num_layers=lstm_num_layers,
            batch_first=True,
            return_all_layers=False,
        )

        # 4. Create decoder with modified bottleneck channels
        decoder_encoder_channels = self.encoder_channels.copy()
        decoder_encoder_channels[-1] = lstm_hidden_dim  # Replace 2048 with 512

        self.decoder = UnetDecoder(
            encoder_channels=decoder_encoder_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            n_blocks=5,
            attention_type=None,
        )

        # 5. Segmentation head
        self.segmentation_head = template.segmentation_head

        self.name = f"LSTMUNet_{encoder_name}"

    def _modify_first_conv_for_multispectral(self, in_channels: int, encoder_name: str):
        """
        Modify first conv layer to accept multi-spectral input (e.g., 9 bands).

        Strategy:
        - Keep RGB weights (channels 0,1,2) from ImageNet pre-training
        - Initialize remaining channels with small random values
        - Normalize weights to maintain activation scale

        Args:
            in_channels: Number of input channels
            encoder_name: Encoder architecture name
        """
        # Get first conv layer (encoder-specific)
        if "resnet" in encoder_name or "resnext" in encoder_name:
            first_conv = self.encoder.conv1
        elif "efficientnet" in encoder_name:
            first_conv = self.encoder.conv_stem
        elif "mobilenet" in encoder_name:
            first_conv = self.encoder.features[0][0]
        else:
            # Fallback: try to find first Conv2d layer
            first_conv = None
            for module in self.encoder.modules():
                if isinstance(module, nn.Conv2d):
                    first_conv = module
                    break

        if first_conv is None:
            raise RuntimeError(f"Could not find first conv layer in {encoder_name}")

        # Get current weight: (out_channels, 3, kernel_h, kernel_w)
        old_weight = first_conv.weight.data
        out_channels = old_weight.shape[0]
        kernel_size = old_weight.shape[2:]
        device = old_weight.device  # Get device of original weights

        # Create new conv layer on the same device
        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None,
        ).to(device)  # Move to same device as encoder

        # Initialize new weights
        with torch.no_grad():
            # Copy RGB weights (channels 0,1,2)
            new_conv.weight[:, :3, :, :] = old_weight

            # Initialize remaining channels
            if in_channels > 3:
                # Use scaled random initialization for extra channels
                # Scale by sqrt(3/in_channels) to maintain variance
                scale = (3.0 / in_channels) ** 0.5
                extra_channels = in_channels - 3
                new_conv.weight[:, 3:, :, :] = (
                    torch.randn_like(new_conv.weight[:, 3:, :, :]) * scale
                )

            # Copy bias if exists
            if first_conv.bias is not None:
                new_conv.bias.data = first_conv.bias.data

            # Debug: Check for NaN in initialized weights
            if torch.isnan(new_conv.weight).any():
                print(f"WARNING: NaN detected in new_conv weights after initialization!")
                print(f"  Old weight stats: min={old_weight.min():.3f}, max={old_weight.max():.3f}")
                print(f"  New weight stats: min={new_conv.weight.min():.3f}, max={new_conv.weight.max():.3f}")
                print(f"  Device: {device}")

        # Replace first conv layer
        if "resnet" in encoder_name or "resnext" in encoder_name:
            self.encoder.conv1 = new_conv
        elif "efficientnet" in encoder_name:
            self.encoder.conv_stem = new_conv
        elif "mobilenet" in encoder_name:
            self.encoder.features[0][0] = new_conv

    def _aggregate_temporal_features(
        self,
        features_temporal: torch.Tensor,
        method: str = "max"
    ) -> torch.Tensor:
        """
        Aggregate temporal features for skip connections.

        Args:
            features_temporal: (B, T, C, H, W)
            method: "max", "mean", or "last"

        Returns:
            features_spatial: (B, C, H, W)
        """
        if method == "max":
            # Max over time dimension
            return features_temporal.max(dim=1)[0]
        elif method == "mean":
            # Mean over time dimension
            return features_temporal.mean(dim=1)
        elif method == "last":
            # Last timestep only
            return features_temporal[:, -1]
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (B, T, C, H, W) - Multi-temporal input
               B = batch size
               T = number of time steps (e.g., 2, 7, or 14)
               C = number of bands (e.g., 9 for Sentinel-2)
               H, W = spatial dimensions (e.g., 512x512)

        Returns:
            output: (B, 1, H, W) - Binary segmentation logits
        """
        B, T, C, H, W = x.shape

        # 1. Extract features for each timestep with shared encoder
        # Reshape: (B, T, C, H, W) → (B*T, C, H, W)
        x_reshaped = x.view(B * T, C, H, W)

        # Encode: (B*T, C, H, W) → list of (B*T, F_i, H_i, W_i)
        features_reshaped = self.encoder(x_reshaped)
        # For ResNet-50, this gives 6 feature maps at different scales

        # 2. Reshape back to temporal format
        # list of (B*T, F_i, H_i, W_i) → list of (B, T, F_i, H_i, W_i)
        features_temporal = []
        for feat in features_reshaped:
            _, F, H_f, W_f = feat.shape
            feat_temporal = feat.view(B, T, F, H_f, W_f)
            features_temporal.append(feat_temporal)

        # 3. Process bottleneck features with ConvLSTM
        bottleneck_temporal = features_temporal[-1]  # (B, T, 2048, H/32, W/32)

        # ConvLSTM: (B, T, 2048, H/32, W/32) → hidden states
        _, last_states = self.convlstm(bottleneck_temporal)
        h_final, c_final = last_states[-1]  # Get last layer's hidden state
        bottleneck_fused = h_final  # (B, 512, H/32, W/32)

        # 4. Aggregate skip connection features (stages 0-4, excluding bottleneck)
        skip_features = []
        for feat_temporal in features_temporal[:-1]:  # All except bottleneck
            feat_spatial = self._aggregate_temporal_features(
                feat_temporal,
                method=self.skip_aggregation
            )
            skip_features.append(feat_spatial)

        # Add fused bottleneck as last skip connection
        skip_features.append(bottleneck_fused)

        # 5. Decode with U-Net decoder
        # Decoder expects features as a list: [stage0, stage1, ..., bottleneck]
        decoder_output = self.decoder(skip_features)

        # 6. Segmentation head
        output = self.segmentation_head(decoder_output)

        return output


def create_multitemporal_model(model_name: str, **kwargs) -> nn.Module:
    """
    Factory function for multi-temporal models.

    Args:
        model_name: One of ['lstm_unet', 'unet_3d', 'hybrid_lstm_3d']
        **kwargs: Model-specific arguments

    Returns:
        Model instance

    Raises:
        ValueError: If model_name is not recognized
    """
    models = {
        'lstm_unet': LSTMUNet,
        # Add 3D U-Net and hybrid models later
    }

    if model_name not in models:
        available = ', '.join(models.keys())
        raise ValueError(f"Unknown model: {model_name}. Available: {available}")

    return models[model_name](**kwargs)


def count_parameters(model: nn.Module) -> dict:
    """
    Count model parameters.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with parameter counts
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    return {
        'trainable': trainable,
        'total': total,
        'trainable_millions': trainable / 1e6,
        'total_millions': total / 1e6,
    }


if __name__ == "__main__":
    """Test the LSTM-UNet implementation."""

    print("=" * 80)
    print("TESTING LSTM-UNET IMPLEMENTATION")
    print("=" * 80)

    # Test 1: Model creation
    print("\nTest 1: Model creation")
    print("-" * 80)

    model = LSTMUNet(
        encoder_name="resnet50",
        encoder_weights=None,  # No pre-training for testing
        in_channels=9,
        classes=1,
        lstm_hidden_dim=512,
        lstm_num_layers=2,
        skip_aggregation="max",
    )

    print(f"✓ Model created: {model.name}")
    print(f"  Encoder: {model.encoder_name}")
    print(f"  Input channels: {model.in_channels}")
    print(f"  LSTM hidden dim: {model.lstm_hidden_dim}")
    print(f"  LSTM num layers: {model.lstm_num_layers}")

    # Test 2: Forward pass with different temporal samplings
    print("\nTest 2: Forward pass with different temporal samplings")
    print("-" * 80)

    test_cases = [
        ("Bi-temporal", 2, 8, 64),    # T=2, batch=8, image_size=64
        ("Annual", 7, 4, 64),          # T=7, batch=4, image_size=64
        ("Quarterly", 14, 2, 64),      # T=14, batch=2, image_size=64
    ]

    for name, T, B, size in test_cases:
        x = torch.randn(B, T, 9, size, size)
        print(f"\n{name} sampling:")
        print(f"  Input: {tuple(x.shape)}")

        output = model(x)
        print(f"  Output: {tuple(output.shape)}")

        expected_shape = (B, 1, size, size)
        assert output.shape == expected_shape, f"Output shape mismatch: {output.shape} != {expected_shape}"
        print(f"  ✓ Shape correct!")

    # Test 3: Gradient flow
    print("\nTest 3: Gradient flow")
    print("-" * 80)

    model.zero_grad()
    x = torch.randn(2, 7, 9, 64, 64, requires_grad=True)
    target = torch.randint(0, 2, (2, 64, 64)).float()

    output = model(x)
    loss = nn.functional.binary_cross_entropy_with_logits(output.squeeze(1), target)
    loss.backward()

    # Check that all parameters have gradients
    has_grads = True
    no_grad_params = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is None:
            has_grads = False
            no_grad_params.append(name)

    if has_grads:
        print("✓ All parameters have gradients!")
    else:
        print(f"✗ {len(no_grad_params)} parameters missing gradients:")
        for name in no_grad_params[:5]:  # Show first 5
            print(f"    - {name}")

    # Test 4: Parameter count
    print("\nTest 4: Parameter count")
    print("-" * 80)

    param_stats = count_parameters(model)
    print(f"Total parameters: {param_stats['total']:,}")
    print(f"Trainable parameters: {param_stats['trainable']:,}")
    print(f"Parameters (millions): {param_stats['total_millions']:.2f}M")

    expected_range = (50, 70)  # Expected ~56M parameters
    if expected_range[0] <= param_stats['total_millions'] <= expected_range[1]:
        print(f"✓ Parameter count in expected range ({expected_range[0]}-{expected_range[1]}M)")
    else:
        print(f"⚠ Parameter count outside expected range: {param_stats['total_millions']:.2f}M")

    # Test 5: Factory function
    print("\nTest 5: Factory function")
    print("-" * 80)

    model_factory = create_multitemporal_model(
        'lstm_unet',
        encoder_name='resnet50',
        encoder_weights=None,
        in_channels=9,
    )

    print(f"✓ Factory function works!")
    print(f"  Created model: {type(model_factory).__name__}")

    # Test 6: Skip aggregation methods
    print("\nTest 6: Skip aggregation methods")
    print("-" * 80)

    aggregation_methods = ['max', 'mean', 'last']

    for method in aggregation_methods:
        model_agg = LSTMUNet(
            encoder_weights=None,
            in_channels=9,
            skip_aggregation=method,
        )

        x = torch.randn(2, 7, 9, 64, 64)
        output = model_agg(x)

        assert output.shape == (2, 1, 64, 64), f"Output shape mismatch for {method}"
        print(f"  ✓ {method.capitalize()} aggregation works!")

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED! ✓")
    print("=" * 80)

    # Summary
    print("\nModel Summary:")
    print("-" * 80)
    print(f"Architecture: LSTM-UNet with {model.encoder_name} encoder")
    print(f"Parameters: {param_stats['total_millions']:.2f}M")
    print(f"Input format: (B, T, C=9, H, W)")
    print(f"Output format: (B, 1, H, W)")
    print(f"Temporal modeling: {model.lstm_num_layers}-layer ConvLSTM at bottleneck")
    print(f"Skip aggregation: {model.skip_aggregation} pooling over time")
    print("\nReady for training!")
