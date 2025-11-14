#!/usr/bin/env python3
"""
Baseline models for land-take detection using bi-temporal VHR imagery.

Implements three U-Net based architectures:
1. U-Net Early Fusion: Concatenate images early (6-channel input)
2. U-Net SiamDiff: Siamese encoders with feature difference
3. U-Net SiamConc: Siamese encoders with feature concatenation

All models use ResNet-50 encoder and U-Net decoder with skip connections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from typing import Optional, List


class UNetEarlyFusion(nn.Module):
    """
    U-Net with early fusion of bi-temporal images.

    Architecture:
    - Concatenate 2018 and 2025 RGB images → 6 channels
    - Single ResNet-50 encoder (modified first conv for 6 channels)
    - U-Net decoder with skip connections
    - Binary segmentation output

    Args:
        encoder_name: Encoder architecture (default: resnet50)
        encoder_weights: Pre-trained weights (default: imagenet)
        in_channels: Number of input channels (default: 6 for bi-temporal RGB)
        classes: Number of output classes (default: 1 for binary)
        activation: Output activation function (default: None, use sigmoid externally)
    """

    def __init__(
        self,
        encoder_name: str = "resnet50",
        encoder_weights: Optional[str] = "imagenet",
        in_channels: int = 6,
        classes: int = 1,
        activation: Optional[str] = None,
    ):
        super().__init__()

        # Use segmentation_models_pytorch Unet
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation,
        )

        self.name = "UNet_EarlyFusion"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, 6, H, W) - concatenated bi-temporal RGB

        Returns:
            Output tensor of shape (B, 1, H, W) - binary segmentation logits
        """
        return self.model(x)


class UNetSiamese(nn.Module):
    """
    Base class for Siamese U-Net architectures.

    Architecture:
    - Two ResNet-50 encoders with shared weights
    - Feature fusion at bottleneck (difference or concatenation)
    - U-Net decoder with skip connections
    - Binary segmentation output

    Args:
        encoder_name: Encoder architecture (default: resnet50)
        encoder_weights: Pre-trained weights (default: imagenet)
        classes: Number of output classes (default: 1 for binary)
        fusion_mode: How to fuse features ('diff' or 'concat')
        activation: Output activation function (default: None)
    """

    def __init__(
        self,
        encoder_name: str = "resnet50",
        encoder_weights: Optional[str] = "imagenet",
        classes: int = 1,
        fusion_mode: str = "diff",
        activation: Optional[str] = None,
    ):
        super().__init__()

        assert fusion_mode in ["diff", "concat"], "fusion_mode must be 'diff' or 'concat'"
        self.fusion_mode = fusion_mode

        # Create a template model to extract encoder and decoder
        template = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,  # Each branch processes 3-channel RGB
            classes=classes,
            activation=activation,
        )

        # Shared encoder (will be used for both branches)
        self.encoder = template.encoder

        # Get encoder output channels for each stage
        self.encoder_channels = template.encoder.out_channels

        # Decoder expects encoder channels
        # For diff: channels stay same [3, 64, 256, 512, 1024, 2048]
        # For concat: double the bottleneck [3, 64, 256, 512, 1024, 4096]
        if fusion_mode == "concat":
            # Modify encoder channels for decoder
            decoder_channels = list(self.encoder_channels)
            decoder_channels[-1] = decoder_channels[-1] * 2  # Double bottleneck channels

            # Create custom decoder with modified input channels
            from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
            self.decoder = UnetDecoder(
                encoder_channels=decoder_channels,
                decoder_channels=(256, 128, 64, 32, 16),
                n_blocks=5,
                use_norm='batchnorm',
                add_center_block=False,
                attention_type=None,
            )
        else:
            # Use original decoder
            self.decoder = template.decoder

        # Segmentation head
        self.segmentation_head = template.segmentation_head

        # Name for logging
        self.name = f"UNet_Siamese{'Diff' if fusion_mode == 'diff' else 'Conc'}"

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with two separate images.

        Args:
            x1: First image tensor of shape (B, 3, H, W) - 2018 RGB
            x2: Second image tensor of shape (B, 3, H, W) - 2025 RGB

        Returns:
            Output tensor of shape (B, 1, H, W) - binary segmentation logits
        """
        # Encode both images with shared encoder
        features1 = self.encoder(x1)
        features2 = self.encoder(x2)

        # Fuse features at each encoder stage
        if self.fusion_mode == "diff":
            # Absolute difference
            fused_features = [torch.abs(f2 - f1) for f1, f2 in zip(features1, features2)]
        else:  # concat
            # Concatenate along channel dimension
            # For all stages except bottleneck, use difference (for skip connections)
            # Only concatenate at bottleneck
            fused_features = []
            for i, (f1, f2) in enumerate(zip(features1, features2)):
                if i == len(features1) - 1:  # Bottleneck (last stage)
                    fused_features.append(torch.cat([f1, f2], dim=1))
                else:  # Skip connection stages
                    fused_features.append(torch.abs(f2 - f1))

        # Decode
        decoder_output = self.decoder(fused_features)

        # Segmentation head
        masks = self.segmentation_head(decoder_output)

        return masks


class UNetSiamDiff(UNetSiamese):
    """
    U-Net with Siamese encoders and feature difference fusion.

    Architecture:
    - Two ResNet-50 encoders with shared weights
    - Absolute difference at bottleneck: |f_t2 - f_t1|
    - U-Net decoder with skip connections from difference features
    - Binary segmentation output

    The difference operation emphasizes changes between time points.
    """

    def __init__(
        self,
        encoder_name: str = "resnet50",
        encoder_weights: Optional[str] = "imagenet",
        classes: int = 1,
        activation: Optional[str] = None,
    ):
        super().__init__(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=classes,
            fusion_mode="diff",
            activation=activation,
        )


class UNetSiamConc(UNetSiamese):
    """
    U-Net with Siamese encoders and feature concatenation fusion.

    Architecture:
    - Two ResNet-50 encoders with shared weights
    - Concatenation at bottleneck: [f_t1; f_t2] (4096 channels)
    - U-Net decoder with skip connections
    - Binary segmentation output

    The concatenation preserves all information from both time points,
    letting the decoder learn what's relevant for change detection.
    """

    def __init__(
        self,
        encoder_name: str = "resnet50",
        encoder_weights: Optional[str] = "imagenet",
        classes: int = 1,
        activation: Optional[str] = None,
    ):
        super().__init__(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=classes,
            fusion_mode="concat",
            activation=activation,
        )


def create_model(model_name: str, **kwargs) -> nn.Module:
    """
    Factory function to create models by name.

    Args:
        model_name: One of ['early_fusion', 'siam_diff', 'siam_conc']
        **kwargs: Additional arguments passed to model constructor

    Returns:
        Model instance

    Example:
        >>> model = create_model('early_fusion', encoder_weights='imagenet')
        >>> model = create_model('siam_diff', encoder_name='resnet50')
    """
    models = {
        'early_fusion': UNetEarlyFusion,
        'siam_diff': UNetSiamDiff,
        'siam_conc': UNetSiamConc,
    }

    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models.keys())}")

    return models[model_name](**kwargs)


def count_parameters(model: nn.Module) -> dict:
    """
    Count trainable and total parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with 'trainable' and 'total' parameter counts
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
    """Test model implementations."""
    print("Testing baseline models...\n")

    batch_size = 2
    height, width = 512, 512

    # Test Early Fusion
    print("="*60)
    print("U-Net Early Fusion")
    print("="*60)
    model_ef = UNetEarlyFusion(encoder_weights=None)  # No pretrained for testing
    x_concat = torch.randn(batch_size, 6, height, width)
    output_ef = model_ef(x_concat)
    params_ef = count_parameters(model_ef)
    print(f"Input shape: {x_concat.shape}")
    print(f"Output shape: {output_ef.shape}")
    print(f"Parameters: {params_ef['trainable_millions']:.2f}M trainable, "
          f"{params_ef['total_millions']:.2f}M total")
    print()

    # Test SiamDiff
    print("="*60)
    print("U-Net SiamDiff")
    print("="*60)
    model_sd = UNetSiamDiff(encoder_weights=None)
    x1 = torch.randn(batch_size, 3, height, width)
    x2 = torch.randn(batch_size, 3, height, width)
    output_sd = model_sd(x1, x2)
    params_sd = count_parameters(model_sd)
    print(f"Input shape (per image): {x1.shape}")
    print(f"Output shape: {output_sd.shape}")
    print(f"Parameters: {params_sd['trainable_millions']:.2f}M trainable, "
          f"{params_sd['total_millions']:.2f}M total")
    print()

    # Test SiamConc
    print("="*60)
    print("U-Net SiamConc")
    print("="*60)
    model_sc = UNetSiamConc(encoder_weights=None)
    output_sc = model_sc(x1, x2)
    params_sc = count_parameters(model_sc)
    print(f"Input shape (per image): {x1.shape}")
    print(f"Output shape: {output_sc.shape}")
    print(f"Parameters: {params_sc['trainable_millions']:.2f}M trainable, "
          f"{params_sc['total_millions']:.2f}M total")
    print()

    # Test factory function
    print("="*60)
    print("Testing factory function")
    print("="*60)
    for name in ['early_fusion', 'siam_diff', 'siam_conc']:
        model = create_model(name, encoder_weights=None)
        print(f"{name}: {model.name}")

    print("\nAll models tested successfully!")
