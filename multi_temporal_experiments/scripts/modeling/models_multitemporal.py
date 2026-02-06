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


class EarlyFusionUNet(nn.Module):
    """
    Early-fusion U-Net for bi-temporal land-take detection.

    Architecture:
    - Stack 2018+2024 images as 18 channels (9 bands × 2 timesteps)
    - Standard U-Net with ResNet encoder
    - No temporal modeling (channels treated as extra features)

    This serves as a baseline to test: "Do we need temporal modeling at all?"

    Args:
        encoder_name: Encoder backbone (default: resnet50)
        encoder_weights: Pre-trained weights (default: imagenet)
        in_channels: Input bands per timestep (default: 9 for Sentinel-2)
        classes: Output classes (default: 1 for binary)
        activation: Output activation (default: None)
    """

    def __init__(
        self,
        encoder_name: str = "resnet50",
        encoder_weights: Optional[str] = "imagenet",
        in_channels: int = 9,
        classes: int = 1,
        activation: Optional[str] = None,
        **kwargs,  # Accept but ignore LSTM-specific params
    ):
        super().__init__()

        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.in_channels = in_channels
        self.classes = classes

        # Number of stacked channels (2 timesteps × bands)
        stacked_channels = 2 * in_channels  # 18 for bi-temporal

        # Create standard U-Net with stacked input channels
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=stacked_channels,
            classes=classes,
            activation=activation,
        )

        self.name = f"EarlyFusionUNet_{encoder_name}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (B, T=2, C=9, H, W) - Bi-temporal input

        Returns:
            output: (B, 1, H, W) - Binary segmentation logits
        """
        B, T, C, H, W = x.shape

        # Stack timesteps as channels: (B, T=2, C=9, H, W) → (B, 18, H, W)
        x_stacked = x.view(B, T * C, H, W)

        # Standard U-Net forward pass
        output = self.unet(x_stacked)

        return output


class LateFusionConcat(nn.Module):
    """
    Late-fusion concatenation model for bi-temporal land-take detection.

    Architecture:
    - Shared ResNet encoder processes each timestep
    - Concatenate bottleneck features from both timesteps
    - 1×1 convolution to fuse concatenated features
    - Max aggregation for skip connections (same as LSTM-UNet)
    - Standard U-Net decoder

    This serves as a baseline to test: "Does recurrence help beyond simple multi-view aggregation?"

    Args:
        encoder_name: Encoder backbone (default: resnet50)
        encoder_weights: Pre-trained weights (default: imagenet)
        in_channels: Input bands per timestep (default: 9 for Sentinel-2)
        classes: Output classes (default: 1 for binary)
        skip_aggregation: How to aggregate skip connections ("max", "mean", "last")
        activation: Output activation (default: None)
    """

    def __init__(
        self,
        encoder_name: str = "resnet50",
        encoder_weights: Optional[str] = "imagenet",
        in_channels: int = 9,
        classes: int = 1,
        skip_aggregation: str = "max",
        activation: Optional[str] = None,
        **kwargs,  # Accept but ignore LSTM-specific params
    ):
        super().__init__()

        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.in_channels = in_channels
        self.classes = classes
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

        # 3. Bottleneck fusion: concat → 1×1 conv
        # For ResNet-50, bottleneck is 2048 channels
        bottleneck_channels = self.encoder_channels[-1]  # 2048
        fused_channels = 512  # Same as LSTM-UNet output

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(2 * bottleneck_channels, fused_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(fused_channels),
            nn.ReLU(inplace=True),
        )

        # 4. Create decoder with modified bottleneck channels
        decoder_encoder_channels = self.encoder_channels.copy()
        decoder_encoder_channels[-1] = fused_channels  # Replace 2048 with 512

        self.decoder = UnetDecoder(
            encoder_channels=decoder_encoder_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            n_blocks=5,
            attention_type=None,
        )

        # 5. Segmentation head
        self.segmentation_head = template.segmentation_head

        self.name = f"LateFusionConcat_{encoder_name}"

    def _modify_first_conv_for_multispectral(self, in_channels: int, encoder_name: str):
        """Modify first conv layer to accept multi-spectral input (same as LSTMUNet)."""
        if "resnet" in encoder_name or "resnext" in encoder_name:
            first_conv = self.encoder.conv1
        elif "efficientnet" in encoder_name:
            first_conv = self.encoder.conv_stem
        elif "mobilenet" in encoder_name:
            first_conv = self.encoder.features[0][0]
        else:
            first_conv = None
            for module in self.encoder.modules():
                if isinstance(module, nn.Conv2d):
                    first_conv = module
                    break

        if first_conv is None:
            raise RuntimeError(f"Could not find first conv layer in {encoder_name}")

        old_weight = first_conv.weight.data
        out_channels = old_weight.shape[0]
        kernel_size = old_weight.shape[2:]
        device = old_weight.device

        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None,
        ).to(device)

        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = old_weight
            if in_channels > 3:
                scale = (3.0 / in_channels) ** 0.5
                new_conv.weight[:, 3:, :, :] = (
                    torch.randn_like(new_conv.weight[:, 3:, :, :]) * scale
                )
            if first_conv.bias is not None:
                new_conv.bias.data = first_conv.bias.data

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
        """Aggregate temporal features for skip connections (same as LSTMUNet)."""
        if method == "max":
            return features_temporal.max(dim=1)[0]
        elif method == "mean":
            return features_temporal.mean(dim=1)
        elif method == "last":
            return features_temporal[:, -1]
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (B, T=2, C=9, H, W) - Bi-temporal input

        Returns:
            output: (B, 1, H, W) - Binary segmentation logits
        """
        B, T, C, H, W = x.shape

        # 1. Extract features for each timestep with shared encoder
        x_reshaped = x.view(B * T, C, H, W)
        features_reshaped = self.encoder(x_reshaped)

        # 2. Reshape back to temporal format
        features_temporal = []
        for feat in features_reshaped:
            _, F, H_f, W_f = feat.shape
            feat_temporal = feat.view(B, T, F, H_f, W_f)
            features_temporal.append(feat_temporal)

        # 3. Concatenate bottleneck features and fuse with 1×1 conv
        bottleneck_temporal = features_temporal[-1]  # (B, T=2, 2048, H/32, W/32)

        # Concat along channel dim: (B, 2*2048, H/32, W/32)
        bottleneck_concat = bottleneck_temporal.view(B, T * bottleneck_temporal.shape[2],
                                                      bottleneck_temporal.shape[3],
                                                      bottleneck_temporal.shape[4])

        # Fuse with 1×1 conv: (B, 512, H/32, W/32)
        bottleneck_fused = self.fusion_conv(bottleneck_concat)

        # 4. Aggregate skip connection features (stages 0-4, excluding bottleneck)
        skip_features = []
        for feat_temporal in features_temporal[:-1]:
            feat_spatial = self._aggregate_temporal_features(
                feat_temporal,
                method=self.skip_aggregation
            )
            skip_features.append(feat_spatial)

        # Add fused bottleneck as last skip connection
        skip_features.append(bottleneck_fused)

        # 5. Decode with U-Net decoder
        decoder_output = self.decoder(skip_features)

        # 6. Segmentation head
        output = self.segmentation_head(decoder_output)

        return output


class LateFusionPool(nn.Module):
    """
    Late-fusion temporal pooling model for multi-temporal land-take detection.

    Architecture:
    - Shared ResNet encoder processes each timestep
    - Mean pool bottleneck features over T → 1×1 conv (2048 → 512) + BN + ReLU
    - Max aggregation for skip connections (same as LSTM-UNet)
    - Standard U-Net decoder

    T-agnostic: works with any number of timesteps.

    Args:
        encoder_name: Encoder backbone (default: resnet50)
        encoder_weights: Pre-trained weights (default: imagenet)
        in_channels: Input bands per timestep (default: 9 for Sentinel-2)
        classes: Output classes (default: 1 for binary)
        skip_aggregation: How to aggregate skip connections ("max", "mean", "last")
        activation: Output activation (default: None)
    """

    def __init__(
        self,
        encoder_name: str = "resnet50",
        encoder_weights: Optional[str] = "imagenet",
        in_channels: int = 9,
        classes: int = 1,
        skip_aggregation: str = "max",
        activation: Optional[str] = None,
        **kwargs,  # Accept but ignore LSTM-specific params
    ):
        super().__init__()

        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.in_channels = in_channels
        self.classes = classes
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

        if in_channels != 3:
            self._modify_first_conv_for_multispectral(in_channels, encoder_name)

        # 3. Bottleneck fusion: mean pool over T → 1×1 conv
        bottleneck_channels = self.encoder_channels[-1]  # 2048
        fused_channels = 512

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(bottleneck_channels, fused_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(fused_channels),
            nn.ReLU(inplace=True),
        )

        # 4. Create decoder with modified bottleneck channels
        decoder_encoder_channels = self.encoder_channels.copy()
        decoder_encoder_channels[-1] = fused_channels

        self.decoder = UnetDecoder(
            encoder_channels=decoder_encoder_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            n_blocks=5,
            attention_type=None,
        )

        # 5. Segmentation head
        self.segmentation_head = template.segmentation_head

        self.name = f"LateFusionPool_{encoder_name}"

    def _modify_first_conv_for_multispectral(self, in_channels: int, encoder_name: str):
        """Modify first conv layer to accept multi-spectral input (same as LSTMUNet)."""
        if "resnet" in encoder_name or "resnext" in encoder_name:
            first_conv = self.encoder.conv1
        elif "efficientnet" in encoder_name:
            first_conv = self.encoder.conv_stem
        elif "mobilenet" in encoder_name:
            first_conv = self.encoder.features[0][0]
        else:
            first_conv = None
            for module in self.encoder.modules():
                if isinstance(module, nn.Conv2d):
                    first_conv = module
                    break

        if first_conv is None:
            raise RuntimeError(f"Could not find first conv layer in {encoder_name}")

        old_weight = first_conv.weight.data
        out_channels = old_weight.shape[0]
        kernel_size = old_weight.shape[2:]
        device = old_weight.device

        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None,
        ).to(device)

        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = old_weight
            if in_channels > 3:
                scale = (3.0 / in_channels) ** 0.5
                new_conv.weight[:, 3:, :, :] = (
                    torch.randn_like(new_conv.weight[:, 3:, :, :]) * scale
                )
            if first_conv.bias is not None:
                new_conv.bias.data = first_conv.bias.data

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
        """Aggregate temporal features for skip connections."""
        if method == "max":
            return features_temporal.max(dim=1)[0]
        elif method == "mean":
            return features_temporal.mean(dim=1)
        elif method == "last":
            return features_temporal[:, -1]
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (B, T, C, H, W) - Multi-temporal input (any T)

        Returns:
            output: (B, 1, H, W) - Binary segmentation logits
        """
        B, T, C, H, W = x.shape

        # 1. Extract features for each timestep with shared encoder
        x_reshaped = x.view(B * T, C, H, W)
        features_reshaped = self.encoder(x_reshaped)

        # 2. Reshape back to temporal format
        features_temporal = []
        for feat in features_reshaped:
            _, F, H_f, W_f = feat.shape
            feat_temporal = feat.view(B, T, F, H_f, W_f)
            features_temporal.append(feat_temporal)

        # 3. Mean pool bottleneck over T → 1×1 conv
        bottleneck_temporal = features_temporal[-1]  # (B, T, 2048, H/32, W/32)
        bottleneck_pooled = bottleneck_temporal.mean(dim=1)  # (B, 2048, H/32, W/32)
        bottleneck_fused = self.fusion_conv(bottleneck_pooled)  # (B, 512, H/32, W/32)

        # 4. Aggregate skip connection features
        skip_features = []
        for feat_temporal in features_temporal[:-1]:
            feat_spatial = self._aggregate_temporal_features(
                feat_temporal,
                method=self.skip_aggregation
            )
            skip_features.append(feat_spatial)

        skip_features.append(bottleneck_fused)

        # 5. Decode
        decoder_output = self.decoder(skip_features)

        # 6. Segmentation head
        output = self.segmentation_head(decoder_output)

        return output


class Conv3DFusion(nn.Module):
    """
    3D convolutional fusion model for multi-temporal land-take detection.

    Architecture:
    - Shared ResNet encoder processes each timestep
    - Bottleneck: rearrange to (B, 2048, T, H, W) →
        Conv3d(2048, 512, k=(3,1,1)) + BN3d + ReLU →
        Conv3d(512, 512, k=(3,1,1)) + BN3d + ReLU →
        mean pool over T → (B, 512, H, W)
    - Max aggregation for skip connections (same as LSTM-UNet)
    - Standard U-Net decoder

    Uses temporal-only 3D conv kernels (3,1,1) — spatial modelling is done by encoder.
    T-agnostic: padding preserves T, then mean pool collapses it.

    Args:
        encoder_name: Encoder backbone (default: resnet50)
        encoder_weights: Pre-trained weights (default: imagenet)
        in_channels: Input bands per timestep (default: 9 for Sentinel-2)
        classes: Output classes (default: 1 for binary)
        skip_aggregation: How to aggregate skip connections ("max", "mean", "last")
        activation: Output activation (default: None)
    """

    def __init__(
        self,
        encoder_name: str = "resnet50",
        encoder_weights: Optional[str] = "imagenet",
        in_channels: int = 9,
        classes: int = 1,
        skip_aggregation: str = "max",
        activation: Optional[str] = None,
        **kwargs,  # Accept but ignore LSTM-specific params
    ):
        super().__init__()

        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.in_channels = in_channels
        self.classes = classes
        self.skip_aggregation = skip_aggregation

        # 1. Create template U-Net to extract encoder architecture
        template = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=classes,
            activation=activation,
        )

        # 2. Extract encoder and modify first conv for multi-spectral input
        self.encoder = template.encoder
        self.encoder_channels = list(template.encoder.out_channels)

        if in_channels != 3:
            self._modify_first_conv_for_multispectral(in_channels, encoder_name)

        # 3. Bottleneck fusion: 3D conv (temporal-only kernels) + mean pool
        bottleneck_channels = self.encoder_channels[-1]  # 2048
        fused_channels = 512

        self.temporal_conv3d = nn.Sequential(
            nn.Conv3d(bottleneck_channels, fused_channels,
                      kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False),
            nn.BatchNorm3d(fused_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(fused_channels, fused_channels,
                      kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False),
            nn.BatchNorm3d(fused_channels),
            nn.ReLU(inplace=True),
        )

        # 4. Create decoder with modified bottleneck channels
        decoder_encoder_channels = self.encoder_channels.copy()
        decoder_encoder_channels[-1] = fused_channels

        self.decoder = UnetDecoder(
            encoder_channels=decoder_encoder_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            n_blocks=5,
            attention_type=None,
        )

        # 5. Segmentation head
        self.segmentation_head = template.segmentation_head

        self.name = f"Conv3DFusion_{encoder_name}"

    def _modify_first_conv_for_multispectral(self, in_channels: int, encoder_name: str):
        """Modify first conv layer to accept multi-spectral input (same as LSTMUNet)."""
        if "resnet" in encoder_name or "resnext" in encoder_name:
            first_conv = self.encoder.conv1
        elif "efficientnet" in encoder_name:
            first_conv = self.encoder.conv_stem
        elif "mobilenet" in encoder_name:
            first_conv = self.encoder.features[0][0]
        else:
            first_conv = None
            for module in self.encoder.modules():
                if isinstance(module, nn.Conv2d):
                    first_conv = module
                    break

        if first_conv is None:
            raise RuntimeError(f"Could not find first conv layer in {encoder_name}")

        old_weight = first_conv.weight.data
        out_channels = old_weight.shape[0]
        kernel_size = old_weight.shape[2:]
        device = old_weight.device

        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None,
        ).to(device)

        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = old_weight
            if in_channels > 3:
                scale = (3.0 / in_channels) ** 0.5
                new_conv.weight[:, 3:, :, :] = (
                    torch.randn_like(new_conv.weight[:, 3:, :, :]) * scale
                )
            if first_conv.bias is not None:
                new_conv.bias.data = first_conv.bias.data

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
        """Aggregate temporal features for skip connections."""
        if method == "max":
            return features_temporal.max(dim=1)[0]
        elif method == "mean":
            return features_temporal.mean(dim=1)
        elif method == "last":
            return features_temporal[:, -1]
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (B, T, C, H, W) - Multi-temporal input (any T)

        Returns:
            output: (B, 1, H, W) - Binary segmentation logits
        """
        B, T, C, H, W = x.shape

        # 1. Extract features for each timestep with shared encoder
        x_reshaped = x.view(B * T, C, H, W)
        features_reshaped = self.encoder(x_reshaped)

        # 2. Reshape back to temporal format
        features_temporal = []
        for feat in features_reshaped:
            _, F, H_f, W_f = feat.shape
            feat_temporal = feat.view(B, T, F, H_f, W_f)
            features_temporal.append(feat_temporal)

        # 3. 3D conv bottleneck fusion
        bottleneck_temporal = features_temporal[-1]  # (B, T, 2048, H/32, W/32)
        # Rearrange to (B, C, T, H, W) for Conv3d
        bottleneck_3d = bottleneck_temporal.permute(0, 2, 1, 3, 4)  # (B, 2048, T, H/32, W/32)
        # Apply temporal 3D convolutions
        bottleneck_3d = self.temporal_conv3d(bottleneck_3d)  # (B, 512, T, H/32, W/32)
        # Mean pool over T
        bottleneck_fused = bottleneck_3d.mean(dim=2)  # (B, 512, H/32, W/32)

        # 4. Aggregate skip connection features
        skip_features = []
        for feat_temporal in features_temporal[:-1]:
            feat_spatial = self._aggregate_temporal_features(
                feat_temporal,
                method=self.skip_aggregation
            )
            skip_features.append(feat_spatial)

        skip_features.append(bottleneck_fused)

        # 5. Decode
        decoder_output = self.decoder(skip_features)

        # 6. Segmentation head
        output = self.segmentation_head(decoder_output)

        return output


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
        convlstm_kernel_size: ConvLSTM spatial kernel size (default: 3).
            Use 1 for per-pixel temporal modeling, 3 for patch-based.
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
        convlstm_kernel_size: int = 3,
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
        self.convlstm_kernel_size = convlstm_kernel_size
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
            kernel_size=convlstm_kernel_size,
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

        self.name = f"LSTMUNet_{encoder_name}_k{convlstm_kernel_size}"

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
        model_name: One of ['lstm_unet', 'early_fusion_unet', 'late_fusion_concat',
                    'late_fusion_pool', 'conv3d_fusion']
        **kwargs: Model-specific arguments

    Returns:
        Model instance

    Raises:
        ValueError: If model_name is not recognized
    """
    models = {
        'lstm_unet': LSTMUNet,
        'early_fusion_unet': EarlyFusionUNet,
        'late_fusion_concat': LateFusionConcat,
        'late_fusion_pool': LateFusionPool,
        'conv3d_fusion': Conv3DFusion,
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

    # Test 7: ConvLSTM kernel size variations (1D vs 2D)
    print("\nTest 7: ConvLSTM kernel size variations")
    print("-" * 80)

    kernel_sizes = [1, 3]

    for ks in kernel_sizes:
        model_ks = LSTMUNet(
            encoder_weights=None,
            in_channels=9,
            convlstm_kernel_size=ks,
        )

        x = torch.randn(2, 7, 9, 64, 64)
        output = model_ks(x)

        assert output.shape == (2, 1, 64, 64), f"Output shape mismatch for kernel_size={ks}"
        print(f"  ✓ kernel_size={ks} ({ks}x{ks}) works! Model name: {model_ks.name}")

    # Test 8: Early-Fusion U-Net (baseline)
    print("\nTest 8: Early-Fusion U-Net (no temporal modeling)")
    print("-" * 80)

    early_fusion = EarlyFusionUNet(
        encoder_name="resnet50",
        encoder_weights=None,
        in_channels=9,
        classes=1,
    )

    print(f"✓ Model created: {early_fusion.name}")

    # Test forward pass with bi-temporal input
    x = torch.randn(4, 2, 9, 64, 64)  # (B=4, T=2, C=9, H=64, W=64)
    output = early_fusion(x)

    assert output.shape == (4, 1, 64, 64), f"Output shape mismatch: {output.shape}"
    print(f"  Input: {tuple(x.shape)}")
    print(f"  Output: {tuple(output.shape)}")
    print(f"  ✓ Forward pass works!")

    ef_params = count_parameters(early_fusion)
    print(f"  Parameters: {ef_params['total_millions']:.2f}M")

    # Test 9: Late-Fusion Concat (baseline)
    print("\nTest 9: Late-Fusion Concat (shared encoder + concat)")
    print("-" * 80)

    late_fusion = LateFusionConcat(
        encoder_name="resnet50",
        encoder_weights=None,
        in_channels=9,
        classes=1,
        skip_aggregation="max",
    )

    print(f"✓ Model created: {late_fusion.name}")

    # Test forward pass with bi-temporal input
    x = torch.randn(4, 2, 9, 64, 64)  # (B=4, T=2, C=9, H=64, W=64)
    output = late_fusion(x)

    assert output.shape == (4, 1, 64, 64), f"Output shape mismatch: {output.shape}"
    print(f"  Input: {tuple(x.shape)}")
    print(f"  Output: {tuple(output.shape)}")
    print(f"  ✓ Forward pass works!")

    lf_params = count_parameters(late_fusion)
    print(f"  Parameters: {lf_params['total_millions']:.2f}M")

    # Test 10: Late-Fusion Pool (T-agnostic)
    print("\nTest 10: Late-Fusion Pool (mean pool bottleneck)")
    print("-" * 80)

    pool_model = LateFusionPool(
        encoder_name="resnet50",
        encoder_weights=None,
        in_channels=9,
        classes=1,
        skip_aggregation="max",
    )

    print(f"✓ Model created: {pool_model.name}")

    x = torch.randn(2, 7, 9, 64, 64)  # T=7 input
    output = pool_model(x)
    assert output.shape == (2, 1, 64, 64), f"Output shape mismatch: {output.shape}"
    print(f"  Input: {tuple(x.shape)}")
    print(f"  Output: {tuple(output.shape)}")
    print(f"  ✓ Forward pass works with T=7!")

    pool_params = count_parameters(pool_model)
    print(f"  Parameters: {pool_params['total_millions']:.2f}M")

    # Test 11: Conv3D Fusion (T-agnostic)
    print("\nTest 11: Conv3D Fusion (temporal 3D conv bottleneck)")
    print("-" * 80)

    conv3d_model = Conv3DFusion(
        encoder_name="resnet50",
        encoder_weights=None,
        in_channels=9,
        classes=1,
        skip_aggregation="max",
    )

    print(f"✓ Model created: {conv3d_model.name}")

    x = torch.randn(2, 7, 9, 64, 64)  # T=7 input
    output = conv3d_model(x)
    assert output.shape == (2, 1, 64, 64), f"Output shape mismatch: {output.shape}"
    print(f"  Input: {tuple(x.shape)}")
    print(f"  Output: {tuple(output.shape)}")
    print(f"  ✓ Forward pass works with T=7!")

    conv3d_params = count_parameters(conv3d_model)
    print(f"  Parameters: {conv3d_params['total_millions']:.2f}M")

    # Test 12: ConvLSTM-lite (hidden_dim=32, num_layers=1) for exp009
    print("\nTest 12: ConvLSTM-lite (hidden_dim=32, num_layers=1)")
    print("-" * 80)

    lite_model = LSTMUNet(
        encoder_name="resnet50",
        encoder_weights=None,
        in_channels=9,
        classes=1,
        lstm_hidden_dim=32,
        lstm_num_layers=1,
        convlstm_kernel_size=3,
        skip_aggregation="max",
    )

    print(f"✓ Model created: {lite_model.name}")

    x = torch.randn(4, 2, 9, 64, 64)  # T=2 input
    output = lite_model(x)
    assert output.shape == (4, 1, 64, 64), f"Output shape mismatch: {output.shape}"
    print(f"  Input: {tuple(x.shape)}")
    print(f"  Output: {tuple(output.shape)}")
    print(f"  ✓ Forward pass works with T=2!")

    lite_params = count_parameters(lite_model)
    print(f"  Parameters: {lite_params['total_millions']:.2f}M")

    # Test 13: Factory function for all models
    print("\nTest 13: Factory function for all models")
    print("-" * 80)

    for model_name in ['lstm_unet', 'early_fusion_unet', 'late_fusion_concat',
                        'late_fusion_pool', 'conv3d_fusion']:
        model_test = create_multitemporal_model(
            model_name,
            encoder_name='resnet50',
            encoder_weights=None,
            in_channels=9,
        )
        x = torch.randn(2, 2, 9, 64, 64)
        output = model_test(x)
        assert output.shape == (2, 1, 64, 64), f"Output shape mismatch for {model_name}"
        print(f"  ✓ {model_name}: works!")

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED! ✓")
    print("=" * 80)

    # Summary
    print("\nModel Summary:")
    print("-" * 80)
    print(f"1. LSTM-UNet: {param_stats['total_millions']:.2f}M params")
    print(f"   - Temporal modeling: {model.lstm_num_layers}-layer ConvLSTM at bottleneck")
    print(f"   - ConvLSTM kernel size: {model.convlstm_kernel_size}x{model.convlstm_kernel_size}")
    print(f"   - Skip aggregation: {model.skip_aggregation} pooling over time")
    print(f"\n2. Early-Fusion U-Net: {ef_params['total_millions']:.2f}M params")
    print(f"   - No temporal modeling (channel stacking)")
    print(f"   - Input: 18 channels (9 bands × 2 timesteps)")
    print(f"\n3. Late-Fusion Concat: {lf_params['total_millions']:.2f}M params")
    print(f"   - Shared encoder + concat + 1×1 conv fusion")
    print(f"   - Skip aggregation: max pooling over time")
    print(f"\n4. Late-Fusion Pool: {pool_params['total_millions']:.2f}M params")
    print(f"   - Shared encoder + mean pool + 1×1 conv fusion")
    print(f"   - T-agnostic (works with any number of timesteps)")
    print(f"\n5. Conv3D Fusion: {conv3d_params['total_millions']:.2f}M params")
    print(f"   - Shared encoder + temporal 3D conv (3,1,1) + mean pool")
    print(f"   - T-agnostic (works with any number of timesteps)")
    print(f"\n6. ConvLSTM-lite: {lite_params['total_millions']:.2f}M params")
    print(f"   - 1-layer ConvLSTM with hidden_dim=32 (parameter-matched)")
    print("\nAll models ready for training!")
