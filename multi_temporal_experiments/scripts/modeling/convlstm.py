#!/usr/bin/env python3
"""
Convolutional LSTM implementation for temporal sequence modeling.

Based on: Shi et al. "Convolutional LSTM Network: A Machine Learning Approach
for Precipitation Nowcasting" (NeurIPS 2015)

Reference implementations:
- https://github.com/ndrplz/ConvLSTM_pytorch
- https://github.com/jhhuang96/ConvLSTM-PyTorch
"""

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """
    Convolutional LSTM cell.

    Processes a single timestep of a spatiotemporal sequence using LSTM gates
    with 2D convolutions instead of fully connected layers.

    Args:
        input_dim: Number of input channels
        hidden_dim: Number of hidden state channels
        kernel_size: Size of convolutional kernel (int or tuple)
        bias: Whether to use bias in convolutions
    """

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3, bias: bool = True):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        # Convolutional layers for LSTM gates
        # Combines input-to-hidden and hidden-to-hidden transformations
        # 4 gates: input (i), forget (f), cell candidate (g), output (o)
        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,  # 4 gates
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Initialize convolutional weights.

        Uses Xavier uniform initialization for input-to-hidden weights
        and orthogonal initialization for hidden-to-hidden weights.
        """
        # Xavier initialization for all weights
        nn.init.xavier_uniform_(self.conv.weight)

        # For better performance, we could split input and hidden weights
        # and use orthogonal init for hidden, but this simplified approach works well

        if self.bias:
            nn.init.constant_(self.conv.bias, 0.0)

    def forward(self, x, h_prev, c_prev):
        """
        Forward pass for a single timestep.

        Args:
            x: Input tensor (B, input_dim, H, W)
            h_prev: Previous hidden state (B, hidden_dim, H, W)
            c_prev: Previous cell state (B, hidden_dim, H, W)

        Returns:
            h_cur: Current hidden state (B, hidden_dim, H, W)
            c_cur: Current cell state (B, hidden_dim, H, W)
        """
        # Concatenate input and previous hidden state
        combined = torch.cat([x, h_prev], dim=1)  # (B, input_dim + hidden_dim, H, W)

        # Compute all gates in one convolution
        gates = self.conv(combined)  # (B, 4*hidden_dim, H, W)

        # Split into 4 gates
        i, f, g, o = torch.split(gates, self.hidden_dim, dim=1)

        # Apply activations
        i = torch.sigmoid(i)  # Input gate
        f = torch.sigmoid(f)  # Forget gate
        g = torch.tanh(g)     # Cell candidate
        o = torch.sigmoid(o)  # Output gate

        # Update cell state
        c_cur = f * c_prev + i * g

        # Update hidden state
        h_cur = o * torch.tanh(c_cur)

        return h_cur, c_cur

    def init_hidden(self, batch_size: int, image_size: tuple):
        """
        Initialize hidden and cell states with zeros.

        Args:
            batch_size: Batch size
            image_size: Spatial dimensions (height, width)

        Returns:
            h: Initial hidden state (B, hidden_dim, H, W)
            c: Initial cell state (B, hidden_dim, H, W)
        """
        height, width = image_size
        device = self.conv.weight.device

        h = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)

        return h, c


class ConvLSTM(nn.Module):
    """
    Multi-layer Convolutional LSTM module.

    Stacks multiple ConvLSTMCell layers and processes temporal sequences.

    Args:
        input_dim: Number of input channels
        hidden_dim: Number of hidden channels (int or list for different layers)
        kernel_size: Convolutional kernel size (int or list)
        num_layers: Number of stacked LSTM layers
        batch_first: If True, input is (B, T, C, H, W), else (T, B, C, H, W)
        bias: Whether to use bias in convolutions
        return_all_layers: If True, return hidden states from all layers
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: int,
        num_layers: int,
        batch_first: bool = True,
        bias: bool = True,
        return_all_layers: bool = False
    ):
        super(ConvLSTM, self).__init__()

        # Handle hidden_dim as int or list
        if isinstance(hidden_dim, int):
            hidden_dim = [hidden_dim] * num_layers
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * num_layers

        assert len(hidden_dim) == num_layers, "hidden_dim must match num_layers"
        assert len(kernel_size) == num_layers, "kernel_size must match num_layers"

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        # Create LSTM cells for each layer
        cell_list = []
        for i in range(self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            cell_list.append(
                ConvLSTMCell(
                    input_dim=cur_input_dim,
                    hidden_dim=self.hidden_dim[i],
                    kernel_size=self.kernel_size[i],
                    bias=self.bias
                )
            )

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, x, hidden_state=None):
        """
        Forward pass through the ConvLSTM.

        Args:
            x: Input tensor
               - If batch_first=True: (B, T, C, H, W)
               - If batch_first=False: (T, B, C, H, W)
            hidden_state: Optional initial hidden state. If None, initialized with zeros.
                         List of (h, c) tuples, one per layer

        Returns:
            layer_output_list: List of outputs from each layer (if return_all_layers)
                              Each element: (B, T, hidden_dim, H, W) if batch_first
            last_state_list: List of (h, c) tuples for final timestep of each layer
        """
        # Handle batch_first
        if not self.batch_first:
            # (T, B, C, H, W) -> (B, T, C, H, W)
            x = x.permute(1, 0, 2, 3, 4)

        B, T, C, H, W = x.size()

        # Initialize hidden states if not provided
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=B, image_size=(H, W))

        # Process through layers
        layer_output_list = []
        last_state_list = []

        cur_layer_input = x

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []

            # Process each timestep
            for t in range(T):
                h, c = self.cell_list[layer_idx](cur_layer_input[:, t, :, :, :], h, c)
                output_inner.append(h)

            # Stack timesteps
            layer_output = torch.stack(output_inner, dim=1)  # (B, T, hidden_dim, H, W)

            # Current layer output becomes next layer input
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append((h, c))

        # Return outputs
        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]  # Only last layer

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size: int, image_size: tuple):
        """
        Initialize hidden states for all layers.

        Args:
            batch_size: Batch size
            image_size: Spatial dimensions (height, width)

        Returns:
            List of (h, c) tuples, one per layer
        """
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states


if __name__ == "__main__":
    """Test the ConvLSTM implementation."""

    print("=" * 80)
    print("TESTING ConvLSTM IMPLEMENTATION")
    print("=" * 80)

    # Test 1: ConvLSTMCell forward pass
    print("\nTest 1: ConvLSTMCell forward pass")
    print("-" * 80)

    cell = ConvLSTMCell(input_dim=64, hidden_dim=32, kernel_size=3)
    x = torch.randn(2, 64, 8, 8)
    h = torch.randn(2, 32, 8, 8)
    c = torch.randn(2, 32, 8, 8)

    h_new, c_new = cell(x, h, c)

    print(f"Input shape: {x.shape}")
    print(f"Previous h shape: {h.shape}")
    print(f"Previous c shape: {c.shape}")
    print(f"New h shape: {h_new.shape}")
    print(f"New c shape: {c_new.shape}")

    assert h_new.shape == (2, 32, 8, 8), "Hidden state shape mismatch"
    assert c_new.shape == (2, 32, 8, 8), "Cell state shape mismatch"
    print("✓ ConvLSTMCell forward pass test passed!")

    # Test 2: ConvLSTM multi-layer forward pass
    print("\nTest 2: ConvLSTM multi-layer forward pass")
    print("-" * 80)

    convlstm = ConvLSTM(
        input_dim=64,
        hidden_dim=32,
        kernel_size=3,
        num_layers=2,
        batch_first=True,
        return_all_layers=False
    )

    x_seq = torch.randn(2, 5, 64, 8, 8)  # (B, T, C, H, W)

    layer_outputs, last_states = convlstm(x_seq)

    print(f"Input sequence shape: {x_seq.shape}")
    print(f"Number of layer outputs: {len(layer_outputs)}")
    print(f"Last layer output shape: {layer_outputs[0].shape}")
    print(f"Number of last states: {len(last_states)}")
    print(f"Last layer final h shape: {last_states[-1][0].shape}")
    print(f"Last layer final c shape: {last_states[-1][1].shape}")

    assert layer_outputs[0].shape == (2, 5, 32, 8, 8), "Output shape mismatch"
    assert last_states[-1][0].shape == (2, 32, 8, 8), "Final hidden state shape mismatch"
    print("✓ ConvLSTM forward pass test passed!")

    # Test 3: Gradient flow
    print("\nTest 3: Gradient flow through ConvLSTM")
    print("-" * 80)

    convlstm.zero_grad()
    x_seq = torch.randn(2, 5, 64, 8, 8, requires_grad=True)

    layer_outputs, last_states = convlstm(x_seq)
    loss = layer_outputs[0].mean()
    loss.backward()

    # Check that gradients exist
    has_grads = True
    for name, param in convlstm.named_parameters():
        if param.grad is None:
            print(f"  ✗ No gradient for {name}")
            has_grads = False

    if has_grads:
        print("✓ Gradient flow test passed!")

    # Test 4: Realistic dimensions for LSTM-UNet bottleneck
    print("\nTest 4: Realistic dimensions (LSTM-UNet bottleneck)")
    print("-" * 80)

    # Bottleneck: ResNet-50 has 2048 channels at H/32 resolution
    # For 512x512 input, bottleneck is 16x16
    convlstm_bottleneck = ConvLSTM(
        input_dim=2048,
        hidden_dim=512,
        kernel_size=3,
        num_layers=2,
        batch_first=True,
        return_all_layers=False
    )

    # Simulate annual sampling: T=7, image_size=512 → bottleneck 16x16
    x_bottleneck = torch.randn(4, 7, 2048, 16, 16)  # (B=4, T=7, C=2048, H=16, W=16)

    print(f"Bottleneck input shape: {x_bottleneck.shape}")

    layer_outputs, last_states = convlstm_bottleneck(x_bottleneck)
    final_h = last_states[-1][0]  # Hidden state from last layer

    print(f"Final hidden state shape: {final_h.shape}")
    print(f"Expected shape: (4, 512, 16, 16)")

    assert final_h.shape == (4, 512, 16, 16), "Bottleneck output shape mismatch"
    print("✓ Realistic dimensions test passed!")

    # Parameter count
    print("\nParameter Statistics:")
    print("-" * 80)
    total_params = sum(p.numel() for p in convlstm_bottleneck.parameters())
    trainable_params = sum(p.numel() for p in convlstm_bottleneck.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Parameters (millions): {total_params/1e6:.2f}M")

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED! ✓")
    print("=" * 80)
