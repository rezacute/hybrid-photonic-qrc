"""
Photonic Delay-Loop Emulation Layer

This module implements the Photonic Delay-Loop Emulation (PDEL) layer that
emulates multiple temporal scales of photonic delay loops using parallel
1D convolution banks.
"""


import torch
import torch.nn as nn


class PhotonicDelayBank(nn.Module):
    """Single Conv1D bank emulating one photonic delay loop.
    
    Uses causal padding to ensure only past information is used.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        activation: str = "tanh",
        frozen: bool = False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.activation = activation
        self.frozen = frozen

        # Conv1D with causal padding
        # padding = kernel_size - 1 ensures output is same length as input
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=kernel_size - 1,  # Causal padding
            bias=False,
        )

        # Activation function
        if activation == "tanh":
            self.activation_fn = nn.Tanh()
        elif activation == "relu":
            self.activation_fn = nn.ReLU()
        elif activation == "identity":
            self.activation_fn = nn.Identity()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Freeze kernels if specified
        if frozen:
            for param in self.conv.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, channels, seq_len)
        Returns:
            Output tensor of shape (batch, out_channels, seq_len)
        """
        # Apply convolution with causal padding
        out = self.conv(x)

        # Trim output to match input length (causal property)
        out = out[:, :, :x.shape[2]]

        # Apply activation
        out = self.activation_fn(out)

        return out

    @property
    def n_params(self) -> int:
        """Number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PhotonicDelayLoopEmulator(nn.Module):
    """Multi-bank PDEL: K parallel Conv1D banks at different temporal scales.
    
    Each bank emulates a photonic delay loop at a different temporal scale
    (e.g., 15min, 1h, 6h, 24h, 1 week).
    """

    def __init__(
        self,
        in_channels: int,
        n_banks: int = 5,
        kernel_sizes: list[int] = None,
        features_per_bank: int = 16,
        activation: str = "tanh",
        frozen_kernels: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.n_banks = n_banks
        self.kernel_sizes = kernel_sizes or [4, 24, 96, 168, 672]
        self.features_per_bank = features_per_bank
        self.activation = activation
        self.frozen_kernels = frozen_kernels
        self.dropout = dropout

        # Create parallel delay banks
        self.banks = nn.ModuleList([
            PhotonicDelayBank(
                in_channels=in_channels,
                out_channels=features_per_bank,
                kernel_size=ks,
                activation=activation,
                frozen=frozen_kernels,
            )
            for ks in self.kernel_sizes
        ])

        # Optional dropout
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Freeze banks if specified
        if frozen_kernels:
            for bank in self.banks:
                for param in bank.parameters():
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, channels, seq_len)
        Returns:
            Output tensor of shape (batch, K*D_phot, seq_len)
        """
        outputs = []

        for bank in self.banks:
            out = bank(x)  # (batch, features_per_bank, seq_len)
            outputs.append(out)

        # Concatenate along channel dimension
        result = torch.cat(outputs, dim=1)  # (batch, K*D_phot, seq_len)

        # Apply dropout
        result = self.dropout_layer(result)

        return result

    @property
    def output_dim(self) -> int:
        """Total output feature dimension."""
        return self.n_banks * self.features_per_bank

    @property
    def n_trainable_params(self) -> int:
        """Number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze(self):
        """Freeze all bank parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Unfreeze all bank parameters."""
        for param in self.parameters():
            param.requires_grad = True
