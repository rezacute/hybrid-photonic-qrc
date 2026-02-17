"""
Photonic Delay-Loop Emulation Layer

This module implements the Photonic Delay-Loop Emulation (PDEL) layer that
emulates multiple temporal scales of photonic delay loops using parallel
1D convolution banks.
"""

import torch
import torch.nn as nn
from typing import List, Optional
import math


class PhotonicDelayBank(nn.Module):
    """Single Photonic Delay Bank using 1D Convolution.
    
    Emulates a photonic delay loop with a specific temporal kernel size.
    Uses causal padding to ensure only past information is used.
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        kernel_size: int,
        kernel_init: str = "random",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.kernel_size = kernel_size
        
        # 1D Convolution with causal padding
        self.conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=0,  # No padding - causal
            bias=False,
        )
        
        # Initialize kernel
        self._init_kernel(kernel_init)
    
    def _init_kernel(self, init_type: str):
        """Initialize convolution kernels."""
        if init_type == "random":
            nn.init.xavier_normal_(self.conv.weight)
        elif init_type == "chirp":
            # Initialize with chirp (linear frequency sweep)
            for i in range(self.conv.in_channels):
                for j in range(self.conv.out_channels):
                    t = torch.arange(self.kernel_size)
                    freq = 0.1 + 0.4 * t / self.kernel_size
                    phase = 2 * math.pi * freq * t
                    kernel = torch.cos(phase) * torch.exp(-t / (self.kernel_size / 3))
                    self.conv.weight[j, i, :] = kernel
        elif init_type == "gabor":
            # Initialize with Gabor (Gaussian modulated sinusoid)
            for i in range(self.conv.in_channels):
                for j in range(self.conv.out_channels):
                    t = torch.arange(self.kernel_size) - self.kernel_size / 2
                    sigma = self.kernel_size / 6
                    freq = 0.2 + 0.1 * j
                    gaussian = torch.exp(-t**2 / (2 * sigma**2))
                    gabor = gaussian * torch.cos(2 * math.pi * freq * t)
                    self.conv.weight[j, i, :] = gabor
        else:
            nn.init.xavier_normal_(self.conv.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
        Returns:
            Output tensor of shape (batch, seq_len - kernel_size + 1, d_model)
        """
        # Transpose for Conv1d: (batch, channels, seq)
        x = x.transpose(1, 2)
        
        # Apply convolution (causal - output shorter than input)
        out = self.conv(x)
        
        # Transpose back: (batch, seq, channels)
        out = out.transpose(1, 2)
        
        return out


class PhotonicDelayLoopEmulator(nn.Module):
    """Multiple Photonic Delay Banks with different temporal scales.
    
    This module emulates photonic delay loops at multiple temporal scales
    (sub-hourly, hourly, daily, weekly) using parallel Conv1D banks.
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        kernel_sizes: List[int],
        kernel_init: str = "random",
        use_batch_norm: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.kernel_sizes = kernel_sizes
        self.n_banks = len(kernel_sizes)
        
        # Create parallel delay banks
        self.banks = nn.ModuleList([
            PhotonicDelayBank(
                input_dim=input_dim,
                d_model=d_model,
                kernel_size=ks,
                kernel_init=kernel_init,
            )
            for ks in kernel_sizes
        ])
        
        # Optional batch normalization
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(d_model) if use_batch_norm else nn.Identity()
            for _ in kernel_sizes
        ])
        
        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Output dimension
        self.output_dim = d_model * len(kernel_sizes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
        Returns:
            Concatenated output of shape (batch, seq_len, output_dim)
        """
        outputs = []
        
        for bank, bn in zip(self.banks, self.batch_norms):
            out = bank(x)
            
            # Apply batch norm and dropout
            out = out.transpose(1, 2)  # (batch, d_model, seq)
            out = bn(out)
            out = out.transpose(1, 2)  # (batch, seq, d_model)
            out = self.dropout(out)
            
            outputs.append(out)
        
        # Handle different sequence lengths due to causal convolution
        min_len = min(o.shape[1] for o in outputs)
        outputs = [o[:, :min_len, :] for o in outputs]
        
        # Concatenate along feature dimension
        result = torch.cat(outputs, dim=-1)
        
        return result
    
    def get_output_dim(self) -> int:
        """Return the output feature dimension."""
        return self.output_dim


class TemporalEncoder(nn.Module):
    """Full Temporal Encoder with optional pre-training capability.
    
    Combines Photonic Delay-Loop Emulation with optional temporal
    feature encoding for enhanced representation learning.
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        kernel_sizes: List[int],
        kernel_init: str = "random",
        use_batch_norm: bool = False,
        dropout: float = 0.0,
        learn_kernels: bool = False,
    ):
        super().__init__()
        
        self.learn_kernels = learn_kernels
        
        # Main photonic delay loop emulator
        self.pdel = PhotonicDelayLoopEmulator(
            input_dim=input_dim,
            d_model=d_model,
            kernel_sizes=kernel_sizes,
            kernel_init=kernel_init,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
        )
        
        # Optional projection layer
        self.projection = nn.Linear(
            self.pdel.output_dim,
            d_model * len(kernel_sizes)
        ) if learn_kernels else nn.Identity()
        
        self.output_dim = d_model * len(kernel_sizes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
        Returns:
            Encoded features of shape (batch, seq_len, output_dim)
        """
        features = self.pdel(x)
        
        if self.learn_kernels:
            features = self.projection(features)
        
        return features
