"""
Temporal Encoder with Photonic Delay-Loop Emulation

Full PDEL module with optional pre-training on auxiliary tasks
and freeze/unfreeze functionality.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, List
from .delay_loop import PhotonicDelayLoopEmulator
from .kernels import initialize_bank


class TemporalEncoder(nn.Module):
    """Full Temporal Encoder wrapping PhotonicDelayLoopEmulator.
    
    Provides:
    - Multi-scale temporal feature extraction
    - Optional pre-training on reconstruction
    - Freeze/unfreeze functionality
    """
    
    def __init__(
        self,
        in_channels: int,
        n_banks: int = 5,
        kernel_sizes: List[int] = None,
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
        
        # Main PDEL module
        self.pdel = PhotonicDelayLoopEmulator(
            in_channels=in_channels,
            n_banks=n_banks,
            kernel_sizes=self.kernel_sizes,
            features_per_bank=features_per_bank,
            activation=activation,
            frozen_kernels=frozen_kernels,
            dropout=dropout,
        )
        
        # Optional projection to fixed output dimension
        self.output_dim = self.pdel.output_dim
        self.projection = nn.Identity()  # Can be replaced with Linear if needed
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, channels, seq_len)
        Returns:
            Encoded features of shape (batch, output_dim, seq_len)
        """
        features = self.pdel(x)
        return self.projection(features)
    
    def freeze(self):
        """Freeze all PDEL parameters."""
        self.pdel.freeze()
    
    def unfreeze(self):
        """Unfreeze all PDEL parameters."""
        self.pdel.unfreeze()
    
    @property
    def is_frozen(self) -> bool:
        """Check if encoder is frozen."""
        return not any(p.requires_grad for p in self.parameters())
    
    def pretrain_on_reconstruction(
        self,
        data_loader: DataLoader,
        epochs: int = 10,
        lr: float = 0.001,
        device: str = "cpu",
    ):
        """Pre-train the encoder on reconstruction task.
        
        Args:
            data_loader: DataLoader with (x, x) pairs
            epochs: Number of pre-training epochs
            lr: Learning rate
            device: Device to train on
        """
        # Unfreeze for pre-training
        self.unfreeze()
        
        # Simple reconstruction loss
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        self.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in data_loader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0].to(device)
                else:
                    x = batch.to(device)
                
                # Forward pass
                reconstructed = self(x)
                
                # Loss: reconstruct original
                loss = criterion(reconstructed, x)
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(data_loader)
            if (epoch + 1) % 2 == 0:
                print(f"Pretrain Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        # Optionally freeze after pre-training
        if self.frozen_kernels:
            self.freeze()
        
        return self
    
    def get_output_dim(self) -> int:
        """Return output feature dimension."""
        return self.output_dim
    
    @property
    def n_trainable_params(self) -> int:
        """Number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TemporalEncoderWithProjection(nn.Module):
    """Temporal encoder with projection to fixed dimension."""
    
    def __init__(
        self,
        in_channels: int,
        output_dim: int,
        n_banks: int = 5,
        kernel_sizes: List[int] = None,
        features_per_bank: int = 16,
        **kwargs
    ):
        super().__init__()
        
        self.encoder = TemporalEncoder(
            in_channels=in_channels,
            n_banks=n_banks,
            kernel_sizes=kernel_sizes,
            features_per_bank=features_per_bank,
            **kwargs
        )
        
        # Projection layer
        self.projection = nn.Linear(
            self.encoder.output_dim,
            output_dim
        )
        
        self.output_dim = output_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        # Apply projection along channel dimension
        features = features.permute(0, 2, 1)  # (batch, seq, channels)
        features = self.projection(features)
        features = features.permute(0, 2, 1)  # (batch, channels, seq)
        return features
    
    def freeze(self):
        self.encoder.freeze()
    
    def unfreeze(self):
        self.encoder.unfreeze()
