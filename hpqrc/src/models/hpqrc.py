"""
HPQRC: Hybrid Photonic-Quantum Reservoir Computing Model

This module implements the full HPQRC architecture combining:
1. Photonic Delay-Loop Emulation Layer (PDEL)
2. Quantum Reservoir
3. Linear Readout
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import numpy as np

from ..photonic.delay_loop import PhotonicDelayLoopEmulator
from ..quantum.reservoir import QuantumReservoir
from ..readout.ridge import RidgeReadout, TrainableLinearReadout


class HPQRC(nn.Module):
    """Hybrid Photonic-Quantum Reservoir Computing Model.
    
    Architecture:
    1. Photonic Delay-Loop Emulation (PDEL) - multiple temporal scales
    2. Quantum Reservoir - fixed, untrainable quantum circuit
    3. Linear Readout - Ridge regression or trainable linear
    """
    
    def __init__(
        self,
        input_dim: int = 1,
        # Photonic config
        photonic_enabled: bool = True,
        n_banks: int = 5,
        kernel_sizes: list = None,
        d_model: int = 32,
        kernel_init: str = "random",
        # Quantum config
        quantum_enabled: bool = True,
        n_qubits: int = 6,
        n_layers: int = 1,
        coupling_strength: float = 1.0,
        transverse_field: float = 0.5,
        # Readout config
        readout_type: str = "ridge",  # "ridge" or "trainable"
        alpha: float = 1.0,
        # Training config
        training_style: str = "rc",  # "rc" = fit only readout, "dl" = backprop all
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.photonic_enabled = photonic_enabled
        self.quantum_enabled = quantum_enabled
        self.training_style = training_style
        
        # Default kernel sizes (sub-hourly to weekly)
        if kernel_sizes is None:
            kernel_sizes = [4, 24, 96, 168, 672]
        self.kernel_sizes = kernel_sizes
        
        # Photonic layer
        if photonic_enabled:
            self.photonic = PhotonicDelayLoopEmulator(
                input_dim=input_dim,
                d_model=d_model,
                kernel_sizes=kernel_sizes,
                kernel_init=kernel_init,
            )
            self.photonic_output_dim = self.photonic.get_output_dim()
        else:
            self.photonic = None
            self.photonic_output_dim = 0
        
        # Quantum reservoir
        if quantum_enabled:
            self.quantum = QuantumReservoir(
                n_qubits=n_qubits,
                n_layers=n_layers,
                coupling_strength=coupling_strength,
                transverse_field=transverse_field,
            )
            self.quantum_output_dim = self.quantum.get_output_dim()
        else:
            self.quantum = None
            self.quantum_output_dim = 0
        
        # Combined feature dimension
        self.feature_dim = self.photonic_output_dim + self.quantum_output_dim
        
        # Readout layer
        if readout_type == "ridge":
            self.readout = RidgeReadout(
                input_dim=self.feature_dim,
                output_dim=1,
                alpha=alpha,
            )
        elif readout_type == "trainable":
            self.readout = TrainableLinearReadout(
                input_dim=self.feature_dim,
                output_dim=1,
            )
        else:
            raise ValueError(f"Unknown readout type: {readout_type}")
        
        self.readout_type = readout_type
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input time series of shape (batch, seq_len, input_dim)
        
        Returns:
            Predictions of shape (batch, seq_len - lookback)
        """
        batch_size, seq_len, _ = x.shape
        
        # Get features from photonic layer
        if self.photonic_enabled:
            photonic_features = self.photonic(x)  # (batch, seq', photonic_dim)
        else:
            photonic_features = None
        
        # Get features from quantum reservoir
        if self.quantum_enabled:
            # For RC style, we process the last timestep
            if self.training_style == "rc":
                # Use mean of photonic features as input to quantum
                if photonic_features is not None:
                    quantum_input = photonic_features.mean(dim=1)  # (batch, photonic_dim)
                    # Project to n_qubits if needed
                    if quantum_input.shape[-1] != self.quantum.n_qubits:
                        # Use first n_qubits dimensions
                        quantum_input = quantum_input[:, :self.quantum.n_qubits]
                else:
                    quantum_input = x[:, -1, :]  # Use last timestep
            else:
                quantum_input = x
            
            quantum_features = self.quantum(quantum_input)  # (batch, quantum_dim)
        else:
            quantum_features = None
        
        # Combine features
        features = []
        if photonic_features is not None:
            # Use last valid timestep
            features.append(photonic_features[:, -1, :])
        if quantum_features is not None:
            features.append(quantum_features)
        
        combined = torch.cat(features, dim=-1)  # (batch, feature_dim)
        
        # Get prediction
        output = self.readout(combined)  # (batch, 1)
        
        return output
    
    def fit_readout(self, X: np.ndarray, y: np.ndarray):
        """Fit the readout layer (for RC-style training)."""
        if self.readout_type == "ridge":
            self.readout.fit(X, y)
        else:
            raise RuntimeError("Only ridge readout can be fit directly")
    
    def get_num_params(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())


class PureQRC(nn.Module):
    """Pure Quantum Reservoir Computing (no photonic layer)."""
    
    def __init__(
        self,
        input_dim: int = 1,
        n_qubits: int = 6,
        n_layers: int = 1,
        readout_type: str = "ridge",
        alpha: float = 1.0,
    ):
        super().__init__()
        
        self.quantum = QuantumReservoir(
            n_qubits=n_qubits,
            n_layers=n_layers,
        )
        
        self.input_proj = nn.Linear(input_dim, n_qubits)
        
        if readout_type == "ridge":
            self.readout = RidgeReadout(n_qubits * 3, 1, alpha)
        else:
            self.readout = TrainableLinearReadout(n_qubits * 3, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Project input
        x_proj = self.input_proj(x)  # (batch, seq, n_qubits)
        
        # Quantum processing (use last timestep)
        quantum_out = self.quantum(x_proj[:, -1, :])
        
        return self.readout(quantum_out)


class PhotonicRC(nn.Module):
    """Photonic Reservoir Computing (no quantum layer)."""
    
    def __init__(
        self,
        input_dim: int = 1,
        n_banks: int = 5,
        kernel_sizes: list = None,
        d_model: int = 32,
        readout_type: str = "ridge",
        alpha: float = 1.0,
    ):
        super().__init__()
        
        if kernel_sizes is None:
            kernel_sizes = [4, 24, 96, 168, 672]
        
        self.photonic = PhotonicDelayLoopEmulator(
            input_dim=input_dim,
            d_model=d_model,
            kernel_sizes=kernel_sizes,
        )
        
        output_dim = self.photonic.get_output_dim()
        
        if readout_type == "ridge":
            self.readout = RidgeReadout(output_dim, 1, alpha)
        else:
            self.readout = TrainableLinearReadout(output_dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.photonic(x)
        # Use last timestep
        return self.readout(features[:, -1, :])
