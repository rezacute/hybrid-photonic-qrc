"""
Quantum Encoding Schemes - Qiskit 2.x Compatible

Angle encoding and amplitude encoding for classical-to-quantum mapping.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class AngleEncoder(nn.Module):
    """Map D-dimensional classical features to N-qubit angle encoding.
    
    Uses RZ gates for encoding. Features are first projected to qubit dimension
    if needed, then mapped to angles.
    """
    
    def __init__(
        self,
        n_qubits: int,
        input_dim: Optional[int] = None,
    ):
        super().__init__()
        
        self.n_qubits = n_qubits
        self.input_dim = input_dim or n_qubits
        
        # Linear projection (stored as numpy array for quantum circuit use)
        if self.input_dim != n_qubits:
            # Learnable projection
            self.projection = nn.Linear(self.input_dim, n_qubits)
            self._projection_np = None
        else:
            self.projection = None
            self._projection_np = np.eye(n_qubits)
    
    def encode(self, features: np.ndarray) -> np.ndarray:
        """Encode features to angles for RZ gates.
        
        Args:
            features: Input features of shape (input_dim,) or (batch, input_dim)
        
        Returns:
            Angles of shape (n_qubits,)
        """
        if features.ndim == 1:
            # Single sample
            if self.projection is not None:
                with torch.no_grad():
                    proj = self.projection.weight.cpu().numpy()
                    features = proj @ features
            # Apply angle encoding: θ = π * tanh(x)
            angles = np.pi * np.tanh(features[:self.n_qubits])
            return angles
        else:
            # Batch
            raise NotImplementedError("Use forward() for batches")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning encoded angles.
        
        Args:
            x: Input features of shape (batch, input_dim)
        
        Returns:
            Angles of shape (batch, n_qubits)
        """
        if self.projection is not None:
            x = self.projection(x)
        
        # Angle encoding: θ = π * tanh(x)
        angles = np.pi * torch.tanh(x[:, :self.n_qubits])
        
        return angles
    
    @property
    def projection_matrix(self) -> np.ndarray:
        """Get projection matrix as numpy array."""
        if self.projection is not None:
            return self.projection.weight.detach().cpu().numpy()
        return self._projection_np


class AmplitudeEncoder(nn.Module):
    """Encode features into quantum state amplitudes.
    
    Requires 2^n normalization for valid quantum state.
    """
    
    def __init__(
        self,
        n_qubits: int,
        input_dim: Optional[int] = None,
    ):
        super().__init__()
        
        self.n_qubits = n_qubits
        self.n_amplitudes = 2 ** n_qubits
        self.input_dim = input_dim or self.n_amplitudes
        
        # Projection layer
        self.projection = nn.Linear(self.input_dim, self.n_amplitudes)
    
    def encode(self, features: np.ndarray) -> np.ndarray:
        """Encode features to normalized state vector.
        
        Args:
            features: Input features
        
        Returns:
            Normalized state vector of shape (2^n_qubits,)
        """
        # Project
        amplitudes = self.projection.weight @ features
        
        # Normalize
        norm = np.linalg.norm(amplitudes)
        if norm > 0:
            amplitudes = amplitudes / norm
        
        return amplitudes
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning normalized amplitudes.
        
        Args:
            x: Input features of shape (batch, input_dim)
        
        Returns:
            Normalized amplitudes of shape (batch, 2^n_qubits)
        """
        amplitudes = self.projection(x)
        
        # L2 normalize
        norm = torch.norm(amplitudes, dim=-1, keepdim=True)
        amplitudes = amplitudes / (norm + 1e-8)
        
        return amplitudes


class BasisEncoder(nn.Module):
    """Basis encoding (one-hot for discrete states)."""
    
    def __init__(self, n_qubits: int, n_states: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_states = n_states
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode integer states to basis vectors."""
        # x should be integers in [0, n_states)
        batch_size = x.shape[0]
        basis = torch.zeros(batch_size, self.n_qubits, device=x.device)
        
        for i in range(batch_size):
            state = int(x[i].item()) % self.n_states
            basis[i, state % self.n_qubits] = 1.0
        
        return basis
