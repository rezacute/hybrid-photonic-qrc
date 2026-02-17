"""
Echo State Network (ESN) - Classical Reservoir Computing

Sparse random reservoir with spectral radius control.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional


class EchoStateNetwork:
    """Echo State Network with leaky integration.
    
    Key features:
    - Sparse random reservoir weight matrix
    - Spectral radius control for echo state property
    - Leaky integration for memory
    """
    
    def __init__(
        self,
        input_dim: int,
        reservoir_size: int = 256,
        spectral_radius: float = 0.95,
        input_scaling: float = 0.5,
        leaking_rate: float = 0.3,
        sparsity: float = 0.9,
        ridge_alpha: float = 1.0,
        seed: int = 42,
    ):
        """
        Args:
            input_dim: Input feature dimension
            reservoir_size: Number of reservoir neurons
            spectral_radius: Spectral radius of reservoir matrix
            input_scaling: Input weight scaling
            leaking_rate: Leaking rate (α)
            sparsity: Fraction of zero connections
            ridge_alpha: Ridge regression regularization
            seed: Random seed
        """
        self.input_dim = input_dim
        self.reservoir_size = reservoir_size
        self.spectral_radius = spectral_radius
        self.input_scaling = input_scaling
        self.leaking_rate = leaking_rate
        self.sparsity = sparsity
        self.ridge_alpha = ridge_alpha
        self.seed = seed
        
        # Initialize
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Build reservoir matrix
        self.W = self._build_reservoir_matrix()
        
        # Build input weight matrix
        self.W_in = self._build_input_matrix()
        
        # Reservoir states
        self.states = np.zeros(reservoir_size)
        
        # Readout
        from ..readout.ridge import RidgeReadout
        self.readout = RidgeReadout(alpha=ridge_alpha)
        self._is_fitted = False
    
    def _build_reservoir_matrix(self) -> np.ndarray:
        """Build sparse random reservoir matrix."""
        # Generate sparse random matrix
        W = np.random.randn(self.reservoir_size, self.reservoir_size)
        
        # Apply sparsity
        mask = np.random.rand(*W.shape) > self.sparsity
        W = W * mask
        
        # Scale to target spectral radius
        eigenvalues = np.linalg.eigvals(W)
        max_eig = np.max(np.abs(eigenvalues))
        if max_eig > 0:
            W = W * (self.spectral_radius / max_eig)
        
        return W
    
    def _build_input_matrix(self) -> np.ndarray:
        """Build input weight matrix."""
        W_in = np.random.randn(self.reservoir_size, self.input_dim)
        W_in = W_in * self.input_scaling
        return W_in
    
    def _compute_states(self, x: np.ndarray) -> np.ndarray:
        """Compute reservoir states for input sequence.
        
        Args:
            x: Input sequence of shape (seq_len, input_dim)
        
        Returns:
            States of shape (seq_len, reservoir_size)
        """
        seq_len = x.shape[0]
        states = np.zeros((seq_len, self.reservoir_size))
        
        state = np.zeros(self.reservoir_size)
        
        for t in range(seq_len):
            # Compute pre-activation
            pre = self.W @ state + self.W_in @ x[t]
            
            # Leaky integration
            state = (1 - self.leaking_rate) * state + self.leaking_rate * np.tanh(pre)
            
            states[t] = state
        
        return states
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "EchoStateNetwork":
        """Fit the readout on reservoir states.
        
        Args:
            X: Input of shape (n_samples, seq_len, input_dim)
            y: Targets of shape (n_samples,) or (n_samples, n_targets)
        
        Returns:
            self
        """
        # Collect all states
        all_states = []
        for i in range(len(X)):
            states = self._compute_states(X[i])
            all_states.append(states)
        
        # Use final state for prediction
        X_states = np.array([s[-1] for s in all_states])
        
        self.readout.fit(X_states, y)
        self._is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using trained readout.
        
        Args:
            X: Input of shape (n_samples, seq_len, input_dim)
        
        Returns:
            Predictions
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")
        
        # Compute states
        all_states = []
        for i in range(len(X)):
            states = self._compute_states(X[i])
            all_states.append(states[-1])
        
        X_states = np.array(all_states)
        
        return self.readout.predict(X_states)
    
    @property
    def n_params(self) -> int:
        """Number of trainable parameters (readout only)."""
        return self.readout.n_params


class ESNtorch(nn.Module):
    """PyTorch-based ESN for GPU acceleration."""
    
    def __init__(
        self,
        input_dim: int,
        reservoir_size: int = 256,
        spectral_radius: float = 0.95,
        input_scaling: float = 0.5,
        leaking_rate: float = 0.3,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.reservoir_size = reservoir_size
        self.leaking_rate = leaking_rate
        
        # Reservoir matrix (fixed)
        W = torch.randn(reservoir_size, reservoir_size) * spectral_radius
        self.register_buffer('W', W)
        
        # Input matrix (fixed)
        W_in = torch.randn(reservoir_size, input_dim) * input_scaling
        self.register_buffer('W_in', W_in)
        
        # State
        self.register_buffer('state', torch.zeros(reservoir_size))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input sequence.
        
        Args:
            x: Input of shape (batch, seq_len, input_dim)
        
        Returns:
            Final states of shape (batch, reservoir_size)
        """
        batch_size, seq_len, _ = x.shape
        
        # Reset state
        state = torch.zeros(batch_size, self.reservoir_size, device=x.device)
        
        for t in range(seq_len):
            pre = torch.matmul(state, self.W.T) + torch.matmul(x[:, t], self.W_in.T)
            state = (1 - self.leaking_rate) * state + self.leaking_rate * torch.tanh(pre)
        
        return state
