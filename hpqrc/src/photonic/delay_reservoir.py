"""
Photonic Delay-Line Reservoir (Fixed)

Emulates a photonic delay-line reservoir with multi-scale delayed feedback.
Fixed: Proper spectral radius scaling with energy budget and sparsity.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List


class PhotonicDelayReservoir(nn.Module):
    """Photonic delay-line reservoir with proper stability control.
    
    Key fixes:
    - Energy-based spectral radius budget across delay taps
    - Sparsity (90%) like physical photonic systems
    - Pre-stacked matrices for batched matmul
    """
    
    def __init__(
        self,
        input_dim: int = 1,
        reservoir_dim: int = 200,
        delay_taps: List[int] = [1, 4, 24, 96, 168],
        target_spectral_radius: float = 0.95,  # Below 1.0 for ESP
        input_scaling: float = 0.5,
        leak_rate: float = 0.1,
        sparsity: float = 0.9,  # 90% zeros like physical systems
        seed: int = 42,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.reservoir_dim = reservoir_dim
        self.delay_taps = delay_taps
        self.n_taps = len(delay_taps)
        self.max_delay = max(delay_taps)
        self.leak_rate = leak_rate
        self.sparsity = sparsity
        
        rng = np.random.default_rng(seed)
        
        # Input weight matrix
        W_in = rng.uniform(-input_scaling, input_scaling, size=(reservoir_dim, input_dim))
        self.register_buffer('W_in', torch.tensor(W_in, dtype=torch.float32))
        
        # Fixed spectral radius budget across taps
        # Energy-based split: each tap gets target_radius / sqrt(n_taps)
        target_radius = target_spectral_radius
        per_tap_radius = target_radius / np.sqrt(self.n_taps)
        
        # Create feedback matrices
        W_fb_list = []
        
        for tau in delay_taps:
            # Random matrix
            W_k = rng.standard_normal((reservoir_dim, reservoir_dim))
            
            # Apply sparsity (like physical photonic systems)
            mask = rng.random(W_k.shape) < (1 - sparsity)
            W_k = W_k * mask
            
            # Scale to target per-tap radius
            current_radius = np.max(np.abs(np.linalg.eigvals(W_k)))
            if current_radius > 0:
                W_k = W_k * (per_tap_radius / current_radius)
            
            W_fb_list.append(torch.tensor(W_k, dtype=torch.float32))
        
        # Stack for batched matmul: (n_taps, reservoir_dim, reservoir_dim)
        self.register_buffer('W_fb', torch.stack(W_fb_list))
        
        # Bias
        self.register_buffer(
            'bias',
            torch.tensor(rng.uniform(-0.1, 0.1, size=reservoir_dim), dtype=torch.float32)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input through delay-line reservoir.
        
        Args:
            x: Input tensor (batch, seq_len, input_dim)
        
        Returns:
            States tensor (batch, seq_len, reservoir_dim)
        """
        batch, seq_len, _ = x.shape
        
        # History buffer
        h_history = torch.zeros(
            batch, self.max_delay + seq_len, self.reservoir_dim,
            device=x.device, dtype=x.dtype
        )
        
        h = torch.zeros(batch, self.reservoir_dim, device=x.device, dtype=x.dtype)
        
        states = []
        
        for t in range(seq_len):
            # Input drive
            drive = torch.matmul(x[:, t, :], self.W_in.T)
            
            # Get delayed states: (n_taps, batch, reservoir)
            delayed_states = torch.stack([
                h_history[:, self.max_delay + t - tau, :]
                for tau in self.delay_taps
            ])
            
            # Batched matmul: (n_taps, batch, reservoir) @ (n_taps, reservoir, reservoir) -> (n_taps, batch, reservoir)
            # Then sum over taps: (batch, reservoir)
            feedback = torch.einsum('tbr,trR->bR', delayed_states, self.W_fb)
            
            # Leaky integration
            h_new = (1 - self.leak_rate) * h + self.leak_rate * torch.tanh(
                drive + feedback + self.bias
            )
            
            h_history[:, self.max_delay + t, :] = h_new
            h = h_new
            states.append(h_new)
        
        return torch.stack(states, dim=1)


class PhotonicDelayReservoirWithReadout(nn.Module):
    """PDR with trainable linear readout."""
    
    def __init__(self, input_dim=1, reservoir_dim=200, **kwargs):
        super().__init__()
        self.reservoir = PhotonicDelayReservoir(input_dim, reservoir_dim, **kwargs)
        self.readout = nn.Linear(reservoir_dim, 1)
    
    def forward(self, x):
        states = self.reservoir(x)
        return self.readout(states[:, -1, :])
