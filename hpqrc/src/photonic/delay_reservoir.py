"""
Photonic Delay-Line Reservoir

Emulates a photonic delay-line reservoir with multi-scale delayed feedback.
Unlike Conv1D banks (feedforward), this maintains recurrent state via delayed self-connections.

Physical analogy: fiber-optic ring cavity with multiple delay loops at τ_k.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List


class PhotonicDelayReservoir(nn.Module):
    """Emulates a photonic delay-line reservoir with multi-scale delayed feedback.
    
    Unlike Conv1D banks (feedforward), this maintains recurrent state via delayed
    self-connections. Physical analogy: fiber-optic ring cavity with multiple
    delay loops at τ_k, each with coupling strength α_k.
    """
    
    def __init__(
        self,
        input_dim: int = 1,
        reservoir_dim: int = 80,
        delay_taps: List[int] = [1, 4, 24, 96, 168],
        spectral_radius: float = 0.9,
        input_scaling: float = 0.5,
        leak_rate: float = 0.3,
        activation: str = "tanh",
        seed: int = 42,
    ):
        """Initialize Photonic Delay Reservoir.
        
        Args:
            input_dim: Input feature dimension
            reservoir_dim: Number of virtual nodes
            delay_taps: List of delay taps (in timesteps)
            spectral_radius: Spectral radius for stability
            input_scaling: Input weight scaling
            leak_rate: Leak rate (inertia, like photonic cavity lifetime)
            activation: Activation function
            seed: Random seed
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.reservoir_dim = reservoir_dim
        self.delay_taps = delay_taps
        self.max_delay = max(delay_taps)
        self.leak_rate = leak_rate
        self.activation = activation
        
        # Random number generator
        rng = np.random.default_rng(seed)
        
        # Input weight matrix (fixed, not trained - reservoir paradigm)
        W_in = rng.uniform(
            -input_scaling, 
            input_scaling, 
            size=(reservoir_dim, input_dim)
        )
        self.register_buffer('W_in', torch.tensor(W_in, dtype=torch.float32))
        
        # Create delayed feedback matrices (one per delay tap)
        # Each W_fb_k is sparse, scaled so combined spectral radius < 1
        self.W_fb_tensors = []
        
        for tau in delay_taps:
            # Random sparse feedback matrix
            W_k = rng.standard_normal((reservoir_dim, reservoir_dim))
            
            # Scale to target spectral radius contribution
            eigenvalues = np.linalg.eigvals(W_k)
            max_eig = np.max(np.abs(eigenvalues))
            if max_eig > 0:
                # Scale so combined taps give ~spectral_radius
                W_k = W_k * (spectral_radius / len(delay_taps)) / max_eig
            
            # Store as buffer (not trained)
            self.register_buffer(f'W_fb_{tau}', torch.tensor(W_k, dtype=torch.float32))
        
        # Bias
        self.register_buffer(
            'bias', 
            torch.tensor(
                rng.uniform(-0.1, 0.1, size=reservoir_dim),
                dtype=torch.float32
            )
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input through delay-line reservoir.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
        
        Returns:
            States tensor of shape (batch, seq_len, reservoir_dim)
        """
        batch, seq_len, _ = x.shape
        
        # Initialize history buffer for delayed feedback
        # Pad with zeros at the beginning
        h_history = torch.zeros(
            batch, 
            self.max_delay + seq_len, 
            self.reservoir_dim, 
            device=x.device,
            dtype=x.dtype
        )
        
        # Initialize hidden state
        h = torch.zeros(batch, self.reservoir_dim, device=x.device, dtype=x.dtype)
        
        states = []
        
        for t in range(seq_len):
            # Input drive
            drive = torch.matmul(x[:, t, :], self.W_in.T)  # (batch, reservoir_dim)
            
            # Delayed feedback from multiple taps
            feedback = torch.zeros_like(drive)
            
            for tau in self.delay_taps:
                # Get delayed state
                if t - tau >= 0:
                    h_delayed = h_history[:, self.max_delay + t - tau, :]
                else:
                    # Not enough history, use initial zeros
                    h_delayed = torch.zeros_like(drive)
                
                # Get feedback matrix
                W_k = getattr(self, f'W_fb_{tau}')
                feedback += torch.matmul(h_delayed, W_k.T)
            
            # Leaky integration (photonic cavity lifetime analogy)
            h_new = (1 - self.leak_rate) * h + self.leak_rate * torch.tanh(
                drive + feedback + self.bias
            )
            
            # Update history and state
            h_history[:, self.max_delay + t, :] = h_new
            h = h_new
            states.append(h_new)
        
        return torch.stack(states, dim=1)  # (batch, seq_len, reservoir_dim)


class PhotonicDelayReservoirWithReadout(nn.Module):
    """Photonic delay reservoir with trainable linear readout."""
    
    def __init__(self, input_dim=1, reservoir_dim=80, **kwargs):
        super().__init__()
        self.reservoir = PhotonicDelayReservoir(input_dim, reservoir_dim, **kwargs)
        self.reservoir_dim = reservoir_dim
        self.readout = nn.Linear(reservoir_dim, 1)
    
    def forward(self, x):
        states = self.reservoir(x)  # (batch, seq, reservoir)
        # Use last state
        last_state = states[:, -1, :]  # (batch, reservoir)
        return self.readout(last_state)
