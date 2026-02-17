"""
Photonic Delay-Line Reservoir with Adaptive Taps

Two variants:
1. PhotonicDelayReservoir - Fixed taps (original)
2. PhotonicDelayReservoirAdaptive - Learnable tap weights
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional


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
        target_spectral_radius: float = 0.95,
        input_scaling: float = 0.5,
        leak_rate: float = 0.1,
        sparsity: float = 0.9,
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
        target_radius = target_spectral_radius
        per_tap_radius = target_radius / np.sqrt(self.n_taps)
        
        # Create feedback matrices
        W_fb_list = []
        
        for tau in delay_taps:
            W_k = rng.standard_normal((reservoir_dim, reservoir_dim))
            mask = rng.random(W_k.shape) < (1 - sparsity)
            W_k = W_k * mask
            
            current_radius = np.max(np.abs(np.linalg.eigvals(W_k)))
            if current_radius > 0:
                W_k = W_k * (per_tap_radius / current_radius)
            
            W_fb_list.append(torch.tensor(W_k, dtype=torch.float32))
        
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
        
        # Pre-compute all input drives
        drive = torch.matmul(x, self.W_in.T)
        
        # History buffer
        h_history = torch.zeros(
            batch, self.max_delay + seq_len, self.reservoir_dim,
            device=x.device, dtype=x.dtype
        )
        
        h = torch.zeros(batch, self.reservoir_dim, device=x.device, dtype=x.dtype)
        
        states = []
        
        for t in range(seq_len):
            delayed_states = torch.stack([
                h_history[:, self.max_delay + t - tau, :]
                for tau in self.delay_taps
            ])
            
            feedback = torch.einsum('tbr,trR->bR', delayed_states, self.W_fb)
            
            h_new = (1 - self.leak_rate) * h + self.leak_rate * torch.tanh(
                drive[:, t, :] + feedback + self.bias
            )
            
            h_history[:, self.max_delay + t, :] = h_new
            h = h_new
            states.append(h_new)
        
        return torch.stack(states, dim=1)


class PhotonicDelayReservoirAdaptive(nn.Module):
    """Photonic delay-line reservoir with adaptive tap weights.
    
    Learnable tap importance weights (alpha_k) that can be tuned per task.
    Still within reservoir paradigm - only K=4 additional parameters.
    """
    
    def __init__(
        self,
        input_dim: int = 1,
        reservoir_dim: int = 200,
        delay_taps: List[int] = [1, 4, 24, 96, 168],
        target_spectral_radius: float = 0.95,
        input_scaling: float = 0.5,
        leak_rate: float = 0.1,
        sparsity: float = 0.9,
        seed: int = 42,
        learn_tap_weights: bool = True,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.reservoir_dim = reservoir_dim
        self.delay_taps = delay_taps
        self.n_taps = len(delay_taps)
        self.max_delay = max(delay_taps)
        self.leak_rate = leak_rate
        self.learn_tap_weights = learn_tap_weights
        
        rng = np.random.default_rng(seed)
        
        # Input weight matrix
        W_in = rng.uniform(-input_scaling, input_scaling, size=(reservoir_dim, input_dim))
        self.register_buffer('W_in', torch.tensor(W_in, dtype=torch.float32))
        
        # Fixed spectral radius budget across taps
        target_radius = target_spectral_radius
        per_tap_radius = target_radius / np.sqrt(self.n_taps)
        
        # Create feedback matrices
        W_fb_list = []
        
        for tau in delay_taps:
            W_k = rng.standard_normal((reservoir_dim, reservoir_dim))
            mask = rng.random(W_k.shape) < (1 - sparsity)
            W_k = W_k * mask
            
            current_radius = np.max(np.abs(np.linalg.eigvals(W_k)))
            if current_radius > 0:
                W_k = W_k * (per_tap_radius / current_radius)
            
            W_fb_list.append(torch.tensor(W_k, dtype=torch.float32))
        
        self.register_buffer('W_fb', torch.stack(W_fb_list))
        
        # Learnable tap importance weights (one per tap)
        if learn_tap_weights:
            # Initialize at 1.0 (uniform importance)
            self.tap_weights = nn.Parameter(torch.ones(self.n_taps))
        else:
            self.register_buffer('tap_weights', torch.ones(self.n_taps))
        
        # Bias
        self.register_buffer(
            'bias',
            torch.tensor(rng.uniform(-0.1, 0.1, size=reservoir_dim), dtype=torch.float32)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input with adaptive tap weights."""
        batch, seq_len, _ = x.shape
        
        # Pre-compute all input drives
        drive = torch.matmul(x, self.W_in.T)
        
        # Normalize tap weights (softmax-like)
        tap_w = torch.softmax(self.tap_weights, dim=0)
        
        # History buffer
        h_history = torch.zeros(
            batch, self.max_delay + seq_len, self.reservoir_dim,
            device=x.device, dtype=x.dtype
        )
        
        h = torch.zeros(batch, self.reservoir_dim, device=x.device, dtype=x.dtype)
        
        states = []
        
        for t in range(seq_len):
            delayed_states = torch.stack([
                h_history[:, self.max_delay + t - tau, :]
                for tau in self.delay_taps
            ])
            
            # Apply learned tap weights
            # tap_w: (n_taps,), delayed_states: (n_taps, batch, reservoir), W_fb: (n_taps, reservoir, reservoir)
            weighted_feedback = torch.zeros(batch, self.reservoir_dim, device=x.device)
            for k in range(self.n_taps):
                weighted_feedback += tap_w[k] * (delayed_states[k] @ self.W_fb[k])
            
            h_new = (1 - self.leak_rate) * h + self.leak_rate * torch.tanh(
                drive[:, t, :] + weighted_feedback + self.bias
            )
            
            h_history[:, self.max_delay + t, :] = h_new
            h = h_new
            states.append(h_new)
        
        return torch.stack(states, dim=1)


class PhotonicDelayReservoirGated(nn.Module):
    """Photonic delay-line reservoir with input-dependent gating.
    
    Uses a simple gate to weight taps based on current input context.
    More parameters but potentially better on complex data.
    """
    
    def __init__(
        self,
        input_dim: int = 1,
        reservoir_dim: int = 200,
        delay_taps: List[int] = [1, 4, 24, 96, 168],
        target_spectral_radius: float = 0.95,
        input_scaling: float = 0.5,
        leak_rate: float = 0.1,
        sparsity: float = 0.9,
        seed: int = 42,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.reservoir_dim = reservoir_dim
        self.delay_taps = delay_taps
        self.n_taps = len(delay_taps)
        self.max_delay = max(delay_taps)
        self.leak_rate = leak_rate
        
        rng = np.random.default_rng(seed)
        
        # Input weight matrix
        W_in = rng.uniform(-input_scaling, input_scaling, size=(reservoir_dim, input_dim))
        self.register_buffer('W_in', torch.tensor(W_in, dtype=torch.float32))
        
        # Tap gating network (small MLP)
        self.gate_net = nn.Sequential(
            nn.Linear(input_dim, self.n_taps),
            nn.Sigmoid()
        )
        
        # Fixed spectral radius budget
        target_radius = target_spectral_radius
        per_tap_radius = target_radius / np.sqrt(self.n_taps)
        
        # Create feedback matrices
        W_fb_list = []
        
        for tau in delay_taps:
            W_k = rng.standard_normal((reservoir_dim, reservoir_dim))
            mask = rng.random(W_k.shape) < (1 - sparsity)
            W_k = W_k * mask
            
            current_radius = np.max(np.abs(np.linalg.eigvals(W_k)))
            if current_radius > 0:
                W_k = W_k * (per_tap_radius / current_radius)
            
            W_fb_list.append(torch.tensor(W_k, dtype=torch.float32))
        
        self.register_buffer('W_fb', torch.stack(W_fb_list))
        
        self.register_buffer(
            'bias',
            torch.tensor(rng.uniform(-0.1, 0.1, size=reservoir_dim), dtype=torch.float32)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input with input-dependent gating."""
        batch, seq_len, _ = x.shape
        
        drive = torch.matmul(x, self.W_in.T)
        
        h_history = torch.zeros(
            batch, self.max_delay + seq_len, self.reservoir_dim,
            device=x.device, dtype=x.dtype
        )
        
        h = torch.zeros(batch, self.reservoir_dim, device=x.device, dtype=x.dtype)
        
        states = []
        
        for t in range(seq_len):
            # Compute gate from current input
            gate = self.gate_net(x[:, t, :])  # (batch, n_taps)
            
            delayed_states = torch.stack([
                h_history[:, self.max_delay + t - tau, :]
                for tau in self.delay_taps
            ])
            
            # Apply gated weights
            weighted_feedback = torch.einsum('bt,br,tR->bR', gate, delayed_states, self.W_fb)
            
            h_new = (1 - self.leak_rate) * h + self.leak_rate * torch.tanh(
                drive[:, t, :] + weighted_feedback + self.bias
            )
            
            h_history[:, self.max_delay + t, :] = h_new
            h = h_new
            states.append(h_new)
        
        return torch.stack(states, dim=1)
