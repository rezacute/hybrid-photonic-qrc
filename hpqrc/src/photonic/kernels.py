"""
Kernel Initialization Strategies for Photonic Delay Banks

Provides various kernel initialization strategies for photonic delay loops:
- Random: Xavier/Gaussian initialization
- Chirp: Linear frequency sweep
- Gabor: Gaussian-modulated sinusoids
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


def random_kernels(
    shape: Tuple[int, ...],
    seed: Optional[int] = None,
    scale: float = 1.0,
) -> torch.Tensor:
    """Initialize kernels with random values (Xavier initialization).
    
    Args:
        shape: Kernel shape (out_channels, in_channels, kernel_size)
        seed: Random seed for reproducibility
        scale: Scaling factor
    
    Returns:
        Tensor of shape shape
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    kernel = torch.randn(shape)
    # Xavier initialization
    nn.init.xavier_uniform_(kernel, gain=scale)
    
    return kernel


def chirp_kernels(
    shape: Tuple[int, ...],
    f_start: float = 0.1,
    f_end: float = 0.5,
) -> torch.Tensor:
    """Initialize kernels with linear frequency sweep (chirp).
    
    Creates kernels that sweep from f_start to f_end across the kernel.
    
    Args:
        shape: Kernel shape (out_channels, in_channels, kernel_size)
        f_start: Starting frequency (cycles per sample)
        f_end: Ending frequency
    
    Returns:
        Tensor with chirp patterns
    """
    out_ch, in_ch, kernel_size = shape
    
    kernels = []
    for i in range(out_ch):
        channel_kernels = []
        for j in range(in_ch):
            # Linear frequency sweep
            t = torch.arange(kernel_size, dtype=torch.float32)
            freq = torch.linspace(f_start, f_end, kernel_size)
            phase = 2 * np.pi * torch.cumsum(freq)
            
            # Chirp with exponential decay
            decay = torch.exp(-t / (kernel_size / 3))
            chirp = decay * torch.cos(phase)
            
            channel_kernels.append(chirp)
        
        kernels.append(torch.stack(channel_kernels))
    
    return torch.stack(kernels)


def gabor_kernels(
    shape: Tuple[int, ...],
    frequencies: Optional[List[float]] = None,
    sigma: Optional[float] = None,
) -> torch.Tensor:
    """Initialize kernels with Gabor wavelets (Gaussian-modulated sinusoids).
    
    Args:
        shape: Kernel shape (out_channels, in_channels, kernel_size)
        frequencies: List of frequencies for each output channel
        sigma: Gaussian width (default: kernel_size / 6)
    
    Returns:
        Tensor with Gabor patterns
    """
    out_ch, in_ch, kernel_size = shape
    
    if sigma is None:
        sigma = kernel_size / 6
    
    if frequencies is None:
        frequencies = [0.1 + 0.1 * i for i in range(out_ch)]
    
    kernels = []
    for i, freq in enumerate(frequencies):
        channel_kernels = []
        for j in range(in_ch):
            # Position around center
            t = torch.arange(kernel_size, dtype=torch.float32) - kernel_size / 2
            
            # Gaussian envelope
            gaussian = torch.exp(-t**2 / (2 * sigma**2))
            
            # Modulating sinusoid
            sinusoid = torch.cos(2 * np.pi * freq * t)
            
            # Gabor wavelet
            gabor = gaussian * sinusoid
            
            channel_kernels.append(gabor)
        
        kernels.append(torch.stack(channel_kernels))
    
    return torch.stack(kernels)


def initialize_bank(
    bank: nn.Conv1d,
    strategy: str = "random",
    **kwargs
) -> nn.Conv1d:
    """Initialize a convolutional bank with specified strategy.
    
    Args:
        bank: Conv1d layer to initialize
        strategy: Initialization strategy ("random", "chirp", "gabor", "xavier")
        **kwargs: Additional arguments for initialization functions
    
    Returns:
        Initialized Conv1d layer
    """
    shape = bank.weight.shape
    
    if strategy == "random":
        kernel = random_kernels(shape, **kwargs)
    elif strategy == "chirp":
        kernel = chirp_kernels(shape, **kwargs)
    elif strategy == "gabor":
        kernel = gabor_kernels(shape, **kwargs)
    elif strategy == "xavier":
        kernel = torch.randn(shape)
        nn.init.xavier_uniform_(kernel)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    with torch.no_grad():
        bank.weight.copy_(kernel)
    
    return bank


class KernelInitializer:
    """Container for kernel initialization utilities."""
    
    @staticmethod
    def random(shape: Tuple[int, ...], **kwargs) -> torch.Tensor:
        return random_kernels(shape, **kwargs)
    
    @staticmethod
    def chirp(shape: Tuple[int, ...], **kwargs) -> torch.Tensor:
        return chirp_kernels(shape, **kwargs)
    
    @staticmethod
    def gabor(shape: Tuple[int, ...], **kwargs) -> torch.Tensor:
        return gabor_kernels(shape, **kwargs)
