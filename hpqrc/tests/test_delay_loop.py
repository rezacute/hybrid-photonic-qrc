"""
Tests for Photonic Delay-Loop module.
"""

import pytest
import torch
import numpy as np

from src.photonic.delay_loop import PhotonicDelayBank, PhotonicDelayLoopEmulator


def test_delay_bank_output_shape():
    """Test output shape of single delay bank."""
    bank = PhotonicDelayBank(
        in_channels=1,
        out_channels=8,
        kernel_size=4,
        activation="tanh",
    )
    
    # Input: (batch, channels, seq_len)
    x = torch.randn(4, 1, 100)
    
    out = bank(x)
    
    # Output should be (batch, out_channels, seq_len)
    assert out.shape == (4, 8, 100)


def test_delay_bank_causal_padding():
    """Test causal padding (no future information leakage)."""
    bank = PhotonicDelayBank(
        in_channels=1,
        out_channels=4,
        kernel_size=4,
        activation="identity",
    )
    
    # Input with clear temporal pattern
    x = torch.zeros(1, 1, 10)
    x[0, 0, 9] = 1.0  # Only last timestep has signal
    
    out = bank(x)
    
    # Output at earlier timesteps should be zero (causal)
    # With causal padding and kernel_size=4, first 3 outputs should be near zero
    assert torch.abs(out[0, 0, :3]).max() < 1e-5


def test_delay_bank_frozen():
    """Test frozen kernels."""
    bank_trainable = PhotonicDelayBank(
        in_channels=1,
        out_channels=4,
        kernel_size=4,
        activation="tanh",
        frozen=False,
    )
    
    bank_frozen = PhotonicDelayBank(
        in_channels=1,
        out_channels=4,
        kernel_size=4,
        activation="tanh",
        frozen=True,
    )
    
    # Trainable should have requires_grad
    assert any(p.requires_grad for p in bank_trainable.parameters())
    
    # Frozen should not
    assert not any(p.requires_grad for p in bank_frozen.parameters())


def test_pdel_output_shape():
    """Test multi-bank PDEL output shape."""
    pdel = PhotonicDelayLoopEmulator(
        in_channels=1,
        n_banks=3,
        kernel_sizes=[4, 8, 16],
        features_per_bank=8,
    )
    
    x = torch.randn(4, 1, 100)
    out = pdel(x)
    
    # Output: (batch, n_banks * features_per_bank, seq_len)
    expected_dim = 3 * 8
    assert out.shape == (4, expected_dim, 100)


def test_pdel_output_dim_property():
    """Test output_dim property."""
    pdel = PhotonicDelayLoopEmulator(
        in_channels=1,
        n_banks=5,
        features_per_bank=16,
    )
    
    assert pdel.output_dim == 5 * 16


def test_pdel_frozen_trainable_count():
    """Test parameter counting for frozen vs trainable."""
    pdel_trainable = PhotonicDelayLoopEmulator(
        in_channels=1,
        n_banks=2,
        kernel_sizes=[4, 8],
        features_per_bank=4,
        frozen_kernels=False,
    )
    
    pdel_frozen = PhotonicDelayLoopEmulator(
        in_channels=1,
        n_banks=2,
        kernel_sizes=[4, 8],
        features_per_bank=4,
        frozen_kernels=True,
    )
    
    # Trainable should have more params with grad
    assert pdel_trainable.n_trainable_params > pdel_frozen.n_trainable_params


def test_pdel_freeze_unfreeze():
    """Test freeze and unfreeze methods."""
    pdel = PhotonicDelayLoopEmulator(
        in_channels=1,
        n_banks=2,
        kernel_sizes=[4, 8],
        features_per_bank=4,
        frozen_kernels=False,
    )
    
    # Initially trainable
    assert pdel.n_trainable_params > 0
    
    # Freeze
    pdel.freeze()
    assert pdel.n_trainable_params == 0
    
    # Unfreeze
    pdel.unfreeze()
    assert pdel.n_trainable_params > 0


def test_pdel_activation_types():
    """Test different activation functions."""
    for activation in ["tanh", "relu", "identity"]:
        pdel = PhotonicDelayLoopEmulator(
            in_channels=1,
            n_banks=2,
            kernel_sizes=[4, 8],
            features_per_bank=4,
            activation=activation,
        )
        
        x = torch.randn(2, 1, 50)
        out = pdel(x)
        
        assert not torch.isnan(out).any()
