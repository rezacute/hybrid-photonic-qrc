"""
End-to-end test for HPQRC model.
"""

import pytest
import numpy as np
import torch

from src.models.hpqrc import HPQRC, HPQRCConfig
from src.data.preprocessing import create_sequences


def test_hpqrc_init(dummy_hpqrc_config, device):
    """Test HPQRC initialization."""
    model = HPQRC(dummy_hpqrc_config, device=device, seed=42)
    
    assert model.config == dummy_hpqrc_config
    assert model.device == device
    assert model.photonic is not None
    assert model.quantum is not None
    assert model.readout is not None


def test_hpqrc_extract_features(dummy_hpqrc_config, device):
    """Test feature extraction."""
    model = HPQRC(dummy_hpqrc_config, device=device, seed=42)
    
    # Create small batch
    batch = np.random.randn(4, 100).astype(np.float32)  # (batch, seq_len)
    
    features = model.extract_features(batch)
    
    # Check output shape
    assert features.shape[0] == 4  # batch size
    assert features.shape[1] > 0  # feature dimension


def test_hpqrc_fit_predict(dummy_hpqrc_config, device):
    """Test full fit/predict pipeline."""
    model = HPQRC(dummy_hpqrc_config, device=device, seed=42)
    
    # Generate small training data
    np.random.seed(42)
    X_train = np.random.randn(50, 50).astype(np.float32)
    y_train = np.random.randn(50).astype(np.float32)
    
    # Fit
    model.fit(X_train, y_train)
    
    # Predict
    X_test = np.random.randn(10, 50).astype(np.float32)
    y_pred = model.predict(X_test)
    
    # Check output
    assert y_pred.shape[0] == 10
    assert not np.isnan(y_pred).any()


def test_hpqrc_parameter_counts(dummy_hpqrc_config, device):
    """Test parameter counting."""
    model = HPQRC(dummy_hpqrc_config, device=device, seed=42)
    
    # Should have some trainable params (readout)
    assert model.trainable_params > 0
    
    # Total params should be >= trainable
    assert model.total_params >= model.trainable_params


def test_hpqrc_deterministic(device):
    """Test deterministic behavior with same seed."""
    config = HPQRCConfig(
        in_channels=1,
        n_banks=2,
        kernel_sizes=[4, 24],
        features_per_bank=4,
        n_qubits=4,
        ridge_alpha=1.0,
    )
    
    # Create two models with same seed
    model1 = HPQRC(config, device=device, seed=42)
    model2 = HPQRC(config, device=device, seed=42)
    
    # Same input
    x = np.random.randn(2, 50).astype(np.float32)
    
    # Extract features
    feat1 = model1.extract_features(x)
    feat2 = model2.extract_features(x)
    
    # Should be identical
    np.testing.assert_array_almost_equal(feat1, feat2)
