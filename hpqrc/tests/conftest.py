"""
Test fixtures for HPQRC tests.
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import tempfile

from src.data.synthetic import generate_synthetic_ev_demand, SyntheticConfig
from src.models.hpqrc import HPQRCConfig


@pytest.fixture
def small_synthetic_data():
    """Generate small synthetic dataset for fast testing."""
    config = SyntheticConfig(
        n_samples=1000,
        resolution_minutes=15,
        seed=42,
    )
    df = generate_synthetic_ev_demand(config)
    return df


@pytest.fixture
def dummy_hpqrc_config():
    """HPQRCConfig with small parameters for testing."""
    return HPQRCConfig(
        in_channels=1,
        n_banks=2,
        kernel_sizes=[4, 24],
        features_per_bank=4,
        n_qubits=4,
        ridge_alpha=1.0,
        feature_mode="phot+qrc",
    )


@pytest.fixture
def device():
    """Device fixture (cpu for testing)."""
    return "cpu"


@pytest.fixture
def tmp_output_dir():
    """Temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def small_sequence_data():
    """Small sequence data for quick tests."""
    np.random.seed(42)
    # Generate simple time series
    t = np.linspace(0, 100, 200)
    data = np.sin(2 * np.pi * t / 24) + 0.1 * np.random.randn(200)
    return data
