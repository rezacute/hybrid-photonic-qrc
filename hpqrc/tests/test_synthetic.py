"""
Tests for synthetic data generation.
"""

import pytest
import numpy as np
from scipy import signal

from src.data.synthetic import generate_synthetic_ev_demand, SyntheticConfig


def test_synthetic_output_shape():
    """Test synthetic data output shape."""
    config = SyntheticConfig(n_samples=1000, seed=42)
    df = generate_synthetic_ev_demand(config)
    
    assert len(df) == 1000
    assert "demand" in df.columns


def test_synthetic_demand_positive():
    """Test that demand is non-negative."""
    config = SyntheticConfig(n_samples=500, seed=42)
    df = generate_synthetic_ev_demand(config)
    
    assert (df["demand"] >= 0).all()


def test_synthetic_has_components():
    """Test that generated data has component columns."""
    config = SyntheticConfig(n_samples=500, seed=42)
    df = generate_synthetic_ev_demand(config)
    
    expected_cols = ["daily", "weekly", "yearly", "noise"]
    for col in expected_cols:
        assert col in df.columns


def test_synthetic_fft_peaks():
    """Test that FFT shows peaks at expected frequencies."""
    config = SyntheticConfig(n_samples=10000, resolution_minutes=15, seed=42)
    df = generate_synthetic_ev_demand(config)
    
    # Compute FFT
    fs = 96  # 96 samples per day (15-min resolution)
    f, psd = signal.welch(df["demand"].values, fs=fs, nperseg=1024)
    
    # Find peaks
    peaks, _ = signal.find_peaks(psd, height=np.max(psd) * 0.1)
    peak_freqs = f[peaks]
    
    # Expected frequencies (cycles per day)
    # Daily: 1/24 = 0.0417, Weekly: 1/168 = 0.006
    daily_period = 24
    weekly_period = 168
    
    # Check for peak near daily frequency
    daily_freq = 1 / daily_period
    has_daily_peak = any(abs(p - daily_freq) / daily_freq < 0.2 for p in peak_freqs)
    
    # Check for peak near weekly frequency  
    weekly_freq = 1 / weekly_period
    has_weekly_peak = any(abs(p - weekly_freq) / weekly_freq < 0.2 for p in peak_freqs)
    
    # At least one should be present
    assert has_daily_peak or has_weekly_peak


def test_synthetic_reproducibility():
    """Test that same seed produces same data."""
    config1 = SyntheticConfig(n_samples=500, seed=42)
    config2 = SyntheticConfig(n_samples=500, seed=42)
    
    df1 = generate_synthetic_ev_demand(config1)
    df2 = generate_synthetic_ev_demand(config2)
    
    np.testing.assert_array_almost_equal(
        df1["demand"].values,
        df2["demand"].values
    )


def test_synthetic_different_seeds():
    """Test that different seeds produce different data."""
    config1 = SyntheticConfig(n_samples=500, seed=42)
    config2 = SyntheticConfig(n_samples=500, seed=123)
    
    df1 = generate_synthetic_ev_demand(config1)
    df2 = generate_synthetic_ev_demand(config2)
    
    # Should be different
    assert not np.allclose(df1["demand"].values, df2["demand"].values)
