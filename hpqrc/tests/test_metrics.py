"""
Tests for evaluation metrics.
"""

import numpy as np
import pytest

from src.evaluation.metrics import (
    compute_all_metrics,
    mae,
    mape,
    r2_score,
    rmse,
    smape,
)


def test_mae():
    """Test MAE calculation."""
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1, 2, 4, 4, 5])

    result = mae(y_true, y_pred)

    # |1-1| + |2-2| + |3-4| + |4-4| + |5-5| = 0 + 0 + 1 + 0 + 0 = 1
    # MAE = 1/5 = 0.2
    assert result == pytest.approx(0.2)


def test_rmse():
    """Test RMSE calculation."""
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 2, 4])

    result = rmse(y_true, y_pred)

    # MSE = ((1-1)² + (2-2)² + (3-4)²) / 3 = (0 + 0 + 1) / 3 = 1/3
    # RMSE = sqrt(1/3) ≈ 0.577
    assert result == pytest.approx(np.sqrt(1/3), rel=1e-3)


def test_mape():
    """Test MAPE calculation."""
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1, 2, 4, 4, 5])

    result = mape(y_true, y_pred)

    # |1-1|/1 + |2-2|/2 + |3-4|/3 + |4-4|/4 + |5-5|/5 = 0 + 0 + 1/3 + 0 + 0
    # MAPE = (1/3) / 5 * 100 = 6.67%
    assert result == pytest.approx(100/15, rel=1e-3)


def test_smape():
    """Test symmetric MAPE."""
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1, 2, 4, 4, 5])

    result = smape(y_true, y_pred)

    # Should be positive
    assert result > 0


def test_r2_perfect():
    """Test R² with perfect prediction."""
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1, 2, 3, 4, 5])

    result = r2_score(y_true, y_pred)

    assert result == pytest.approx(1.0)


def test_r2_worst():
    """Test R² with constant prediction."""
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([3, 3, 3, 3, 3])
    
    result = r2_score(y_true, y_pred)
    
    # Should be 0 or negative (allowing small numerical error)
    assert result <= 0 + 1e-6


def test_compute_all_metrics():
    """Test compute_all_metrics returns all metrics."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])

    result = compute_all_metrics(y_true, y_pred)

    # Check all keys present
    assert "mae" in result
    assert "rmse" in result
    assert "mape" in result
    assert "smape" in result
    assert "r2" in result

    # All should be numeric
    assert isinstance(result["mae"], (int, float))
    assert isinstance(result["rmse"], (int, float))


def test_mae_zeros():
    """Test MAE with exact values."""
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 2, 3])

    result = mae(y_true, y_pred)

    assert result == 0.0


def test_rmse_zeros():
    """Test RMSE with exact values."""
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 2, 3])

    result = rmse(y_true, y_pred)

    assert result == 0.0
