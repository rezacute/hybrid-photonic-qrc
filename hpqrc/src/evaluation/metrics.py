"""
Evaluation Metrics for Time Series Forecasting

Standard metrics: MAE, RMSE, MAPE, and utilities.
"""

import numpy as np
from typing import Dict, Optional


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
    
    Returns:
        MAE
    """
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
    
    Returns:
        RMSE
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    epsilon: float = 1e-8,
) -> float:
    """Mean Absolute Percentage Error.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        epsilon: Small value to avoid division by zero
    
    Returns:
        MAPE (%)
    """
    mask = np.abs(y_true) > epsilon
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric MAPE.
    
    Returns:
        sMAPE (%)
    """
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denominator > 1e-8
    return np.mean(numerator[mask] / denominator[mask]) * 100


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² (coefficient of determination).
    
    Returns:
        R² score
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / (ss_tot + 1e-8)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute all standard metrics.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
    
    Returns:
        Dictionary of metrics
    """
    # Flatten if needed
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def normalized_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Normalized RMSE (by standard deviation of true values).
    
    Returns:
        NRMSE (%)
    """
    return rmse(y_true, y_pred) / (np.std(y_true) + 1e-8) * 100


def coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Prediction interval coverage.
    
    Returns:
        Coverage (%)
    """
    covered = np.sum((y_true >= lower) & (y_true <= upper))
    return covered / len(y_true) * 100
