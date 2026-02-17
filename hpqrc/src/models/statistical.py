"""
Statistical Models for Time Series Forecasting

SARIMA and Persistence baselines.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple
from statsmodels.tsa.statespace.sarimax import SARIMAX


class SARIMAModel:
    """SARIMA model for time series forecasting.
    
    Uses statsmodels SARIMAX.
    """
    
    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 24),
        seasonal_period: int = 24,
    ):
        """
        Args:
            order: (p, d, q) ARIMA order
            seasonal_order: (P, D, Q, s) seasonal order
            seasonal_period: Seasonal period (e.g., 24 for hourly data)
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.seasonal_period = seasonal_period
        self.model = None
        self.results = None
    
    def fit(self, series: pd.Series) -> "SARIMAModel":
        """Fit SARIMA model.
        
        Args:
            series: Time series data
        
        Returns:
            self
        """
        self.model = SARIMAX(
            series,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        
        # Fit with limited iterations for speed
        self.results = self.model.fit(disp=False, maxiter=100)
        
        return self
    
    def predict(self, horizon: int) -> np.ndarray:
        """Predict future values.
        
        Args:
            horizon: Number of steps to predict
        
        Returns:
            Predictions of shape (horizon,)
        """
        if self.results is None:
            raise RuntimeError("Model not fitted")
        
        forecast = self.results.forecast(steps=horizon)
        return forecast.values
    
    @property
    def aic(self) -> Optional[float]:
        """Akaike Information Criterion."""
        if self.results is None:
            return None
        return self.results.aic


class PersistenceModel:
    """Persistence model (naive baseline).
    
    Predicts the value from the same time in the previous period.
    """
    
    def __init__(self, period: int = 24):
        """
        Args:
            period: Seasonal period (e.g., 24 for hourly data)
        """
        self.period = period
    
    def fit(self, series: pd.Series) -> "PersistenceModel":
        """Fit (no-op for persistence)."""
        return self
    
    def predict(self, horizon: int, history: np.ndarray) -> np.ndarray:
        """Predict using persistence.
        
        Args:
            horizon: Number of steps to predict
            history: Historical data
        
        Returns:
            Predictions
        """
        # Use last 'period' values repeatedly
        n_history = len(history)
        
        if n_history < self.period:
            # Not enough history, use last value
            return np.full(horizon, history[-1])
        
        predictions = []
        for h in range(horizon):
            # Index h steps back
            idx = n_history - self.period + (h % self.period)
            predictions.append(history[idx])
        
        return np.array(predictions)


class SeasonalNaive:
    """Seasonal naive model - repeat last season."""
    
    def __init__(self, period: int = 24):
        self.period = period
    
    def fit(self, series: pd.Series) -> "SeasonalNaive":
        return self
    
    def predict(self, horizon: int) -> np.ndarray:
        """Predict by repeating last season."""
        # This is simplified - actual implementation would use the series
        raise NotImplementedError("Use predict_with_history()")


class MovingAverage:
    """Simple moving average baseline."""
    
    def __init__(self, window: int = 24):
        self.window = window
    
    def fit(self, series: pd.Series) -> "MovingAverage":
        return self
    
    def predict(self, horizon: int) -> np.ndarray:
        """Predict as flat line of recent average."""
        raise NotImplementedError("Use predict_with_history()")


class ExponentialSmoothing:
    """Exponential smoothing baseline."""
    
    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
    
    def fit(self, series: pd.Series) -> "ExponentialSmoothing":
        return self
    
    def predict(self, horizon: int) -> np.ndarray:
        raise NotImplementedError("Use predict_with_history()")
