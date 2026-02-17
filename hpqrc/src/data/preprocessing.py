"""
Data Preprocessing Utilities

Normalization, feature engineering, and resampling.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


@dataclass
class Scaler:
    """Base scaler class."""
    pass


@dataclass
class StandardScaler:
    """Standard normalization (z-score)."""
    mean: np.ndarray
    std: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (self.std + 1e-8)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean


@dataclass
class MinMaxScaler:
    """Min-max normalization."""
    min: np.ndarray
    max: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.min) / (self.max - self.min + 1e-8)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * (self.max - self.min) + self.min


@dataclass
class RobustScaler:
    """Robust scaling using median and IQR."""
    median: np.ndarray
    iqr: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.median) / (self.iqr + 1e-8)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.iqr + self.median


def normalize(
    df: pd.DataFrame,
    method: Literal["standard", "minmax", "robust"] = "standard",
    columns: list | None = None,
) -> tuple[pd.DataFrame, Scaler]:
    """Normalize DataFrame columns.
    
    Args:
        df: Input DataFrame
        method: Normalization method
        columns: Columns to normalize (default: all numeric)
    
    Returns:
        (normalized_df, scaler)
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    if method == "standard":
        mean = df[columns].mean().values
        std = df[columns].std().values
        scaler = StandardScaler(mean, std)

        df_norm = df.copy()
        df_norm[columns] = (df[columns].values - mean) / (std + 1e-8)

    elif method == "minmax":
        min_val = df[columns].min().values
        max_val = df[columns].max().values
        scaler = MinMaxScaler(min_val, max_val)

        df_norm = df.copy()
        df_norm[columns] = (df[columns].values - min_val) / (max_val - min_val + 1e-8)

    elif method == "robust":
        median = df[columns].median().values
        q75 = df[columns].quantile(0.75).values
        q25 = df[columns].quantile(0.25).values
        iqr = q75 - q25
        scaler = RobustScaler(median, iqr)

        df_norm = df.copy()
        df_norm[columns] = (df[columns].values - median) / (iqr + 1e-8)

    return df_norm, scaler


def inverse_normalize(
    values: np.ndarray,
    scaler: Scaler,
) -> np.ndarray:
    """Inverse transform normalized values.
    
    Args:
        values: Normalized values
        scaler: Fitted scaler
    
    Returns:
        Original scale values
    """
    return scaler.inverse_transform(values)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features.
    
    Args:
        df: DataFrame with datetime index
    
    Returns:
        DataFrame with additional time features
    """
    df = df.copy()

    # Basic time features
    df["hour_of_day"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["day_of_month"] = df.index.day
    df["month"] = df.index.month
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)

    # Cyclical encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df


def resample(
    df: pd.DataFrame,
    target_freq: str,
    agg_method: str = "mean",
) -> pd.DataFrame:
    """Resample time series to target frequency.
    
    Args:
        df: Input DataFrame
        target_freq: Target frequency (e.g., '1H', '15T', '1D')
        agg_method: Aggregation method ('mean', 'sum', 'max', 'min')
    
    Returns:
        Resampled DataFrame
    """
    agg_map = {
        "mean": "mean",
        "sum": "sum",
        "max": "max",
        "min": "min",
    }

    return df.resample(target_freq).agg(agg_map[agg_method])


def create_sequences(
    data: np.ndarray,
    seq_len: int,
    horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Create input sequences and targets for forecasting.
    
    Args:
        data: Time series of shape (n_samples, n_features)
        seq_len: Input sequence length
        horizon: Forecast horizon
    
    Returns:
        (X, y) where X is (n_samples - seq_len - horizon + 1, seq_len, n_features)
                and y is (n_samples - seq_len - horizon + 1, horizon, n_features)
    """
    X, y = [], []

    for i in range(len(data) - seq_len - horizon + 1):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len:i + seq_len + horizon])

    return np.array(X), np.array(y)


def train_test_split(
    data: np.ndarray,
    test_ratio: float = 0.2,
    gap: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Split data into train and test.
    
    Args:
        data: Input array
        test_ratio: Fraction for test
        gap: Gap between train and test
    
    Returns:
        (train, test)
    """
    split_idx = int(len(data) * (1 - test_ratio))

    train = data[:split_idx]
    test = data[split_idx + gap:]

    return train, test
