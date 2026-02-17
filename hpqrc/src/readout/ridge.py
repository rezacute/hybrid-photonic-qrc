"""
Ridge Regression Readout for Reservoir Computing

Simple sklearn-based Ridge regression wrapper for reservoir readout.
"""


import numpy as np
from sklearn.linear_model import Ridge


class RidgeReadout:
    """Sklearn Ridge regression wrapper for reservoir readout.
    
    Simple, fast, and effective for reservoir computing.
    """

    def __init__(self, alpha: float = 1.0):
        """
        Args:
            alpha: Regularization strength (higher = more regularization)
        """
        self.alpha = alpha
        self.model = Ridge(alpha=alpha, fit_intercept=True)
        self._is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeReadout":
        """Fit the Ridge model.
        
        Args:
            X: Features of shape (n_samples, n_features)
            y: Targets of shape (n_samples,) or (n_samples, n_targets)
        
        Returns:
            self
        """
        self.model.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using fitted model.
        
        Args:
            X: Features of shape (n_samples, n_features)
        
        Returns:
            Predictions of shape (n_samples,) or (n_samples, n_targets)
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.model.predict(X)

    @property
    def n_params(self) -> int:
        """Number of parameters (weights + bias)."""
        if not self._is_fitted:
            return 0
        n_features = self.model.coef_.shape[-1]
        n_outputs = self.model.coef_.shape[0] if self.model.coef_.ndim > 1 else 1
        return n_features * n_outputs + n_outputs  # weights + bias

    @property
    def coef_(self) -> np.ndarray:
        """Model coefficients."""
        return self.model.coef_

    @property
    def intercept_(self) -> np.ndarray:
        """Model intercept."""
        return self.model.intercept_
