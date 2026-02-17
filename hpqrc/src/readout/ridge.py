"""
Ridge Regression Readout

This module implements the Ridge Regression readout layer commonly used
in Reservoir Computing.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import Ridge
from typing import Optional


class RidgeReadout(nn.Module):
    """Ridge Regression Readout for Reservoir Computing.
    
    This is a non-differentiable readout that is fit using scikit-learn's
    Ridge regression. It's the standard approach in RC methods.
    """
    
    def __init__(self, input_dim: int, output_dim: int, alpha: float = 1.0):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.alpha = alpha
        
        # Ridge model (not a PyTorch module)
        self.ridge = Ridge(alpha=alpha, fit_intercept=True)
        self._is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeReadout":
        """Fit the Ridge regression model.
        
        Args:
            X: Input features of shape (n_samples, input_dim)
            y: Target values of shape (n_samples, output_dim)
        
        Returns:
            Self
        """
        self.ridge.fit(X, y)
        self._is_fitted = True
        return self
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict using the fitted Ridge model.
        
        Args:
            x: Input features of shape (batch, input_dim)
        
        Returns:
            Predictions of shape (batch, output_dim)
        """
        if not self._is_fitted:
            raise RuntimeError("RidgeReadout must be fitted before inference")
        
        with torch.no_grad():
            X = x.detach().cpu().numpy()
            y_pred = self.ridge.predict(X)
            return torch.from_numpy(y_pred).to(x.device)
    
    def get_params(self) -> dict:
        """Get model parameters."""
        return {
            "alpha": self.alpha,
            "coef_": self.ridge.coef_,
            "intercept_": self.ridge.intercept_,
        }


class TrainableLinearReadout(nn.Module):
    """Trainable linear readout layer.
    
    Alternative to Ridge regression that is fully differentiable and
    can be trained via backpropagation.
    """
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features of shape (batch, input_dim) or (batch, seq, input_dim)
        
        Returns:
            Predictions of shape (batch, output_dim) or (batch, seq, output_dim)
        """
        return self.linear(x)
