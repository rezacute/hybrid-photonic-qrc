"""
LSTM Forecaster for Time Series

2-layer LSTM for time series forecasting.
"""


import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    """2-layer LSTM for time series forecasting.
    
    Uses last hidden state for prediction.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        output_dim: int = 1,
    ):
        """
        Args:
            input_dim: Input feature dimension
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            output_dim: Output dimension
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_dim = output_dim

        # LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        # Output projection
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input of shape (batch, seq_len, input_dim)
        
        Returns:
            Predictions of shape (batch, output_dim)
        """
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(x)

        # Use last hidden state
        last_hidden = hidden[-1]  # (batch, hidden_size)

        # Project to output
        out = self.fc(last_hidden)  # (batch, output_dim)

        return out

    @property
    def n_params(self) -> int:
        """Number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MultiStepLSTM(nn.Module):
    """Multi-step LSTM forecaster."""

    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        horizon: int = 1,
    ):
        super().__init__()

        self.horizon = horizon

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict multiple steps ahead."""
        lstm_out, (hidden, cell) = self.lstm(x)
        last_hidden = hidden[-1]
        return self.fc(last_hidden)
