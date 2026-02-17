"""
Trainable Linear Readout for Gradient-Based Fine-Tuning

PyTorch-based linear readout for end-to-end gradient-based optimization.
"""


import torch
import torch.nn as nn


class TrainableLinearReadout(nn.Module):
    """Optional PyTorch linear readout for gradient-based fine-tuning.
    
    Use this when you want to fine-tune the readout end-to-end with
    gradient descent rather than using Ridge regression.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        bias: bool = True,
    ):
        """
        Args:
            input_dim: Input feature dimension
            output_dim: Output dimension (1 for univariate forecasting)
            bias: Whether to include bias term
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.linear = nn.Linear(input_dim, output_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input of shape (batch, input_dim) or (batch, seq, input_dim)
        
        Returns:
            Output of shape (batch, output_dim) or (batch, seq, output_dim)
        """
        return self.linear(x)

    @property
    def n_params(self) -> int:
        """Number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_weights(self) -> torch.Tensor:
        """Get weight matrix."""
        return self.linear.weight

    def get_bias(self) -> torch.Tensor | None:
        """Get bias vector."""
        return self.linear.bias


class EnsembleReadout(nn.Module):
    """Ensemble of multiple readouts with soft voting."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        n_readouts: int = 3,
    ):
        super().__init__()

        self.readouts = nn.ModuleList([
            TrainableLinearReadout(input_dim, output_dim)
            for _ in range(n_readouts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Average predictions from all readouts."""
        outputs = [readout(x) for readout in self.readouts]
        return torch.mean(torch.stack(outputs), dim=0)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
