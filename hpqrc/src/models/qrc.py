"""
Pure Quantum Reservoir Computing Model

QRC without photonic layer - baseline for comparison.
"""

import numpy as np
import torch
from typing import Optional

from ..quantum.reservoir import QuantumReservoir
from ..readout.ridge import RidgeReadout


class PureQRC:
    """Pure Quantum Reservoir Computing model.
    
    Uses quantum reservoir for temporal processing,
    with Ridge regression readout.
    """
    
    def __init__(
        self,
        n_qubits: int = 8,
        coupling_range: tuple = (0.5, 1.5),
        field_range: tuple = (0.5, 1.5),
        evolution_time: float = 1.0,
        n_trotter_steps: int = 10,
        ridge_alpha: float = 1.0,
        device: str = "cpu",
        seed: int = 42,
    ):
        """Initialize Pure QRC.
        
        Args:
            n_qubits: Number of qubits
            coupling_range: Range for J coupling
            field_range: Range for transverse field
            evolution_time: Evolution time
            n_trotter_steps: Number of Trotter steps
            ridge_alpha: Ridge regularization
            device: Device for quantum simulation
            seed: Random seed
        """
        self.n_qubits = n_qubits
        self.device = device
        self.seed = seed
        
        # Set seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Quantum reservoir
        self.quantum = QuantumReservoir(
            n_qubits=n_qubits,
            coupling_range=coupling_range,
            field_range=field_range,
            evolution_time=evolution_time,
            n_trotter_steps=n_trotter_steps,
            seed=seed,
        )
        
        # Readout
        self.readout = RidgeReadout(alpha=ridge_alpha)
        
        self._is_fitted = False
    
    def extract_features(self, x: np.ndarray) -> np.ndarray:
        """Extract quantum features from input.
        
        Args:
            x: Input of shape (batch, seq_len) or (batch, n_qubits)
        
        Returns:
            Feature array
        """
        # Ensure correct shape
        if x.ndim == 2:
            x = x[:, np.newaxis, :]  # (batch, 1, seq)
        
        # Use last n_qubits values as quantum input
        x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
        
        # Take last n_qubits time steps
        quantum_input = x_tensor[:, 0, -self.n_qubits:]
        
        with torch.no_grad():
            q_feat = self.quantum(quantum_input).cpu().numpy()
        
        return q_feat
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "PureQRC":
        """Fit the model.
        
        Args:
            X_train: Training input
            y_train: Training targets
        """
        # Extract features
        X_feat = self.extract_features(X_train)
        
        # Fit readout
        self.readout.fit(X_feat, y_train)
        
        self._is_fitted = True
        return self
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict on new data.
        
        Args:
            X_test: Test input
        
        Returns:
            Predictions
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")
        
        # Extract features
        X_feat = self.extract_features(X_test)
        
        # Predict
        return self.readout.predict(X_feat)
    
    @property
    def n_params(self) -> int:
        """Number of trainable parameters."""
        return self.readout.n_params


class QRCWithReadout:
    """QRC with trainable PyTorch readout for end-to-end training."""
    
    def __init__(
        self,
        n_qubits: int = 8,
        readout_dim: int = 64,
        device: str = "cpu",
        seed: int = 42,
    ):
        self.n_qubits = n_qubits
        self.device = device
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Quantum reservoir
        self.quantum = QuantumReservoir(n_qubits=n_qubits, seed=seed)
        
        # Trainable readout
        from ..readout.linear import TrainableLinearReadout
        self.readout = TrainableLinearReadout(
            input_dim=3 * n_qubits,
            output_dim=1,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        q_feat = self.quantum(x)
        return self.readout(q_feat)
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 100):
        """Train with backpropagation."""
        # Placeholder for full training
        pass
