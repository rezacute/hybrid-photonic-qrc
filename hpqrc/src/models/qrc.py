"""
Pure Quantum Reservoir Computing Model

QRC without photonic layer - baseline for comparison.
"""

import numpy as np
import torch
from typing import Optional, Dict

from ..quantum.reservoir import QuantumReservoir
from ..readout.ridge import RidgeReadout


class PureQRC:
    """Pure Quantum Reservoir Computing model.
    
    Uses quantum reservoir for temporal processing,
    with Ridge regression readout.
    
    Supports multi-horizon prediction via separate readouts per horizon.
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
        self.ridge_alpha = ridge_alpha
        
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
        
        # Single readout (for horizon=1)
        self.readout = RidgeReadout(alpha=ridge_alpha)
        
        # Multi-horizon readouts (for horizon > 1)
        self.readouts: Dict[int, RidgeReadout] = {}
        
        self._is_fitted = False
        self._horizon = 1
    
    def extract_features(self, x: np.ndarray) -> np.ndarray:
        """Extract quantum features from input.
        
        Args:
            x: Input of shape (batch, seq_len) or (batch, n_qubits)
        
        Returns:
            Feature array of shape (batch, feature_dim)
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
            X_train: Training input of shape (n_samples, seq_len)
            y_train: Training targets of shape:
                - (n_samples,) for horizon=1
                - (n_samples, horizon) for horizon>1
        """
        # Extract features
        X_feat = self.extract_features(X_train)
        
        # Determine horizon from y_train shape
        if y_train.ndim == 2:
            horizon = y_train.shape[1]
        else:
            horizon = 1
            y_train = y_train.reshape(-1, 1)
        
        self._horizon = horizon
        
        if horizon == 1:
            # Single output - use main readout
            self.readout.fit(X_feat, y_train.flatten())
        else:
            # Multi-horizon - fit separate readout per horizon
            self.readouts = {}
            for h in range(horizon):
                readout = RidgeReadout(alpha=self.ridge_alpha)
                readout.fit(X_feat, y_train[:, h])
                self.readouts[h] = readout
        
        self._is_fitted = True
        return self
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict on new data.
        
        Args:
            X_test: Test input of shape (n_samples, seq_len)
        
        Returns:
            Predictions of shape:
                - (n_samples,) for horizon=1
                - (n_samples, horizon) for horizon>1
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")
        
        # Extract features
        X_feat = self.extract_features(X_test)
        
        if self._horizon == 1:
            # Single output
            return self.readout.predict(X_feat).flatten()
        else:
            # Multi-horizon
            predictions = np.zeros((len(X_test), self._horizon))
            for h in range(self._horizon):
                predictions[:, h] = self.readouts[h].predict(X_feat).flatten()
            return predictions
    
    @property
    def n_params(self) -> int:
        """Number of trainable parameters."""
        if self._horizon == 1:
            return self.readout.n_params
        else:
            return sum(r.n_params for r in self.readouts.values())


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
