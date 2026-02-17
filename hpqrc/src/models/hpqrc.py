"""
HPQRC Model - Full Hybrid Photonic-Quantum Reservoir Computing Pipeline

Orchestrates photonic delay loops, quantum reservoir, and readout.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from ..photonic.delay_loop import PhotonicDelayLoopEmulator
from ..quantum.reservoir import QuantumReservoir
from ..readout.ridge import RidgeReadout


@dataclass
class HPQRCConfig:
    """Configuration for HPQRC model."""
    # Photonic layer
    in_channels: int = 1
    n_banks: int = 5
    kernel_sizes: list = None
    features_per_bank: int = 16
    activation: str = "tanh"
    frozen_kernels: bool = False
    dropout: float = 0.0

    # Quantum layer
    n_qubits: int = 8
    coupling_range: tuple = (0.5, 1.5)
    field_range: tuple = (0.5, 1.5)
    evolution_time: float = 1.0
    n_trotter_steps: int = 10

    # Readout
    ridge_alpha: float = 1.0

    # Feature mode
    feature_mode: Literal["phot+qrc", "qrc-only", "phot-only"] = "phot+qrc"

    def __post_init__(self):
        if self.kernel_sizes is None:
            self.kernel_sizes = [4, 24, 96, 168, 672]


class HPQRC:
    """Full Hybrid Photonic-Quantum Reservoir Computing pipeline.
    
    Combines:
    - Photonic Delay-Loop Emulation (PDEL)
    - Quantum Reservoir Computing (QRC)
    - Ridge Regression Readout
    """

    def __init__(
        self,
        config: HPQRCConfig,
        device: str = "cuda",
        seed: int = 42,
    ):
        self.config = config
        self.device = device
        self.seed = seed

        # Set seeds
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Initialize components
        self._init_photonic()
        self._init_quantum()
        self._init_readout()

        self._is_fitted = False

    def _init_photonic(self):
        """Initialize photonic delay-loop emulator."""
        self.photonic = PhotonicDelayLoopEmulator(
            in_channels=self.config.in_channels,
            n_banks=self.config.n_banks,
            kernel_sizes=self.config.kernel_sizes,
            features_per_bank=self.config.features_per_bank,
            activation=self.config.activation,
            frozen_kernels=self.config.frozen_kernels,
            dropout=self.config.dropout,
        ).to(self.device)

        self.photonic.eval()  # No training for photonic

    def _init_quantum(self):
        """Initialize quantum reservoir."""
        self.quantum = QuantumReservoir(
            n_qubits=self.config.n_qubits,
            coupling_range=self.config.coupling_range,
            field_range=self.config.field_range,
            evolution_time=self.config.evolution_time,
            n_trotter_steps=self.config.n_trotter_steps,
            seed=self.seed,
        ).to(self.device)

        self.quantum.eval()  # No training for quantum

    def _init_readout(self):
        """Initialize Ridge readout."""
        self.readout = RidgeReadout(alpha=self.config.ridge_alpha)

    def extract_features(self, x_batch: np.ndarray) -> np.ndarray:
        """Extract features from input using photonic + quantum layers.
        
        Args:
            x_batch: Input of shape (batch, seq_len) or (batch, 1, seq_len)
        
        Returns:
            Feature array of shape (batch, feature_dim)
        """
        # Ensure correct shape
        if x_batch.ndim == 2:
            x_batch = x_batch[:, np.newaxis, :]  # (batch, 1, seq)

        x_tensor = torch.tensor(x_batch, dtype=torch.float32, device=self.device)

        features_list = []

        # Photonic features
        if self.config.feature_mode in ["phot+qrc", "phot-only"]:
            with torch.no_grad():
                phot_out = self.photonic(x_tensor)  # (batch, phot_dim, seq)
                # Take last time step
                phot_feat = phot_out[:, :, -1].cpu().numpy()
                features_list.append(phot_feat)

        # Quantum features
        if self.config.feature_mode in ["phot+qrc", "qrc-only"]:
            # Need to prepare input for quantum
            # Use photonic output or raw input
            if self.config.feature_mode == "qrc-only":
                # Use last value as quantum input
                quantum_input = x_tensor[:, 0, -self.config.n_qubits:].cpu().numpy()
            else:
                # Use photonic output summary
                with torch.no_grad():
                    phot_out = self.photonic(x_tensor)
                    # Average pool over time
                    quantum_input = phot_out.mean(dim=2).cpu().numpy()
                    # Take last n_qubits
                    quantum_input = quantum_input[:, :self.config.n_qubits]

            # Run quantum reservoir
            q_input = torch.tensor(quantum_input, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                q_feat = self.quantum(q_input).cpu().numpy()
            features_list.append(q_feat)

        # Concatenate features
        features = np.concatenate(features_list, axis=1)

        return features

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "HPQRC":
        """Fit the model on training data.
        
        Args:
            X_train: Training input of shape (n_samples, seq_len)
            y_train: Training targets of shape (n_samples,) or (n_samples, n_targets)
        
        Returns:
            self
        """
        # Extract features
        print("Extracting features...")
        X_feat = self.extract_features(X_train)

        # Fit readout
        print(f"Fitting readout on {X_feat.shape[0]} samples, {X_feat.shape[1]} features...")
        self.readout.fit(X_feat, y_train)

        self._is_fitted = True
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict on new data.
        
        Args:
            X_test: Test input of shape (n_samples, seq_len)
        
        Returns:
            Predictions of shape (n_samples,) or (n_samples, n_targets)
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Extract features
        X_feat = self.extract_features(X_test)

        # Predict
        return self.readout.predict(X_feat)

    @property
    def total_params(self) -> int:
        """Total number of parameters in the model."""
        return self.n_photonic_params + self.n_quantum_params + self.readout.n_params

    @property
    def trainable_params(self) -> int:
        """Number of trainable parameters (readout only)."""
        return self.readout.n_params

    @property
    def n_photonic_params(self) -> int:
        """Number of parameters in photonic layer."""
        return self.photonic.n_trainable_params

    @property
    def n_quantum_params(self) -> int:
        """Number of parameters in quantum layer (fixed, not trainable)."""
        return 0  # Quantum reservoir has no trainable params

    def save(self, path: Path):
        """Save model state."""
        state = {
            "config": self.config,
            "readout_coef": self.readout.coef_,
            "readout_intercept": self.readout.intercept_,
        }
        np.savez(path, **state)

    def load(self, path: Path):
        """Load model state."""
        data = np.load(path)
        # Reconstruct readout
        # (simplified - full implementation would reconstruct model)


class PureQRC:
    """Quantum Reservoir Computing only (no photonic layer)."""

    def __init__(
        self,
        n_qubits: int = 8,
        coupling_range: tuple = (0.5, 1.5),
        field_range: tuple = (0.5, 1.5),
        evolution_time: float = 1.0,
        n_trotter_steps: int = 10,
        ridge_alpha: float = 1.0,
        device: str = "cuda",
        seed: int = 42,
    ):
        self.n_qubits = n_qubits
        self.device = device

        # Initialize quantum
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.quantum = QuantumReservoir(
            n_qubits=n_qubits,
            coupling_range=coupling_range,
            field_range=field_range,
            evolution_time=evolution_time,
            n_trotter_steps=n_trotter_steps,
            seed=seed,
        ).to(device)
        self.quantum.eval()

        self.readout = RidgeReadout(alpha=ridge_alpha)
        self._is_fitted = False

    def extract_features(self, x: np.ndarray) -> np.ndarray:
        """Extract quantum features."""
        if x.ndim == 2:
            x = x[:, np.newaxis, :]  # (batch, 1, seq)

        # Use last n_qubits values as quantum input
        x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
        quantum_input = x_tensor[:, 0, -self.n_qubits:]

        with torch.no_grad():
            q_feat = self.quantum(quantum_input).cpu().numpy()

        return q_feat

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "PureQRC":
        X_feat = self.extract_features(X_train)
        self.readout.fit(X_feat, y_train)
        self._is_fitted = True
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")
        X_feat = self.extract_features(X_test)
        return self.readout.predict(X_feat)


class PhotonicRC:
    """Photonic Delay-Loop + Readout (no quantum layer)."""

    def __init__(
        self,
        in_channels: int = 1,
        n_banks: int = 5,
        kernel_sizes: list = None,
        features_per_bank: int = 16,
        activation: str = "tanh",
        ridge_alpha: float = 1.0,
        device: str = "cuda",
        seed: int = 42,
    ):
        self.device = device
        if kernel_sizes is None:
            kernel_sizes = [4, 24, 96, 168, 672]

        torch.manual_seed(seed)

        self.photonic = PhotonicDelayLoopEmulator(
            in_channels=in_channels,
            n_banks=n_banks,
            kernel_sizes=kernel_sizes,
            features_per_bank=features_per_bank,
            activation=activation,
        ).to(device)
        self.photonic.eval()

        self.readout = RidgeReadout(alpha=ridge_alpha)
        self._is_fitted = False

    def extract_features(self, x: np.ndarray) -> np.ndarray:
        """Extract photonic features."""
        if x.ndim == 2:
            x = x[:, np.newaxis, :]

        x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            phot_out = self.photonic(x_tensor)
            phot_feat = phot_out[:, :, -1].cpu().numpy()

        return phot_feat

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "PhotonicRC":
        X_feat = self.extract_features(X_train)
        self.readout.fit(X_feat, y_train)
        self._is_fitted = True
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")
        X_feat = self.extract_features(X_test)
        return self.readout.predict(X_feat)
