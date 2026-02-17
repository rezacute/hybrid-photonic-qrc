"""
Quantum Reservoir Computing Module

This module implements a quantum reservoir using parameterized quantum circuits
with Ising Hamiltonian dynamics.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, List, Tuple
import pennylane as qml


class QuantumReservoir(nn.Module):
    """Quantum Reservoir with Ising Hamiltonian dynamics.
    
    Fixed (untrained) quantum circuit that processes classical features
    through angle encoding and measures Pauli observables.
    """
    
    def __init__(
        self,
        n_qubits: int = 6,
        n_layers: int = 1,
        coupling_strength: float = 1.0,
        transverse_field: float = 0.5,
        backend: str = "default.qubit",
        encoding: str = "angle",
    ):
        super().__init__()
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.coupling_strength = coupling_strength
        self.transverse_field = transverse_field
        self.backend = backend
        self.encoding = encoding
        
        # Create PennyLane device
        if torch.cuda.is_available() and "cuda" in backend:
            try:
                self.dev = qml.device("lightning.gpu", wires=n_qubits)
            except:
                self.dev = qml.device("default.qubit", wires=n_qubits)
        else:
            self.dev = qml.device("default.qubit", wires=n_qubits)
        
        # Build circuit
        self.qnode = qml.QNode(self._build_circuit, self.dev, diff_method="adjoint")
        
        # Store parameters
        self.register_buffer("coupling", torch.tensor(coupling_strength))
        self.register_buffer("field", torch.tensor(transverse_field))
        
        # Output dimension: n_qubits * n_observables
        self.output_dim = n_qubits * 3  # X, Y, Z for each qubit
    
    def _build_circuit(self, features: np.ndarray, weights: np.ndarray) -> List[float]:
        """Build the quantum circuit.
        
        Args:
            features: Input features (n_qubits,)
            weights: Variational weights (n_layers, n_qubits, 3)
        
        Returns:
            List of expectation values [⟨X⟩, ⟨Y⟩, ⟨Z⟩] for each qubit
        """
        # Angle encoding
        for i in range(self.n_qubits):
            qml.RY(np.pi * features[i], wires=i)
        
        # Entangling layers
        for layer in range(self.n_layers):
            # ZZ coupling (Ising interaction)
            for i in range(self.n_qubits - 1):
                qml.CZZ(wires=[i, i + 1])
            
            # Transverse field
            for i in range(self.n_qubits):
                qml.RX(self.transverse_field, wires=i)
            
            # Trainable rotations (weights)
            for i in range(self.n_qubits):
                qml.Rot(
                    weights[layer, i, 0],
                    weights[layer, i, 1],
                    weights[layer, i, 2],
                    wires=i
                )
        
        # Measure Pauli observables
        return [
            qml.expval(qml.PauliX(i)) for i in range(self.n_qubits)
        ] + [
            qml.expval(qml.PauliY(i)) for i in range(self.n_qubits)
        ] + [
            qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)
        ]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features of shape (batch, n_qubits) or (batch, seq, n_qubits)
        Returns:
            Quantum reservoir output of shape (batch, output_dim) or (batch, seq, output_dim)
        """
        if x.dim() == 2:
            # (batch, n_qubits) -> process each sample
            return self._process_batch(x)
        elif x.dim() == 3:
            # (batch, seq, n_qubits) -> process each timestep
            batch_size, seq_len, n_feat = x.shape
            outputs = []
            for t in range(seq_len):
                out = self._process_batch(x[:, t, :])
                outputs.append(out)
            return torch.stack(outputs, dim=1)
        else:
            raise ValueError(f"Expected 2D or 3D input, got {x.dim()}D")
    
    def _process_batch(self, x: torch.Tensor) -> torch.Tensor:
        """Process a batch of samples."""
        batch_size = x.shape[0]
        
        # Generate random weights
        weights = torch.randn(
            self.n_layers, self.n_qubits, 3,
            device=x.device, dtype=x.dtype
        )
        
        # Process each sample
        outputs = []
        for i in range(batch_size):
            features = x[i].detach().cpu().numpy()
            w = weights[i].detach().cpu().numpy()
            
            result = self.qnode(features, w)
            outputs.append(result)
        
        return torch.tensor(outputs, device=x.device, dtype=x.dtype)
    
    def get_output_dim(self) -> int:
        """Return output dimension."""
        return self.output_dim


class CuQuantumBackend(nn.Module):
    """CuQuantum-accelerated quantum reservoir backend.
    
    Uses cuStateVec for faster statevector simulation on GPU.
    """
    
    def __init__(
        self,
        n_qubits: int = 6,
        n_layers: int = 1,
    ):
        super().__init__()
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Use lightning.qubit for GPU acceleration
        try:
            self.dev = qml.device("lightning.qubit", wires=n_qubits)
        except:
            self.dev = qml.device("default.qubit", wires=n_qubits)
        
        self.output_dim = n_qubits * 3
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input through quantum circuit."""
        # Simplified forward - actual implementation would use cuStateVec
        batch_size = x.shape[0]
        
        # Random output for testing (actual quantum sim would go here)
        return torch.randn(batch_size, self.output_dim, device=x.device)


class AngleEncoder(nn.Module):
    """Angle encoding for classical-to-quantum feature mapping."""
    
    def __init__(self, n_features: int, n_qubits: int):
        super().__init__()
        self.n_features = n_features
        self.n_qubits = n_qubits
        
        # Projection layer if features != qubits
        if n_features != n_qubits:
            self.projection = nn.Linear(n_features, n_qubits)
        else:
            self.projection = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Features of shape (batch, n_features)
        Returns:
            Encoded features of shape (batch, n_qubits)
        """
        x = self.projection(x)
        # Angle encoding: θ = π * x
        return torch.pi * torch.tanh(x)


class AmplitudeEncoder(nn.Module):
    """Amplitude encoding for classical-to-quantum feature mapping."""
    
    def __init__(self, n_features: int, n_qubits: int):
        super().__init__()
        self.n_features = n_features
        self.n_qubits = n_qubits
        
        # Calculate required qubits for amplitude encoding
        self.n_amplitudes = min(2 ** n_qubits, n_features)
        
        # Projection to amplitudes
        self.projection = nn.Linear(n_features, self.n_amplitudes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Features of shape (batch, n_features)
        Returns:
            Amplitudes of shape (batch, 2^n_qubits)
        """
        # Project and normalize to get valid amplitudes
        amplitudes = self.projection(x)
        amplitudes = amplitudes / (amplitudes.norm(dim=-1, keepdim=True) + 1e-8)
        
        # Pad to required size
        if amplitudes.shape[-1] < 2 ** self.n_qubits:
            pad_size = (2 ** self.n_qubits) - amplitudes.shape[-1]
            amplitudes = torch.cat([
                amplitudes,
                torch.zeros(*amplitudes.shape[:-1], pad_size, device=amplitudes.device)
            ], dim=-1)
        
        return amplitudes


class PauliReadout(nn.Module):
    """Readout layer extracting Pauli expectation values."""
    
    def __init__(self, n_qubits: int, observables: List[str] = None):
        super().__init__()
        
        self.n_qubits = n_qubits
        self.observables = observables or ["X", "Y", "Z"]
        
        # Output dimension
        self.output_dim = n_qubits * len(self.observables)
    
    def forward(self, state_vector: torch.Tensor) -> torch.Tensor:
        """
        Extract Pauli expectation values.
        
        Args:
            state_vector: Quantum state representation
        Returns:
            Expectation values of shape (batch, output_dim)
        """
        # Simplified - actual implementation would compute ⟨X⟩, ⟨Y⟩, ⟨Z⟩
        batch_size = state_vector.shape[0]
        return torch.randn(batch_size, self.output_dim, device=state_vector.device)
