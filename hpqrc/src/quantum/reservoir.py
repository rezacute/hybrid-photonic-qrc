"""
Quantum Reservoir Computing Module - Qiskit 2.x Compatible

This module implements a quantum reservoir using Qiskit 2.x APIs.
Following Qiskit 2.x migration rules:
- Use V2 primitives (EstimatorV2, SamplerV2)
- Use SparsePauliOp for observables
- Use AerSimulator from qiskit_aer
- Explicit shots parameter in primitives
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, List, Tuple
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2


class QuantumReservoirQiskit(nn.Module):
    """Quantum Reservoir using Qiskit 2.x APIs.
    
    Uses:
    - AerSimulator for simulation
    - EstimatorV2 for expectation values
    - SparsePauliOp for observables
    """
    
    def __init__(
        self,
        n_qubits: int = 6,
        n_layers: int = 1,
        coupling_strength: float = 1.0,
        transverse_field: float = 0.5,
        use_gpu: bool = False,
        shots: Optional[int] = 1024,
    ):
        super().__init__()
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.coupling_strength = coupling_strength
        self.transverse_field = transverse_field
        self.use_gpu = use_gpu
        self.shots = shots
        
        # Create simulator - Qiskit 2.x pattern
        if use_gpu:
            try:
                self.simulator = AerSimulator(
                    method='statevector',
                    device='GPU'
                )
            except:
                # Fallback to CPU
                self.simulator = AerSimulator(method='statevector')
        else:
            self.simulator = AerSimulator(method='statevector')
        
        # Build observables - Qiskit 2.x uses SparsePauliOp
        self.observables = self._build_observables()
        
        # Create estimator - Qiskit 2.x V2 primitive
        self.estimator = EstimatorV2(
            backend=self.simulator,
            options={"shots": shots} if shots else {}
        )
        
        # Output dimension: n_qubits * 3 (X, Y, Z per qubit)
        self.output_dim = n_qubits * 3
        
        # Register non-trainable parameters
        self.register_buffer("coupling", torch.tensor(coupling_strength))
        self.register_buffer("field", torch.tensor(transverse_field))
    
    def _build_observables(self) -> List[SparsePauliOp]:
        """Build Pauli observables using SparsePauliOp."""
        observables = []
        
        for i in range(self.n_qubits):
            # X observable on qubit i
            pauli_x = "I" * i + "X" + "I" * (self.n_qubits - i - 1)
            observables.append(SparsePauliOp(pauli_x))
            
            # Y observable on qubit i
            pauli_y = "I" * i + "Y" + "I" * (self.n_qubits - i - 1)
            observables.append(SparsePauliOp(pauli_y))
            
            # Z observable on qubit i
            pauli_z = "I" * i + "Z" + "I" * (self.n_qubits - i - 1)
            observables.append(SparsePauliOp(pauli_z))
        
        return observables
    
    def _build_circuit(self, features: np.ndarray) -> QuantumCircuit:
        """Build quantum circuit with angle encoding.
        
        Args:
            features: Input features of shape (n_qubits,)
        
        Returns:
            QuantumCircuit
        """
        qc = QuantumCircuit(self.n_qubits)
        
        # Angle encoding - RY gates
        for i in range(self.n_qubits):
            qc.ry(np.pi * features[i], i)
        
        # Entangling layers - Qiskit 2.x style
        for layer in range(self.n_layers):
            # ZZ coupling (Ising interaction)
            for i in range(self.n_qubits - 1):
                qc.czz(i, i + 1)
            
            # Transverse field - RX gates
            for i in range(self.n_qubits):
                qc.rx(self.transverse_field, i)
        
        return qc
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input through quantum reservoir.
        
        Args:
            x: Input features of shape (batch, n_qubits) or (batch, seq, n_qubits)
        
        Returns:
            Expectation values of shape (batch, output_dim)
        """
        if x.dim() == 2:
            return self._process_batch(x)
        elif x.dim() == 3:
            batch_size, seq_len, _ = x.shape
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
        outputs = []
        
        for i in range(batch_size):
            features = x[i].detach().cpu().numpy()
            
            # Build circuit
            qc = self._build_circuit(features)
            
            # Run simulation - Qiskit 2.x V2 primitive pattern
            job = self.estimator.run([(qc, self.observables)])
            result = job.result()
            
            # Extract expectation values
            exp_values = result[0].data.evs
            outputs.append(exp_values)
        
        return torch.tensor(outputs, device=x.device, dtype=torch.float32)
    
    def get_output_dim(self) -> int:
        """Return output dimension."""
        return self.output_dim


class AngleEncoder(nn.Module):
    """Angle encoding for classical-to-quantum feature mapping - Qiskit 2.x."""
    
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
        """Angle encoding: θ = π * tanh(x)"""
        x = self.projection(x)
        return torch.pi * torch.tanh(x)


class AmplitudeEncoder(nn.Module):
    """Amplitude encoding - Qiskit 2.x."""
    
    def __init__(self, n_features: int, n_qubits: int):
        super().__init__()
        self.n_features = n_features
        self.n_qubits = n_qubits
        self.n_amplitudes = min(2 ** n_qubits, n_features)
        
        self.projection = nn.Linear(n_features, self.n_amplitudes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Amplitude encoding with normalization."""
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


class IsingReservoir(nn.Module):
    """Ising Hamiltonian quantum reservoir - Qiskit 2.x optimized."""
    
    def __init__(
        self,
        n_qubits: int = 6,
        n_layers: int = 1,
        j_coupling: float = 1.0,
        h_field: float = 0.5,
    ):
        super().__init__()
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.j_coupling = j_coupling
        self.h_field = h_field
        
        # CPU simulator for speed
        self.simulator = AerSimulator(method='statevector')
        
        # Observables
        self.observables = self._build_observables()
        
        # Estimator - Qiskit 2.x V2
        self.estimator = EstimatorV2(
            backend=self.simulator,
            options={"shots": 1024}
        )
        
        self.output_dim = n_qubits * 3
    
    def _build_observables(self) -> List[SparsePauliOp]:
        """Build observables using SparsePauliOp."""
        obs = []
        for i in range(self.n_qubits):
            for pauli in ['X', 'Y', 'Z']:
                pauli_str = "I" * i + pauli + "I" * (self.n_qubits - i - 1)
                obs.append(SparsePauliOp(pauli_str))
        return obs
    
    def _build_ising_circuit(self, features: np.ndarray) -> QuantumCircuit:
        """Build Ising Hamiltonian circuit."""
        qc = QuantumCircuit(self.n_qubits)
        
        # Initial encoding
        for i in range(self.n_qubits):
            qc.ry(np.pi * np.tanh(features[i]), i)
        
        # Ising layers
        for _ in range(self.n_layers):
            # ZZ couplings
            for i in range(self.n_qubits - 1):
                qc.czz(i, i + 1)
            
            # Transverse field
            for i in range(self.n_qubits):
                qc.rx(self.h_field, i)
        
        return qc
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        batch_size = x.shape[0]
        results = []
        
        for i in range(batch_size):
            feat = x[i].detach().cpu().numpy()
            qc = self._build_ising_circuit(feat)
            
            job = self.estimator.run([(qc, self.observables)])
            exp_vals = job.result()[0].data.evs
            results.append(exp_vals)
        
        return torch.tensor(results, device=x.device)


def create_quantum_reservoir(
    n_qubits: int = 6,
    backend: str = "cpu",
    **kwargs
) -> QuantumReservoirQiskit:
    """Factory function to create quantum reservoir.
    
    Args:
        n_qubits: Number of qubits
        backend: "cpu" or "gpu"
        **kwargs: Additional arguments
    
    Returns:
        QuantumReservoirQiskit instance
    """
    use_gpu = (backend == "gpu")
    return QuantumReservoirQiskit(
        n_qubits=n_qubits,
        use_gpu=use_gpu,
        **kwargs
    )
