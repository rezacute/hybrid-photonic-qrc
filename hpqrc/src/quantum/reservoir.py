"""
Quantum Reservoir - Qiskit 2.x Compatible

Fixed-Hamiltonian quantum reservoir with Ising dynamics.
Follows Qiskit 2.x migration rules:
- Use SparsePauliOp from qiskit.quantum_info
- Use AerSimulator from qiskit_aer
- Use Statevector for expectation values
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, List, Tuple
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator


class QuantumReservoir(nn.Module):
    """Fixed-Hamiltonian quantum reservoir with Ising dynamics.
    
    Key features:
    - Fixed random J (ZZ couplings) and g (transverse fields) at init
    - Angle encoding of input features
    - Trotterized evolution
    - Readout via Pauli expectation values
    """
    
    def __init__(
        self,
        n_qubits: int = 8,
        coupling_range: Tuple[float, float] = (0.5, 1.5),
        field_range: Tuple[float, float] = (0.5, 1.5),
        evolution_time: float = 1.0,
        n_trotter_steps: int = 10,
        seed: int = 42,
    ):
        super().__init__()
        
        self.n_qubits = n_qubits
        self.coupling_range = coupling_range
        self.field_range = field_range
        self.evolution_time = evolution_time
        self.n_trotter_steps = n_trotter_steps
        
        # Set random seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Initialize fixed random parameters (not trainable)
        # J coupling strengths for ZZ interactions
        self.J = nn.Parameter(
            torch.tensor(
                np.random.uniform(coupling_range[0], coupling_range[1], n_qubits - 1)
            ),
            requires_grad=False
        )
        
        # Transverse field strengths
        self.g = nn.Parameter(
            torch.tensor(
                np.random.uniform(field_range[0], field_range[1], n_qubits)
            ),
            requires_grad=False
        )
        
        # Output dimension: 3 * n_qubits (X, Y, Z for each qubit)
        self.output_dim = 3 * n_qubits
        
        # Build observables using SparsePauliOp (Qiskit 2.x)
        self.observables = self._build_observables()
        
        # Create simulator
        self.simulator = AerSimulator(method='statevector')
        
        # Use Statevector for expectation values (no EstimatorV2 backend needed)
    
    def _build_observables(self) -> List[SparsePauliOp]:
        """Build Pauli observables using SparsePauliOp.
        
        Returns:
            List of SparsePauliOp for ⟨X_i⟩, ⟨Y_i⟩, ⟨Z_i⟩ for each qubit
        """
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
    
    def build_circuit(self, input_features: np.ndarray) -> QuantumCircuit:
        """Build quantum circuit with angle encoding and Ising evolution.
        
        Args:
            input_features: Array of shape (n_qubits,) - angles for encoding
        
        Returns:
            QuantumCircuit
        """
        qc = QuantumCircuit(self.n_qubits)
        
        # Initial superposition via H gates
        for i in range(self.n_qubits):
            qc.h(i)
        
        # Trotterized evolution
        dt = self.evolution_time / self.n_trotter_steps
        
        for step in range(self.n_trotter_steps):
            # ZZ coupling (Ising interaction) - use CNOT + RZ
            for i in range(self.n_qubits - 1):
                qc.cx(i, i + 1)
                qc.rz(self.J[i].item() * dt, i + 1)
                qc.cx(i, i + 1)
            
            # Transverse field - RX gates
            for i in range(self.n_qubits):
                qc.rx(self.g[i].item() * dt, i)
        
        # Final angle encoding - RZ gates on input features
        for i in range(min(len(input_features), self.n_qubits)):
            qc.rz(input_features[i], i)
        
        return qc
    
    def get_observables(self) -> List[SparsePauliOp]:
        """Return list of Pauli observables."""
        return self.observables
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input through quantum reservoir.
        
        Args:
            x: Input features of shape (batch, n_qubits) or (batch, seq, n_qubits)
        
        Returns:
            Expectation values of shape (batch, output_dim) or (batch, seq, output_dim)
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
        """Process a batch of samples using Statevector."""
        batch_size = x.shape[0]
        outputs = []
        
        for i in range(batch_size):
            # Get input features
            features = x[i].detach().cpu().numpy()
            
            # Build circuit
            circuit = self.build_circuit(features)
            
            # Get statevector and compute expectation values
            sv = Statevector(circuit)
            exp_values = []
            for obs in self.observables:
                exp_val = sv.expectation_value(obs).real
                exp_values.append(exp_val)
            
            outputs.append(exp_values)
        
        return torch.tensor(outputs, device=x.device, dtype=torch.float32)
    
    def get_output_dim(self) -> int:
        """Return output dimension."""
        return self.output_dim


def create_quantum_reservoir(
    n_qubits: int = 8,
    seed: int = 42,
    **kwargs
) -> QuantumReservoir:
    """Factory function to create quantum reservoir."""
    return QuantumReservoir(
        n_qubits=n_qubits,
        seed=seed,
        **kwargs
    )
