"""
CuQuantum Backend - Qiskit 2.x Compatible

GPU-accelerated statevector simulation via Aer GPU or cuQuantum cuStateVec.

IMPORTANT: 
- Do NOT use qiskit-aer-gpu package (incompatible with Qiskit 2.x)
- Use qiskit-aer built from source with CUDA support
- Or use cuquantum-python-cu12 for standalone cuStateVec

Qiskit 2.x pattern:
- Use AerSimulator from qiskit_aer (NOT qiskit.providers.aer)
- Use EstimatorV2 from qiskit_aer.primitives
"""

import numpy as np
import torch
from typing import List, Optional, Dict, Any
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2


class CuQuantumBackend:
    """GPU-accelerated statevector simulation via Aer GPU or cuQuantum.
    
    Primary path: AerSimulator(method='statevector', device='GPU')
    Fallback: AerSimulator(method='statevector') on CPU
    
    IMPORTANT: qiskit-aer-gpu is INCOMPATIBLE with Qiskit 2.x
    """
    
    def __init__(
        self,
        n_qubits: int,
        device: str = "cuda",
        use_custatevec: bool = True,
    ):
        self.n_qubits = n_qubits
        self.device = device
        self.use_custatevec = use_custatevec
        
        # Try GPU simulation
        self.gpu_available = False
        self.simulator = None
        self.estimator = None
        
        self._init_gpu()
        
        if not self.gpu_available:
            print("WARNING: GPU not available, falling back to CPU")
            self._init_cpu()
    
    def _init_gpu(self):
        """Initialize GPU simulator."""
        try:
            # Qiskit 2.x: AerSimulator with GPU
            self.simulator = AerSimulator(
                method='statevector',
                device='GPU',
            )
            
            # Try to enable cuStateVec if requested
            if self.use_custatevec:
                try:
                    # This requires Aer built with cuStateVec support
                    self.simulator.set_option(cuStateVec_enable=True)
                except Exception as e:
                    print(f"cuStateVec not available: {e}")
            
            # Create estimator (Qiskit 2.x V2)
            self.estimator = EstimatorV2(
                backend=self.simulator,
                options={"shots": None}
            )
            
            self.gpu_available = True
            print(f"GPU simulation enabled: {self.simulator.device}")
            
        except Exception as e:
            print(f"GPU initialization failed: {e}")
            self.gpu_available = False
    
    def _init_cpu(self):
        """Initialize CPU simulator as fallback."""
        self.simulator = AerSimulator(method='statevector')
        self.estimator = EstimatorV2(
            backend=self.simulator,
            options={"shots": None}
        )
        print("Using CPU simulation")
    
    def run_circuit(self, circuit: QuantumCircuit) -> Statevector:
        """Run single circuit and return statevector.
        
        Args:
            circuit: QuantumCircuit
        
        Returns:
            Statevector
        """
        job = self.simulator.run(circuit)
        result = job.result()
        return result.get_statevector()
    
    def run_batch(self, circuits: List[QuantumCircuit]) -> List[Statevector]:
        """Run multiple circuits and return statevectors.
        
        Args:
            circuits: List of QuantumCircuit
        
        Returns:
            List of Statevector
        """
        results = []
        for circuit in circuits:
            sv = self.run_circuit(circuit)
            results.append(sv)
        return results
    
    def compute_expectations(
        self,
        circuit: QuantumCircuit,
        observables: List[SparsePauliOp],
    ) -> np.ndarray:
        """Compute expectation values via EstimatorV2.
        
        Qiskit 2.x pattern for hardware-compatible computation.
        
        Args:
            circuit: QuantumCircuit
            observables: List of SparsePauliOp
        
        Returns:
            Array of expectation values
        """
        if self.estimator is None:
            # Fallback to statevector
            sv = self.run_circuit(circuit)
            exp_vals = []
            for obs in observables:
                exp_vals.append(sv.expectation_value(obs))
            return np.array(exp_vals)
        
        # Use EstimatorV2 (Qiskit 2.x)
        job = self.estimator.run([(circuit, observables)])
        result = job.result()
        
        return result[0].data.evs
    
    def compute_expectations_batch(
        self,
        circuits: List[QuantumCircuit],
        observables: List[SparsePauliOp],
    ) -> np.ndarray:
        """Batch compute expectation values.
        
        Args:
            circuits: List of QuantumCircuit
            observables: List of SparsePauliOp
        
        Returns:
            Array of shape (n_circuits, n_observables)
        """
        if self.estimator is None:
            # Fallback
            results = []
            for circuit in circuits:
                exp_vals = []
                sv = self.run_circuit(circuit)
                for obs in observables:
                    exp_vals.append(sv.expectation_value(obs))
                results.append(exp_vals)
            return np.array(results)
        
        # Batched execution with EstimatorV2 (Qiskit 2.x PUB format)
        pubs = [(qc, observables) for qc in circuits]
        job = self.estimator.run(pubs)
        result = job.result()
        
        outputs = []
        for i in range(len(circuits)):
            outputs.append(result[i].data.evs)
        
        return np.array(outputs)
    
    @property
    def is_gpu(self) -> bool:
        """Check if GPU is available."""
        return self.gpu_available
    
    def __repr__(self) -> str:
        return f"CuQuantumBackend(n_qubits={self.n_qubits}, gpu={self.gpu_available})"


def create_cuquantum_backend(
    n_qubits: int,
    device: str = "cuda",
    **kwargs
) -> CuQuantumBackend:
    """Factory function to create CuQuantum backend."""
    return CuQuantumBackend(
        n_qubits=n_qubits,
        device=device,
        **kwargs
    )
