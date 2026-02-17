"""
Pauli Observables - Qiskit 2.x Compatible

Extract expectation values from quantum states using:
- Statevector (exact)
- EstimatorV2 (hardware-compatible)
"""


import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2


class PauliReadout:
    """Extract expectation values ⟨X_i⟩, ⟨Y_i⟩, ⟨Z_i⟩ for each qubit."""

    def __init__(
        self,
        n_qubits: int,
        use_estimator: bool = True,
    ):
        self.n_qubits = n_qubits
        self.use_estimator = use_estimator

        # Build observables using SparsePauliOp (Qiskit 2.x)
        self.observables = self._build_observables()

        # Create simulator and estimator if needed
        if use_estimator:
            self.simulator = AerSimulator(method='statevector')
            self.estimator = EstimatorV2(
                backend=self.simulator,
                options={"shots": None}  # Exact
            )

    def _build_observables(self) -> list[SparsePauliOp]:
        """Build Pauli observables using SparsePauliOp."""
        observables = []

        for i in range(self.n_qubits):
            # X observable
            pauli_x = "I" * i + "X" + "I" * (self.n_qubits - i - 1)
            observables.append(SparsePauliOp(pauli_x))

            # Y observable
            pauli_y = "I" * i + "Y" + "I" * (self.n_qubits - i - 1)
            observables.append(SparsePauliOp(pauli_y))

            # Z observable
            pauli_z = "I" * i + "Z" + "I" * (self.n_qubits - i - 1)
            observables.append(SparsePauliOp(pauli_z))

        return observables

    def compute_from_statevector(self, circuit: QuantumCircuit) -> np.ndarray:
        """Extract expectation values from statevector.
        
        Args:
            circuit: QuantumCircuit
        
        Returns:
            Array of shape (3 * n_qubits,) with ⟨X⟩, ⟨Y⟩, ⟨Z⟩ for each qubit
        """
        # Get statevector
        sv = Statevector(circuit)

        # Compute expectation values
        exp_values = []
        for obs in self.observables:
            exp_val = sv.expectation_value(obs)
            exp_values.append(exp_val)

        return np.array(exp_values)

    def compute_via_estimator(
        self,
        circuit: QuantumCircuit,
        precision: float = 0.001,
    ) -> np.ndarray:
        """Extract expectation values via EstimatorV2.
        
        Qiskit 2.x pattern using EstimatorV2 with PUB format.
        
        Args:
            circuit: QuantumCircuit
            precision: Precision for estimator
        
        Returns:
            Array of shape (3 * n_qubits,)
        """
        # Run estimator (Qiskit 2.x V2 pattern)
        job = self.estimator.run([(circuit, self.observables)], precision=precision)
        result = job.result()

        # Extract expectation values
        return result[0].data.evs

    def compute(self, circuit: QuantumCircuit) -> np.ndarray:
        """Compute expectation values (auto-selects method)."""
        if self.use_estimator:
            return self.compute_via_estimator(circuit)
        else:
            return self.compute_from_statevector(circuit)

    @property
    def output_dim(self) -> int:
        """Output dimension (3 * n_qubits)."""
        return 3 * self.n_qubits


class BatchedPauliReadout:
    """Batched Pauli readout for multiple circuits."""

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.reader = PauliReadout(n_qubits, use_estimator=True)
        self.simulator = AerSimulator(method='statevector')
        self.estimator = EstimatorV2(
            backend=self.simulator,
            options={"shots": None}
        )

        # Build observables
        self.observables = self.reader.observables

    def compute_batch(
        self,
        circuits: list[QuantumCircuit],
    ) -> np.ndarray:
        """Compute expectation values for multiple circuits.
        
        Args:
            circuits: List of QuantumCircuit
        
        Returns:
            Array of shape (n_circuits, 3 * n_qubits)
        """
        # Use EstimatorV2 for batched execution
        job = self.estimator.run(
            [(qc, self.observables) for qc in circuits],
            precision=0.001
        )
        result = job.result()

        # Extract all expectation values
        outputs = []
        for i in range(len(circuits)):
            outputs.append(result[i].data.evs)

        return np.array(outputs)
