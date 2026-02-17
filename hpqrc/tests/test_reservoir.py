"""
Tests for Quantum Reservoir module.
"""

import pytest
import numpy as np
import torch

from src.quantum.reservoir import QuantumReservoir


def test_reservoir_init():
    """Test quantum reservoir initialization."""
    qr = QuantumReservoir(
        n_qubits=6,
        coupling_range=(0.5, 1.5),
        field_range=(0.5, 1.5),
        seed=42,
    )
    
    assert qr.n_qubits == 6
    assert qr.output_dim == 3 * 6  # X, Y, Z for each qubit


def test_reservoir_circuit_qubit_count():
    """Test circuit has correct number of qubits."""
    qr = QuantumReservoir(n_qubits=8, seed=42)
    
    # Build a test circuit
    features = np.random.randn(8)
    circuit = qr.build_circuit(features)
    
    assert circuit.num_qubits == 8


def test_reservoir_observable_count():
    """Test observable count = 3 * N_qubits."""
    qr = QuantumReservoir(n_qubits=6, seed=42)
    
    observables = qr.get_observables()
    
    # Should have 3 observables per qubit (X, Y, Z)
    assert len(observables) == 3 * 6


def test_reservoir_deterministic():
    """Test deterministic behavior with same seed."""
    qr1 = QuantumReservoir(n_qubits=4, seed=42)
    qr2 = QuantumReservoir(n_qubits=4, seed=42)
    
    # Same J and g coupling
    np.testing.assert_array_almost_equal(
        qr1.J.cpu().numpy(),
        qr2.J.cpu().numpy()
    )
    np.testing.assert_array_almost_equal(
        qr1.g.cpu().numpy(),
        qr2.g.cpu().numpy()
    )


def test_reservoir_forward_2d():
    """Test forward pass with 2D input."""
    qr = QuantumReservoir(n_qubits=4, seed=42)
    
    # Input: (batch, n_qubits)
    x = torch.randn(3, 4)
    
    out = qr(x)
    
    # Output: (batch, 3*n_qubits)
    assert out.shape == (3, 12)


def test_reservoir_forward_3d():
    """Test forward pass with 3D input."""
    qr = QuantumReservoir(n_qubits=4, seed=42)
    
    # Input: (batch, seq, n_qubits)
    x = torch.randn(3, 5, 4)
    
    out = qr(x)
    
    # Output: (batch, seq, 3*n_qubits)
    assert out.shape == (3, 5, 12)


def test_reservoir_output_dim():
    """Test output_dim property."""
    qr = QuantumReservoir(n_qubits=8, seed=42)
    
    assert qr.get_output_dim() == 24  # 3 * 8
