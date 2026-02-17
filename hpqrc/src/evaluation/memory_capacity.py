"""
Memory Capacity Analysis for Reservoir Computing

Measures:
- Short-term memory (STM): How well past inputs are recalled
- Information processing capacity (IPC): Generalization to nonlinear tasks
"""

import numpy as np
from typing import Dict, List, Tuple
from sklearn.linear_model import Ridge


def short_term_memory_capacity(
    reservoir_states: np.ndarray,
    input_sequence: np.ndarray,
    max_delay: int = 200,
    ridge_alpha: float = 1.0,
) -> Dict:
    """Compute short-term memory capacity.
    
    Measures how well the reservoir can recall past inputs.
    MC = Σ_k corr²(ŷ_k(t), u(t-k)) for k = 1 to max_delay
    
    Args:
        reservoir_states: Reservoir states of shape (n_samples, state_dim)
        input_sequence: Input signal of shape (n_samples,)
        max_delay: Maximum delay to test
        ridge_alpha: Ridge regularization
    
    Returns:
        Dictionary with total_capacity, per_delay correlations, max_tested_delay
    """
    n_samples = len(reservoir_states)
    max_delay = min(max_delay, n_samples - 1)
    
    # 70/30 split
    split_idx = int(n_samples * 0.7)
    
    train_states = reservoir_states[:split_idx]
    test_states = reservoir_states[split_idx:]
    train_input = input_sequence[:split_idx]
    test_input = input_sequence[split_idx:]
    
    per_delay_corr = {}
    total_capacity = 0.0
    
    for delay in range(1, max_delay + 1):
        # Target: input at time t-delay
        train_target = train_input[delay:]
        test_target = test_input[delay:]
        
        # Input: states from delay onwards
        train_X = train_states[:-delay] if delay > 0 else train_states
        test_X = test_states[:-delay] if delay > 0 else test_states
        
        if len(train_X) < 10 or len(test_X) < 10:
            break
        
        # Fit readout
        ridge = Ridge(alpha=ridge_alpha)
        ridge.fit(train_X, train_target)
        
        # Predict
        pred = ridge.predict(test_X)
        
        # Correlation
        corr = np.corrcoef(pred, test_target)[0, 1]
        
        if np.isnan(corr):
            corr = 0.0
        
        corr_sq = corr ** 2
        per_delay_corr[delay] = float(corr_sq)
        total_capacity += corr_sq
        
        # Early stop if correlation drops below threshold
        if corr_sq < 0.01:
            break
    
    return {
        "total_capacity": float(total_capacity),
        "per_delay": per_delay_corr,
        "max_tested_delay": len(per_delay_corr),
    }


def information_processing_capacity(
    reservoir_states: np.ndarray,
    input_sequence: np.ndarray,
    max_degree: int = 3,
    max_delay: int = 100,
    ridge_alpha: float = 1.0,
) -> Dict:
    """Compute information processing capacity (IPC).
    
    Extends STM with Legendre polynomial targets.
    Measures reservoir's ability to compute nonlinear functions of past inputs.
    
    Args:
        reservoir_states: Reservoir states
        input_sequence: Input signal (should be in [-1, 1])
        max_degree: Maximum polynomial degree
        max_delay: Maximum delay
        ridge_alpha: Ridge regularization
    
    Returns:
        Dictionary with total_ipc, by_degree
    """
    n_samples = len(reservoir_states)
    max_delay = min(max_delay, n_samples - 1)
    
    # Normalize input to [-1, 1]
    input_norm = 2 * (input_sequence - input_sequence.min()) / (input_sequence.max() - input_sequence.min() + 1e-8) - 1
    
    # 70/30 split
    split_idx = int(n_samples * 0.7)
    
    train_states = reservoir_states[:split_idx]
    test_states = reservoir_states[split_idx:]
    train_input = input_norm[:split_idx]
    test_input = input_norm[split_idx:]
    
    by_degree = {}
    total_ipc = 0.0
    
    for degree in range(1, max_degree + 1):
        # Compute Legendre polynomial targets
        train_targets = np.zeros((len(train_input) - max_delay, max_delay))
        test_targets = np.zeros((len(test_input) - max_delay, max_delay))
        
        for d in range(max_delay):
            train_targets[:, d] = np.roll(train_input, d)[max_delay:]
            test_targets[:, d] = np.roll(test_input, d)[max_delay:]
        
        # Compute P_legendre of degree
        train_legendre = legendre_polynomial(train_input[max_delay:], degree)
        test_legendre = legendre_polynomial(test_input[max_delay:], degree)
        
        # Fit readout
        train_X = train_states[max_delay:]
        test_X = test_states[max_delay:]
        
        ridge = Ridge(alpha=ridge_alpha)
        ridge.fit(train_X, train_legendre)
        
        pred = ridge.predict(test_X)
        
        # Correlation
        corr = np.corrcoef(pred, test_legendre)[0, 1]
        if np.isnan(corr):
            corr = 0.0
        
        ipc_degree = corr ** 2
        by_degree[degree] = float(ipc_degree)
        total_ipc += ipc_degree
    
    return {
        "total_ipc": float(total_ipc),
        "by_degree": by_degree,
    }


def legendre_polynomial(x: np.ndarray, degree: int) -> np.ndarray:
    """Compute Legendre polynomial of given degree.
    
    Args:
        x: Input values (should be in [-1, 1])
        degree: Polynomial degree
    
    Returns:
        P_degree(x)
    """
    if degree == 0:
        return np.ones_like(x)
    elif degree == 1:
        return x.copy()
    elif degree == 2:
        return 0.5 * (3 * x ** 2 - 1)
    elif degree == 3:
        return 0.5 * (5 * x ** 3 - 3 * x)
    else:
        # General recurrence
        P0 = np.ones_like(x)
        P1 = x        for n.copy()
        
 in range(2, degree + 1):
            Pn = ((2 * n - 1) * x * P1 - (n - 1) * P0) / n
            P0, P1 = P1, Pn
        
        return P1


def compute_memory曲线(reservoir_states: np.ndarray, input_sequence: np.ndarray) -> Dict:
    """Compute full memory curve.
    
    Args:
        reservoir_states: States from reservoir
        input_sequence: Input signal
    
    Returns:
        Memory curve data
    """
    max_delay = min(200, len(reservoir_states) // 2)
    
    result = short_term_memory_capacity(
        reservoir_states,
        input_sequence,
        max_delay=max_delay,
    )
    
    return result
