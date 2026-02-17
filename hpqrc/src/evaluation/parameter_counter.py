"""
Parameter Counting Utilities

Count trainable and total parameters for various model types.
"""

import torch
import numpy as np
from typing import Any, Optional


def count_trainable(model: Any) -> int:
    """Count number of trainable parameters.
    
    Works with:
    - PyTorch nn.Module
    - Custom objects with .parameters() method
    - Objects with .readout attribute (RC models)
    
    Args:
        model: Model object
    
    Returns:
        Number of trainable parameters
    """
    # PyTorch models
    if hasattr(model, 'parameters'):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # RC models with readout
    if hasattr(model, 'readout') and hasattr(model.readout, 'n_params'):
        return model.readout.n_params
    
    # Has n_params attribute
    if hasattr(model, 'n_params'):
        return model.n_params
    
    # Has trainable_params property
    if hasattr(model, 'trainable_params'):
        return model.trainable_params
    
    return 0


def count_total(model: Any) -> int:
    """Count total number of parameters (including non-trainable).
    
    Args:
        model: Model object
    
    Returns:
        Total parameter count
    """
    # PyTorch models
    if hasattr(model, 'parameters'):
        return sum(p.numel() for p in model.parameters())
    
    # Has total_params property
    if hasattr(model, 'total_params'):
        return model.total_params
    
    # RC models
    if hasattr(model, 'n_photonic_params') and hasattr(model, 'n_quantum_params'):
        phot = getattr(model, 'n_photonic_params', 0)
        quantum = getattr(model, 'n_quantum_params', 0)
        readout = count_trainable(model)
        return phot + quantum + readout
    
    return 0


def get_param_breakdown(model: Any) -> dict:
    """Get detailed parameter breakdown.
    
    Args:
        model: Model object
    
    Returns:
        Dictionary with parameter counts by component
    """
    breakdown = {
        "trainable": count_trainable(model),
        "total": count_total(model),
    }
    
    # Add component breakdown if available
    if hasattr(model, 'n_photonic_params'):
        breakdown["photonic"] = model.n_photonic_params
    
    if hasattr(model, 'n_quantum_params'):
        breakdown["quantum"] = model.n_quantum_params
    
    if hasattr(model, 'readout'):
        if hasattr(model.readout, 'n_params'):
            breakdown["readout"] = model.readout.n_params
    
    return breakdown


def print_model_summary(model: Any, model_name: str = "Model") -> None:
    """Print model parameter summary.
    
    Args:
        model: Model object
        model_name: Name for display
    """
    breakdown = get_param_breakdown(model)
    
    print(f"\n{'='*50}")
    print(f"{model_name} Summary")
    print(f"{'='*50}")
    
    for key, value in breakdown.items():
        if key not in ["trainable", "total"]:
            print(f"  {key:20s}: {value:>10,}")
    
    print(f"{'-'*50}")
    print(f"  {'Trainable':20s}: {breakdown['trainable']:>10,}")
    print(f"  {'Total':20s}: {breakdown['total']:>10,}")
    print(f"{'='*50}\n")


def compare_param_efficiency(
    models: dict,
    metric_name: str = "accuracy",
    metrics: dict = None,
) -> pd.DataFrame:
    """Compare parameter efficiency across models.
    
    Args:
        models: Dictionary of {name: model}
        metric_name: Name of metric (e.g., "accuracy", "mse")
        metrics: Dictionary of {name: metric_value}
    
    Returns:
        DataFrame with comparison
    """
    import pandas as pd
    
    rows = []
    
    for name, model in models.items():
        params = count_trainable(model)
        metric = metrics.get(name, None)
        
        rows.append({
            "model": name,
            "params": params,
            metric_name: metric,
            "efficiency": metric / params if metric and params > 0 else None,
        })
    
    return pd.DataFrame(rows).sort_values(metric_name, ascending=False)
