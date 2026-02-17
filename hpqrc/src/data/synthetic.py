"""
Synthetic EV Charging Data Generator

Generates synthetic EV charging demand data with realistic patterns:
- Daily patterns (24h periodicity)
- Weekly patterns (168h periodicity)
- Yearly patterns (8760h periodicity)
- AR(1) noise
- Random spikes
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List
from pathlib import Path


def generate_synthetic_ev_demand(
    n_samples: int = 175200,
    resolution_minutes: int = 15,
    seed: int = 42,
    frequencies: Optional[List[Dict]] = None,
    noise_type: str = "ar1",
    noise_phi: float = 0.8,
    noise_sigma: float = 0.1,
    spike_probability: float = 0.01,
    spike_amplitude: float = 2.0,
) -> pd.DataFrame:
    """Generate synthetic EV charging demand data.
    
    Args:
        n_samples: Number of time steps to generate
        resolution_minutes: Time resolution in minutes
        seed: Random seed for reproducibility
        frequencies: List of frequency components to add
        noise_type: Type of noise ("ar1", "white")
        noise_phi: AR(1) autocorrelation coefficient
        noise_sigma: Noise standard deviation
        spike_probability: Probability of a spike
        spike_amplitude: Spike amplitude
    
    Returns:
        DataFrame with timestamp and energy columns
    """
    np.random.seed(seed)
    
    # Default frequencies
    if frequencies is None:
        frequencies = [
            {"period": 24, "amplitude": 1.0, "phase": 0.0},      # Daily
            {"period": 168, "amplitude": 0.5, "phase": 0.0},   # Weekly
            {"period": 8760, "amplitude": 0.3, "phase": 0.0},  # Yearly
        ]
    
    # Generate timestamps
    freq_hours = resolution_minutes / 60
    n_hours = n_samples * freq_hours
    timestamps = pd.date_range(
        start="2024-01-01",
        periods=n_samples,
        freq=f"{resolution_minutes}min"
    )
    
    # Generate base signal with multiple frequencies
    t = np.arange(n_samples) * freq_hours  # Time in hours
    
    signal = np.zeros(n_samples)
    for freq in frequencies:
        period = freq["period"]
        amplitude = freq["amplitude"]
        phase = freq.get("phase", 0.0)
        signal += amplitude * np.sin(2 * np.pi * t / period + phase)
    
    # Add mean
    signal += signal.mean()
    
    # Add noise
    if noise_type == "ar1":
        noise = np.zeros(n_samples)
        z = np.random.randn(n_samples) * noise_sigma
        for i in range(1, n_samples):
            noise[i] = noise_phi * noise[i-1] + np.sqrt(1 - noise_phi**2) * z[i]
    else:
        noise = np.random.randn(n_samples) * noise_sigma
    
    signal += noise
    
    # Add random spikes
    if spike_probability > 0:
        spike_mask = np.random.rand(n_samples) < spike_probability
        n_spikes = np.sum(spike_mask)
        if n_spikes > 0:
            spike_values = np.random.randn(n_spikes) * spike_amplitude * signal.std()
            signal[spike_mask] += spike_values
    
    # Ensure non-negative
    signal = np.maximum(signal, 0)
    
    # Create DataFrame
    df = pd.DataFrame({
        "timestamp": timestamps,
        "energy": signal,
    })
    df.set_index("timestamp", inplace=True)
    
    return df


def save_synthetic_data(
    output_dir: str,
    n_samples: int = 175200,
    resolution_minutes: int = 15,
    **kwargs
) -> Path:
    """Generate and save synthetic data to CSV.
    
    Args:
        output_dir: Directory to save the data
        n_samples: Number of samples to generate
        resolution_minutes: Data resolution
        **kwargs: Additional arguments for generate_synthetic_ev_demand
    
    Returns:
        Path to saved file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    df = generate_synthetic_ev_demand(
        n_samples=n_samples,
        resolution_minutes=resolution_minutes,
        **kwargs
    )
    
    filename = f"synthetic_ev_{resolution_minutes}min_{n_samples}.csv"
    filepath = output_path / filename
    df.to_csv(filepath)
    
    print(f"Generated {len(df)} samples")
    print(f"Saved to: {filepath}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"Energy range: {df['energy'].min():.2f} to {df['energy'].max():.2f}")
    
    return filepath


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="./data/synthetic", help="Output directory")
    parser.add_argument("--n-samples", type=int, default=175200, help="Number of samples")
    parser.add_argument("--resolution", type=int, default=15, help="Resolution in minutes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    save_synthetic_data(
        output_dir=args.output,
        n_samples=args.n_samples,
        resolution_minutes=args.resolution,
        seed=args.seed,
    )
