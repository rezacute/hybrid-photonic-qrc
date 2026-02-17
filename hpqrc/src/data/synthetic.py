"""
Synthetic EV Charging Data Generator

Generates realistic synthetic EV charging demand data with:
- Multiple periodicity (daily, weekly, yearly)
- Autocorrelated noise
- Random spikes
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict
from dataclasses import dataclass


@dataclass
class SyntheticConfig:
    """Configuration for synthetic data generation."""
    n_samples: int = 175200  # 1 year at 15-min resolution
    resolution_minutes: int = 15
    seed: int = 42
    
    # Signal components
    daily_amplitude: float = 1.0
    weekly_amplitude: float = 0.5
    yearly_amplitude: float = 0.3
    
    # Noise
    noise_type: str = "ar1"
    noise_phi: float = 0.95  # AR(1) coefficient
    noise_sigma: float = 0.1
    
    # Spikes
    spike_probability: float = 0.005
    spike_amplitude_range: tuple = (2.0, 5.0)


def generate_synthetic_ev_demand(
    config: Optional[SyntheticConfig] = None,
    **kwargs
) -> pd.DataFrame:
    """Generate synthetic EV charging demand data.
    
    Args:
        config: SyntheticConfig object
        **kwargs: Override config parameters
    
    Returns:
        DataFrame with 'demand' column and optional component columns
    """
    if config is None:
        config = SyntheticConfig(**kwargs)
    
    np.random.seed(config.seed)
    
    # Generate timestamps
    n_samples = config.n_samples
    freq_minutes = config.resolution_minutes
    timestamps = pd.date_range(
        start="2024-01-01",
        periods=n_samples,
        freq=f"{freq_minutes}min"
    )
    
    # Time in hours
    t_hours = np.arange(n_samples) * (freq_minutes / 60)
    
    # Initialize components
    daily = np.zeros(n_samples)
    weekly = np.zeros(n_samples)
    yearly = np.zeros(n_samples)
    noise = np.zeros(n_samples)
    spikes = np.zeros(n_samples)
    
    # Daily pattern (24h period)
    daily = config.daily_amplitude * np.sin(2 * np.pi * t_hours / 24)
    
    # Weekly pattern (168h period)
    weekly = config.weekly_amplitude * np.sin(2 * np.pi * t_hours / 168)
    
    # Yearly pattern (8760h period)
    yearly = config.yearly_amplitude * np.sin(2 * np.pi * t_hours / 8760)
    
    # AR(1) noise
    if config.noise_type == "ar1":
        z = np.random.randn(n_samples) * config.noise_sigma
        for i in range(1, n_samples):
            noise[i] = config.noise_phi * noise[i-1] + np.sqrt(1 - config.noise_phi**2) * z[i]
    else:
        noise = np.random.randn(n_samples) * config.noise_sigma
    
    # Random spikes
    n_spikes = int(config.spike_probability * n_samples)
    spike_indices = np.random.choice(n_samples, n_spikes, replace=False)
    spike_magnitudes = np.random.uniform(
        config.spike_amplitude_range[0],
        config.spike_amplitude_range[1],
        n_spikes
    ) * np.std(daily + weekly + yearly)
    
    for idx, mag in zip(spike_indices, spike_magnitudes):
        spikes[idx] = mag
    
    # Combine components
    demand = daily + weekly + yearly + noise + spikes
    
    # Ensure positive
    demand = np.maximum(demand, 0)
    
    # Add baseline
    baseline = config.daily_amplitude + config.weekly_amplitude + config.yearly_amplitude
    demand = demand + baseline
    
    # Create DataFrame
    df = pd.DataFrame({
        "timestamp": timestamps,
        "demand": demand,
        "daily": daily,
        "weekly": weekly,
        "yearly": yearly,
        "noise": noise,
        "spikes": spikes,
    })
    df.set_index("timestamp", inplace=True)
    
    return df


def generate_train_test_split(
    df: pd.DataFrame,
    test_ratio: float = 0.2,
    gap: int = 0,
) -> tuple:
    """Split data into train and test.
    
    Args:
        df: Input DataFrame
        test_ratio: Fraction for test
        gap: Gap between train and test
    
    Returns:
        (train_df, test_df)
    """
    n = len(df)
    split_idx = int(n * (1 - test_ratio))
    
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx + gap:]
    
    return train_df, test_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="./data/synthetic")
    parser.add_argument("--n-samples", type=int, default=175200)
    parser.add_argument("--resolution", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    config = SyntheticConfig(
        n_samples=args.n_samples,
        resolution_minutes=args.resolution,
        seed=args.seed,
    )
    
    df = generate_synthetic_ev_demand(config)
    
    # Save
    from pathlib import Path
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filename = f"synthetic_ev_{args.resolution}min_{args.n_samples}.csv"
    df.to_csv(output_path / filename)
    
    print(f"Generated {len(df)} samples")
    print(f"Saved to: {output_path / filename}")
