"""
Generate Synthetic Dataset

Load synthetic.yaml config, generate dataset, save to data/synthetic/.
"""

import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from src.data.synthetic import generate_synthetic_ev_demand, SyntheticConfig
from src.data.preprocessing import normalize


@hydra.main(config_path="../configs", config_name="default")
def main(cfg: DictConfig):
    """Generate synthetic EV charging dataset."""
    print(OmegaConf.to_yaml(cfg))
    
    # Get synthetic data config
    data_cfg = cfg.data
    
    # Create config
    synthetic_cfg = SyntheticConfig(
        n_samples=data_cfg.get("n_samples", 175200),
        resolution_minutes=data_cfg.get("resolution_minutes", 15),
        seed=cfg.seed,
        daily_amplitude=data_cfg.get("daily_amplitude", 1.0),
        weekly_amplitude=data_cfg.get("weekly_amplitude", 0.5),
        yearly_amplitude=data_cfg.get("yearly_amplitude", 0.3),
        noise_phi=data_cfg.get("noise_phi", 0.95),
        noise_sigma=data_cfg.get("noise_sigma", 0.1),
        spike_probability=data_cfg.get("spike_probability", 0.005),
    )
    
    # Generate
    print("\nGenerating synthetic data...")
    df = generate_synthetic_ev_demand(synthetic_cfg)
    
    # Print summary
    print(f"\nDataset Summary:")
    print(f"  Samples: {len(df)}")
    print(f"  Date range: {df.index[0]} to {df.index[-1]}")
    print(f"  Resolution: {synthetic_cfg.resolution_minutes} minutes")
    print(f"\nDemand statistics:")
    print(f"  Mean: {df['demand'].mean():.3f}")
    print(f"  Std:  {df['demand'].std():.3f}")
    print(f"  Min:  {df['demand'].min():.3f}")
    print(f"  Max:  {df['demand'].max():.3f}")
    
    # Verify planted frequencies
    print(f"\nPlanted components:")
    print(f"  Daily period: 24h (amplitude: {synthetic_cfg.daily_amplitude})")
    print(f"  Weekly period: 168h (amplitude: {synthetic_cfg.weekly_amplitude})")
    print(f"  Yearly period: 8760h (amplitude: {synthetic_cfg.yearly_amplitude})")
    print(f"  AR(1) coefficient: {synthetic_cfg.noise_phi}")
    print(f"  Spike probability: {synthetic_cfg.spike_probability}")
    
    # Save
    output_dir = Path("data/synthetic")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"synthetic_{data_cfg.resolution_minutes}min_{data_cfg.n_samples}.csv"
    filepath = output_dir / filename
    
    df.to_csv(filepath)
    print(f"\nSaved to: {filepath}")
    
    # Also save normalized version
    df_norm, scaler = normalize(df, method="standard")
    filename_norm = filename.replace(".csv", "_norm.csv")
    filepath_norm = output_dir / filename_norm
    df_norm.to_csv(filepath_norm)
    print(f"Saved normalized to: {filepath_norm}")


if __name__ == "__main__":
    main()
