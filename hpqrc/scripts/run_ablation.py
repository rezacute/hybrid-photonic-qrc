"""
Run Ablation Experiments

Execute specific ablation experiments from config.
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
from pathlib import Path
import json

from src.models.hpqrc import HPQRC, HPQRCConfig
from src.data.synthetic import generate_synthetic_ev_demand, SyntheticConfig
from src.data.preprocessing import normalize, create_sequences
from src.training.trainer import Trainer
from src.evaluation.metrics import compute_all_metrics
from src.utils.reproducibility import set_global_seed


def run_ablation(cfg: DictConfig) -> dict:
    """Run a single ablation experiment."""
    set_global_seed(cfg.seed, deterministic=True)
    
    # Load data
    synthetic_cfg = SyntheticConfig(
        n_samples=cfg.data.n_samples,
        resolution_minutes=cfg.data.resolution_minutes,
        seed=cfg.seed,
    )
    df = generate_synthetic_ev_demand(synthetic_cfg)
    
    df_norm, _ = normalize(df, method="standard")
    X, y = create_sequences(df_norm["demand"].values, cfg.data.seq_len, cfg.data.horizon)
    
    split_idx = int(len(X) * (1 - cfg.data.test_ratio))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Create model variant
    model_cfg = HPQRCConfig(
        in_channels=1,
        n_banks=cfg.model.n_banks,
        features_per_bank=cfg.model.features_per_bank,
        n_qubits=cfg.model.n_qubits,
        ridge_alpha=cfg.model.ridge_alpha,
        feature_mode=cfg.model.feature_mode,
        frozen_kernels=cfg.model.frozen_kernels,
    )
    
    model = HPQRC(model_cfg, device=cfg.hardware.device, seed=cfg.seed)
    
    # Train
    model.fit(X_train, y_train.flatten())
    
    # Evaluate
    y_pred = model.predict(X_test)
    metrics = compute_all_metrics(y_test.flatten(), y_pred.flatten())
    
    return {
        "variant": cfg.ablation.variant_name,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "metrics": metrics,
        "params": {
            "trainable": model.trainable_params,
            "total": model.total_params,
        }
    }


@hydra.main(config_path="../configs", config_name="default")
def main(cfg: DictConfig):
    """Run ablation experiments."""
    print(OmegaConf.to_yaml(cfg))
    
    # Get ablation experiment config
    ablation_cfg = cfg.ablation
    
    results = []
    
    for variant_name in ablation_cfg.variants:
        print(f"\n{'='*50}")
        print(f"Running: {variant_name}")
        print(f"{'='*50}")
        
        # Update config for this variant
        variant_cfg = cfg.copy()
        variant_cfg.ablation.variant_name = variant_name
        
        # Apply variant-specific overrides
        if variant_name == "no_photonic":
            variant_cfg.model.feature_mode = "qrc-only"
        elif variant_name == "no_quantum":
            variant_cfg.model.feature_mode = "phot-only"
        elif variant_name == "frozen_photonic":
            variant_cfg.model.frozen_kernels = True
        elif variant_name == "more_banks":
            variant_cfg.model.n_banks = 8
        elif variant_name == "more_qubits":
            variant_cfg.model.n_qubits = 12
        
        result = run_ablation(variant_cfg)
        results.append(result)
        
        print(f"RMSE: {result['metrics']['rmse']:.4f}")
    
    # Save results
    output_dir = Path(ablation_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"ablation_{ablation_cfg.experiment_name}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    # Print summary
    print("\n" + "="*50)
    print("ABLATION SUMMARY")
    print("="*50)
    
    for result in sorted(results, key=lambda x: x['metrics']['rmse']):
        print(f"{result['variant']:20s} RMSE: {result['metrics']['rmse']:.4f}")


if __name__ == "__main__":
    main()
