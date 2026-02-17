"""
HPQRC Training Script

Hydra-based CLI for training HPQRC and baseline models.
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
from pathlib import Path
import wandb
import warnings

from src.models.hpqrc import HPQRC, PureQRC, PhotonicRC
from src.models.esn import EchoStateNetwork
from src.models.lstm_model import LSTMForecaster
from src.models.transformer_model import TransformerForecaster
from src.data.synthetic import generate_synthetic_ev_demand, SyntheticConfig
from src.data.preprocessing import normalize, create_sequences
from src.training.trainer import Trainer
from src.evaluation.metrics import compute_all_metrics
from src.utils.reproducibility import set_global_seed, capture_environment


def create_model(cfg: DictConfig):
    """Create model from config."""
    model_name = cfg.model.name
    
    if model_name == "hpqrc":
        from src.models.hpqrc import HPQRCConfig
        config = HPQRCConfig(
            in_channels=cfg.model.get("in_channels", 1),
            n_banks=cfg.model.get("n_banks", 5),
            features_per_bank=cfg.model.get("features_per_bank", 16),
            n_qubits=cfg.model.get("n_qubits", 8),
            ridge_alpha=cfg.model.get("ridge_alpha", 1.0),
        )
        return HPQRC(config, device=cfg.hardware.device, seed=cfg.seed)
    
    elif model_name == "pure_qrc":
        return PureQRC(
            n_qubits=cfg.model.get("n_qubits", 8),
            ridge_alpha=cfg.model.get("ridge_alpha", 1.0),
            device=cfg.hardware.device,
            seed=cfg.seed,
        )
    
    elif model_name == "photonic_rc":
        return PhotonicRC(
            in_channels=cfg.model.get("in_channels", 1),
            n_banks=cfg.model.get("n_banks", 5),
            features_per_bank=cfg.model.get("features_per_bank", 16),
            ridge_alpha=cfg.model.get("ridge_alpha", 1.0),
            device=cfg.hardware.device,
            seed=cfg.seed,
        )
    
    elif model_name == "esn":
        return EchoStateNetwork(
            input_dim=cfg.data.input_dim,
            reservoir_size=cfg.model.get("reservoir_size", 256),
            ridge_alpha=cfg.model.get("ridge_alpha", 1.0),
            seed=cfg.seed,
        )
    
    elif model_name == "lstm":
        return LSTMForecaster(
            input_dim=cfg.data.input_dim,
            hidden_size=cfg.model.get("hidden_size", 64),
            num_layers=cfg.model.get("num_layers", 2),
            dropout=cfg.model.get("dropout", 0.1),
        )
    
    elif model_name == "transformer":
        return TransformerForecaster(
            input_dim=cfg.data.input_dim,
            d_model=cfg.model.get("d_model", 64),
            nhead=cfg.model.get("nhead", 4),
            num_layers=cfg.model.get("num_layers", 2),
            dropout=cfg.model.get("dropout", 0.1),
        )
    
    else:
        raise ValueError(f"Unknown model: {model_name}")


def load_data(cfg: DictConfig):
    """Load dataset from config."""
    data_name = cfg.data.name
    
    if data_name == "synthetic":
        synthetic_cfg = SyntheticConfig(
            n_samples=cfg.data.get("n_samples", 175200),
            resolution_minutes=cfg.data.get("resolution_minutes", 15),
            seed=cfg.seed,
        )
        df = generate_synthetic_ev_demand(synthetic_cfg)
        
    else:
        raise ValueError(f"Unknown data: {data_name}")
    
    # Normalize
    df_norm, scaler = normalize(df, method=cfg.data.get("normalize", "standard"))
    
    # Create sequences
    X, y = create_sequences(
        df_norm["demand"].values,
        seq_len=cfg.data.seq_len,
        horizon=cfg.data.horizon,
    )
    
    # Train/test split
    split_idx = int(len(X) * (1 - cfg.data.test_ratio))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return X_train, y_train, X_test, y_test, scaler


@hydra.main(config_path="../configs", config_name="default")
def main(cfg: DictConfig):
    """Main training function."""
    print(OmegaConf.to_yaml(cfg))
    
    # Set seed
    set_global_seed(cfg.seed, deterministic=True)
    
    # Setup wandb
    if cfg.logging.wandb:
        wandb.init(
            project=cfg.logging.project,
            name=f"{cfg.model.name}_{cfg.data.name}_{cfg.seed}",
            config=OmegaConf.to_container(cfg, resolve=True),
        )
    
    # Load data
    print(f"\nLoading data: {cfg.data.name}")
    X_train, y_train, X_test, y_test, scaler = load_data(cfg)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Create model
    print(f"\nCreating model: {cfg.model.name}")
    model = create_model(cfg)
    
    # Train
    print("\nTraining...")
    trainer_config = {
        "epochs": cfg.training.epochs,
        "lr": cfg.training.lr,
        "batch_size": cfg.training.batch_size,
    }
    
    trainer = Trainer(
        model=model,
        config=trainer_config,
        device=cfg.hardware.device,
    )
    
    # Convert to DataLoader format for RC
    from torch.utils.data import TensorDataset, DataLoader
    
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)
    
    if cfg.model.name in ["hpqrc", "pure_qrc", "photonic_rc"]:
        # RC model - fit directly
        model.fit(X_train, y_train.flatten())
        y_pred = model.predict(X_test)
    else:
        # DL model
        train_ds = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size)
        trainer.fit(train_loader)
        y_pred = trainer.predict(DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32))))
    
    # Evaluate
    metrics = compute_all_metrics(y_test.flatten(), y_pred.flatten())
    print(f"\nMetrics: {metrics}")
    
    # Log to wandb
    if cfg.logging.wandb:
        wandb.log(metrics)
        wandb.finish()
    
    # Save results
    output_dir = Path(cfg.logging.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    import json
    results = {
        "model": cfg.model.name,
        "data": cfg.data.name,
        "seed": cfg.seed,
        "metrics": metrics,
    }
    
    with open(output_dir / f"{cfg.model.name}_{cfg.data.name}_{cfg.seed}.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
