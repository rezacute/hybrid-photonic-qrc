"""
Experiment Runner

Orchestrates multi-seed, multi-fold, multi-horizon experiments.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
import warnings

# Try to import wandb
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    warnings.warn("wandb not installed, logging disabled")


@dataclass
class ExperimentConfig:
    """Configuration for experiment."""
    # Model
    model_name: str
    model_config: Dict
    
    # Data
    data_name: str
    data_config: Dict
    
    # Experiment
    n_seeds: int = 3
    n_folds: int = 5
    horizons: List[int] = None
    
    # Training
    epochs: int = 100
    batch_size: int = 64
    lr: float = 0.001
    
    # Output
    output_dir: str = "./outputs/results"
    use_wandb: bool = True
    
    def __post_init__(self):
        if self.horizons is None:
            self.horizons = [1, 6, 24]


class ResultsCollection:
    """Container for experiment results."""
    
    def __init__(self):
        self.results = []
    
    def add(self, result: Dict):
        self.results.append(result)
    
    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.results)
    
    def to_dict(self) -> Dict:
        return {"results": self.results}
    
    def save(self, path: Path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def summary(self) -> Dict:
        df = self.to_dataframe()
        
        # Group by model
        summary = {}
        for model in df['model'].unique():
            model_df = df[df['model'] == model]
            
            summary[model] = {
                "n_experiments": len(model_df),
                "mean_metric": float(model_df['metric'].mean()),
                "std_metric": float(model_df['metric'].std()),
                "min_metric": float(model_df['metric'].min()),
                "max_metric": float(model_df['metric'].max()),
            }
        
        return summary


class ExperimentRunner:
    """Orchestrates experiments."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results = ResultsCollection()
        
        # Setup output
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup wandb
        if config.use_wandb and HAS_WANDB:
            wandb.init(
                project="hpqrc-experiments",
                name=f"{config.model_name}_{config.data_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                config=asdict(config),
            )
    
    def run(
        self,
        model_factory,
        data_loader,
    ) -> ResultsCollection:
        """Run full experiment suite.
        
        Args:
            model_factory: Function that creates model instances
            data_loader: Function that loads data
        
        Returns:
            ResultsCollection
        """
        seeds = list(range(self.config.n_seeds))
        
        for seed in seeds:
            print(f"\n{'='*50}")
            print(f"SEED {seed}")
            print(f"{'='*50}")
            
            # Set seed
            np.random.seed(seed)
            
            # Run for each horizon
            for horizon in self.config.horizons:
                print(f"\n--- Horizon: {horizon} ---")
                
                # Load data
                train_data, test_data = data_loader(
                    data_name=self.config.data_name,
                    horizon=horizon,
                    seed=seed,
                )
                
                # Create model
                model = model_factory(
                    model_name=self.config.model_name,
                    config=self.config.model_config,
                    seed=seed,
                )
                
                # Train
                from .trainer import Trainer
                trainer = Trainer(
                    model=model,
                    config={
                        "epochs": self.config.epochs,
                        "lr": self.config.lr,
                        "batch_size": self.config.batch_size,
                    },
                )
                
                # Simple train/val split
                from torch.utils.data import TensorDataset, DataLoader as TorchLoader
                
                X_train = torch.tensor(train_data['X'], dtype=torch.float32)
                y_train = torch.tensor(train_data['y'], dtype=torch.float32)
                
                train_ds = TensorDataset(X_train, y_train)
                train_loader = TorchLoader(train_ds, batch_size=self.config.batch_size)
                
                trainer.fit(train_loader)
                
                # Evaluate
                X_test = torch.tensor(test_data['X'], dtype=torch.float32)
                y_test = torch.tensor(test_data['y'], dtype=torch.float32)
                
                test_ds = TensorDataset(X_test, y_test)
                test_loader = TorchLoader(test_ds, batch_size=self.config.batch_size)
                
                predictions = trainer.predict(test_loader)
                
                # Compute metrics
                from ..evaluation.metrics import compute_all_metrics
                metrics = compute_all_metrics(y_test.numpy(), predictions)
                
                # Store result
                result = {
                    "model": self.config.model_name,
                    "data": self.config.data_name,
                    "seed": seed,
                    "horizon": horizon,
                    "metric": metrics.get('rmse', 0),
                    "mae": metrics.get('mae', 0),
                    "mape": metrics.get('mape', 0),
                    "r2": metrics.get('r2', 0),
                    "timestamp": datetime.now().isoformat(),
                }
                
                self.results.add(result)
                
                # Log to wandb
                if self.config.use_wandb and HAS_WANDB:
                    wandb.log(result)
        
        return self.results
    
    def save_results(self, name: Optional[str] = None):
        """Save results to file."""
        if name is None:
            name = f"{self.config.model_name}_{self.config.data_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        path = self.output_dir / f"{name}.json"
        self.results.save(path)
        print(f"Results saved to: {path}")
        
        return path
    
    def get_summary(self) -> Dict:
        """Get results summary."""
        return self.results.summary()


def run_standard_benchmark(
    models: Dict[str, Any],
    data: Dict[str, Any],
    output_dir: str = "./outputs/results",
) -> ResultsCollection:
    """Run standard benchmark comparison.
    
    Args:
        models: Dictionary of {name: model_class}
        data: Dictionary with 'train' and 'test' keys
        output_dir: Output directory
    
    Returns:
        ResultsCollection
    """
    results = ResultsCollection()
    
    for model_name, model_class in models.items():
        print(f"\n{'='*50}")
        print(f"Model: {model_name}")
        print(f"{'='*50}")
        
        # Train
        model = model_class()
        
        # Simple training (placeholder)
        # In practice, use Trainer class
        
        # Evaluate
        # ...
        
        results.add({
            "model": model_name,
            "metric": 0.0,
        })
    
    return results
