"""
Phase 1: Baseline Experiments Runner

Runs all baseline models and collects metrics.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings

from src.data.synthetic import generate_synthetic_ev_demand, SyntheticConfig
from src.data.cv_splitter import ExpandingWindowSplitter
from src.data.preprocessing import normalize, create_sequences
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.memory_capacity import short_term_memory_capacity
from src.models.qrc import PureQRC
from src.models.esn import EchoStateNetwork
from src.models.lstm_model import LSTMForecaster
from src.models.transformer_model import TransformerForecaster
from src.utils.reproducibility import set_global_seed


def run_baseline(
    name: str,
    model_class,
    model_kwargs: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int = 42,
) -> dict:
    """Run a single baseline model."""
    set_global_seed(seed)
    
    try:
        # Create model
        model = model_class(**model_kwargs)
        
        # Fit
        if hasattr(model, 'fit'):
            if name in ['LSTM', 'Transformer']:
                # DL models need different fitting
                from torch.utils.data import TensorDataset, DataLoader
                import torch
                
                train_ds = TensorDataset(
                    torch.tensor(X_train, dtype=torch.float32),
                    torch.tensor(y_train, dtype=torch.float32)
                )
                train_loader = DataLoader(train_ds, batch_size=64)
                
                # Simple training loop
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                criterion = torch.nn.MSELoss()
                
                for epoch in range(20):
                    model.train()
                    for X_batch, y_batch in train_loader:
                        optimizer.zero_grad()
                        pred = model(X_batch)
                        loss = criterion(pred, y_batch.squeeze())
                        loss.backward()
                        optimizer.step()
                
                model.eval()
                with torch.no_grad():
                    y_pred = model(torch.tensor(X_test, dtype=torch.float32)).numpy().squeeze()
            else:
                # RC models
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
        else:
            raise ValueError(f"Model {name} has no fit method")
        
        # Metrics
        metrics = compute_all_metrics(y_test, y_pred)
        
        return {
            "name": name,
            "success": True,
            "metrics": metrics,
            "n_params": model.n_params if hasattr(model, 'n_params') else 0,
        }
        
    except Exception as e:
        return {
            "name": name,
            "success": False,
            "error": str(e),
        }


def run_memory_benchmark(
    name: str,
    model,
    input_sequence: np.ndarray,
    max_delay: int = 50,
) -> dict:
    """Run memory capacity benchmark."""
    try:
        # Get reservoir states
        X = input_sequence[:-max_delay].reshape(-1, 1, len(input_sequence[:-max_delay]))
        states = model.extract_features(X)
        
        # STM capacity
        stm = short_term_memory_capacity(
            states,
            input_sequence[:-max_delay],
            max_delay=max_delay,
        )
        
        return {
            "name": name,
            "stm_capacity": stm["total_capacity"],
        }
    except Exception as e:
        return {"name": name, "error": str(e)}


def main():
    """Run all Phase 1 baselines."""
    print("=" * 60)
    print("PHASE 1: BASELINE EXPERIMENTS")
    print("=" * 60)
    
    # Config
    n_seeds = 3
    horizons = [1, 6, 24]  # forecast horizons
    results = []
    
    # Generate synthetic data
    print("\n[1/4] Generating synthetic data...")
    config = SyntheticConfig(n_samples=50000, seed=42)
    df = generate_synthetic_ev_demand(config)
    df_norm, _ = normalize(df, method="standard")
    
    # Prepare data for different horizons
    for horizon in horizons:
        print(f"\n  Horizon: {horizon} steps")
        
        X, y = create_sequences(df_norm["demand"].values, seq_len=96, horizon=horizon)
        
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Baselines
        baselines = [
            ("PureQRC", PureQRC, {"n_qubits": 8, "device": "cpu", "seed": 42}),
            ("ESN", EchoStateNetwork, {"input_dim": 1, "reservoir_size": 256, "seed": 42}),
            # ("LSTM", LSTMForecaster, {"input_dim": 1, "hidden_size": 64}),
            # ("Transformer", TransformerForecaster, {"input_dim": 1}),
        ]
        
        for name, model_class, kwargs in baselines:
            print(f"    Running {name}...")
            
            for seed in range(n_seeds):
                result = run_baseline(name, model_class, kwargs, X_train, y_train, X_test, y_test, seed)
                result["horizon"] = horizon
                result["seed"] = seed
                results.append(result)
                
                if result["success"]:
                    print(f"      {name} seed={seed}: RMSE={result['metrics']['rmse']:.4f}")
    
    # Memory benchmarks
    print("\n[2/4] Memory capacity benchmarks...")
    input_seq = df_norm["demand"].values[:2000]
    
    for name, model_class, kwargs in [("PureQRC", PureQRC, {"n_qubits": 8}), ("ESN", EchoStateNetwork, {"input_dim": 1})]:
        try:
            model = model_class(**kwargs)
            mem_result = run_memory_benchmark(name, model, input_seq)
            print(f"    {name} STM: {mem_result.get('stm_capacity', 'N/A'):.2f}")
            results.append(mem_result)
        except Exception as e:
            print(f"    {name} failed: {e}")
    
    # Save results
    print("\n[3/4] Saving results...")
    output_dir = Path("./outputs/baselines")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"phase1_baselines_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"  Saved to {output_file}")
    
    # Summary
    print("\n[4/4] Summary:")
    successful = [r for r in results if r.get("success")]
    print(f"  Successful: {len(successful)}/{len(results)}")
    
    if successful:
        df_results = pd.DataFrame(successful)
        if "metrics" in df_results.columns:
            print("\n  By Model:")
            for name in df_results["name"].unique():
                name_df = df_results[df_results["name"] == name]
                rmse_mean = np.mean([m["rmse"] for m in name_df["metrics"]])
                print(f"    {name}: RMSE = {rmse_mean:.4f}")
    
    print("\n" + "=" * 60)
    print("PHASE 1 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
