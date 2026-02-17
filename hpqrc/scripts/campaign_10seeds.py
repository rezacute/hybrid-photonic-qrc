#!/usr/bin/env python3
"""10-seed campaign for PhotonicRC vs ESN vs LSTM"""

import sys
sys.path.insert(0, '.')

import json
import time
import numpy as np
import pandas as pd
import torch
from src.data.preprocessing import normalize, create_sequences
from src.photonic.delay_reservoir import PhotonicDelayReservoir
from src.models.esn import EchoStateNetwork
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144]
HORIZONS = [1, 6, 24, 168]
SEQ_LEN = 336

def run_experiment(model_name, horizon, seed, seq_len=SEQ_LEN):
    """Run a single experiment"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Generate data
    n_samples = 2000
    t = np.arange(n_samples)
    df = pd.DataFrame({'energy': np.sin(2*np.pi*t/96) + 0.5*np.sin(2*np.pi*t/336) + np.random.randn(n_samples) * 0.3})
    df_norm, _ = normalize(df, method='standard')
    
    X, y = create_sequences(df_norm.values, seq_len=seq_len, horizon=horizon)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train = y[:split, 0, 0] if horizon == 1 else y[:split, horizon-1, 0]
    y_test = y[split:, 0, 0] if horizon == 1 else y[split:, horizon-1, 0]
    
    if model_name == "PhotonicRC":
        pdr = PhotonicDelayReservoir(
            input_dim=1, reservoir_dim=128, delay_taps=[1, 4, 24, 96],
            target_spectral_radius=0.95, leak_rate=0.2, sparsity=0.9, seed=seed
        ).cuda()
        
        X_train_t = torch.tensor(X_train, dtype=torch.float32).cuda()
        X_test_t = torch.tensor(X_test, dtype=torch.float32).cuda()
        
        with torch.no_grad():
            train_f = pdr(X_train_t).cpu().numpy()[:, -1, :]
            test_f = pdr(X_test_t).cpu().numpy()[:, -1, :]
        
        ridge = Ridge(alpha=1.0)
        ridge.fit(train_f, y_train)
        y_pred = ridge.predict(test_f)
        
    elif model_name == "ESN-128":
        esn = EchoStateNetwork(input_dim=1, reservoir_size=128, seed=seed)
        esn.fit(np.array(list(X_train)), y_train)
        y_pred = esn.predict(np.array(list(X_test)))
        
    elif model_name == "LSTM":
        class SimpleLSTM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = torch.nn.LSTM(1, 64, 2, batch_first=True)
                self.fc = torch.nn.Linear(64, 1)
            def forward(self, x):
                _, (h, _) = self.lstm(x)
                return self.fc(h[-1])
        
        model = SimpleLSTM()
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        crit = torch.nn.MSELoss()
        
        X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
        X_test_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
        
        model.train()
        for _ in range(50):
            opt.zero_grad()
            loss = crit(model(X_train_t), y_train_t)
            loss.backward()
            opt.step()
        
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test_t).numpy().flatten()
            
    elif model_name == "Persistence":
        if horizon == 1:
            y_pred = X_test[:, -1, 0]
        else:
            y_pred = np.full_like(y_test, y_train[-1])
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return rmse, y_test, y_pred

# Main loop
results = []
MODELS = ["PhotonicRC", "ESN-128", "LSTM", "Persistence"]

total = len(SEEDS) * len(HORIZONS) * len(MODELS)
done = 0

print(f"Running {total} experiments...")

for seed in SEEDS:
    for horizon in HORIZONS:
        for model_name in MODELS:
            t0 = time.perf_counter()
            try:
                rmse, y_true, y_pred = run_experiment(model_name, horizon, seed)
                elapsed = time.perf_counter() - t0
                
                result = {
                    "model": model_name,
                    "horizon": horizon,
                    "seed": seed,
                    "rmse": float(rmse),
                    "time_s": elapsed,
                }
                results.append(result)
                
                # Save predictions
                np.savez(f"outputs/results/preds/{model_name}_H{horizon}_seed{seed}.npz",
                        y_true=y_true, y_pred=y_pred)
                
                print(f"{model_name} H={horizon} seed={seed}: RMSE={rmse:.4f} ({elapsed:.1f}s)")
                
            except Exception as e:
                print(f"ERROR {model_name} H={horizon} seed={seed}: {e}")
                results.append({
                    "model": model_name,
                    "horizon": horizon,
                    "seed": seed,
                    "rmse": None,
                    "error": str(e)
                })
            
            done += 1
            
            # Save incrementally
            with open("outputs/results/campaign_results.json", "w") as f:
                json.dump(results, f, indent=2)

print(f"\nCampaign complete! {done}/{total} experiments")
