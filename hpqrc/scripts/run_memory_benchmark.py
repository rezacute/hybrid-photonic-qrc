"""
Memory Capacity Benchmark

Run IPC/STM/NMC for all architectures on synthetic data.
"""

import json
from pathlib import Path

import numpy as np

from src.data.preprocessing import normalize
from src.data.synthetic import SyntheticConfig, generate_synthetic_ev_demand
from src.evaluation.memory_capacity import (
    information_processing_capacity,
    short_term_memory_capacity,
)
from src.models.esn import EchoStateNetwork
from src.models.hpqrc import HPQRC, PhotonicRC, PureQRC
from src.utils.reproducibility import set_global_seed


def compute_reservoir_states(model, X: np.ndarray) -> np.ndarray:
    """Extract reservoir states from model."""
    if hasattr(model, 'extract_features'):
        # HPQRC or similar
        return model.extract_features(X)
    else:
        raise NotImplementedError(f"Cannot extract states from {type(model)}")


def run_memory_benchmark(
    model,
    input_sequence: np.ndarray,
    max_delay: int = 200,
) -> dict:
    """Run memory capacity benchmark."""
    # Get reservoir states
    X = input_sequence[:-max_delay].reshape(-1, 1, len(input_sequence[:-max_delay]))
    states = model.extract_features(X)

    # STM capacity
    stm = short_term_memory_capacity(
        states,
        input_sequence[:-max_delay],
        max_delay=max_delay,
    )

    # IPC
    ipc = information_processing_capacity(
        states,
        input_sequence[:-max_delay],
        max_degree=3,
        max_delay=min(50, max_delay),
    )

    return {
        "stm_capacity": stm["total_capacity"],
        "ipc_total": ipc["total_ipc"],
        "ipc_by_degree": ipc["by_degree"],
    }


def main():
    """Main benchmark function."""
    set_global_seed(42, deterministic=True)

    # Generate synthetic data
    print("Generating synthetic data...")
    synthetic_cfg = SyntheticConfig(
        n_samples=5000,
        resolution_minutes=15,
        seed=42,
    )
    df = generate_synthetic_ev_demand(synthetic_cfg)
    df_norm, _ = normalize(df, method="standard")
    input_sequence = df_norm["demand"].values

    # Models to benchmark
    models = {
        "hpqrc": HPQRC(
            config=None,  # Will use defaults
            device="cpu",
            seed=42,
        ),
        "photonic_rc": PhotonicRC(
            in_channels=1,
            n_banks=5,
            features_per_bank=16,
            device="cpu",
            seed=42,
        ),
        "pure_qrc": PureQRC(
            n_qubits=8,
            device="cpu",
            seed=42,
        ),
        "esn": EchoStateNetwork(
            input_dim=1,
            reservoir_size=256,
            seed=42,
        ),
    }

    results = {}

    for name, model in models.items():
        print(f"\nBenchmarking {name}...")

        try:
            result = run_memory_benchmark(model, input_sequence, max_delay=100)
            results[name] = result

            print(f"  STM: {result['stm_capacity']:.3f}")
            print(f"  IPC: {result['ipc_total']:.3f}")

        except Exception as e:
            print(f"  Error: {e}")
            results[name] = {"error": str(e)}

    # Save results
    output_dir = Path("./outputs/memory")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "memory_benchmark.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_file}")

    # Print summary
    print("\n" + "="*50)
    print("MEMORY BENCHMARK SUMMARY")
    print("="*50)

    for name, result in sorted(results.items(), key=lambda x: x[1].get('stm_capacity', 0), reverse=True):
        if "error" not in result:
            print(f"{name:20s} STM: {result['stm_capacity']:.3f}  IPC: {result['ipc_total']:.3f}")


if __name__ == "__main__":
    main()
