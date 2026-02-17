# HPQRC: Hybrid Photonic-Quantum Reservoir Computing

**Short-term memory enhancement for Quantum Reservoir Computing using Photonic Delay-Loop Emulation**

## Overview

HPQRC solves the short-memory problem of Quantum Reservoir Computing (QRC) by adding a Photonic Delay-Loop Emulation Layer (PDEL) before the quantum reservoir. Pure QRC can only "remember" ~5 time steps due to decoherence; HPQRC extends this to 100+ steps using parallel 1D convolutions that emulate photonic delay loops at multiple temporal scales.

## Architecture

```
Input: EV Charging Demand Time Series
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 1: Photonic Delay-Loop Emulation Layer (PDEL)        │
│  K parallel Conv1D banks with kernel sizes                  │
│  τ = {4, 24, 96, 168, 672} (sub-hourly to weekly)        │
│  Output: h_phot(t) ∈ R^{K × D_phot}                       │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 2: Quantum Reservoir                                 │
│  N-qubit Ising Hamiltonian with ZZ coupling                │
│  + transverse field. Fixed (untrained) parameters.        │
│  Angle encoding of photonic features.                      │
│  Readout via Pauli expectation values ⟨X⟩,⟨Y⟩,⟨Z⟩        │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 3: Linear Readout                                   │
│  Ridge regression on concatenated [h_phot; r_qrc] features. │
│  Only trained layer.                                       │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
Output: EV Charging Demand Forecast (Δ ∈ {1h, 6h, 24h, 72h, 168h})
```

## Quick Start

```bash
# Install
pip install -e .

# Run default experiment
python scripts/train.py model=hpqrc data=acn_caltech

# Run ablation
python scripts/run_ablation.py experiment=ablation_memory
```

## Project Structure

```
hpqrc/
├── configs/          # Hydra YAML configs
├── src/              # Source modules
│   ├── photonic/     # Photonic delay loop emulation
│   ├── quantum/      # Quantum reservoir
│   ├── readout/      # Linear readout
│   ├── models/       # Model implementations
│   ├── data/         # Data loading
│   ├── evaluation/   # Metrics & benchmarks
│   ├── training/    # Training loops
│   └── utils/       # Utilities
├── scripts/          # Executable scripts
├── tests/            # Test suite
├── notebooks/        # Jupyter notebooks
├── data/             # Data directory
├── outputs/          # Model/results output
└── paper/           # LaTeX paper
```

## Datasets

- **ACN-Caltech**: 15min resolution, 672 lookback
- **Boulder**: 1h resolution, 168 lookback
- **Pecan Street**: 15min resolution, 672 lookback
- **UK Power**: 30min resolution, 336 lookback
- **Synthetic**: 15min resolution, 175,200 samples

## Models

- **HPQRC**: Full photonic + quantum + readout
- **QRC Baseline**: Pure quantum reservoir (no photonic)
- **Photonic RC**: Photonic delay loops + readout (no quantum)
- **ESN**: Echo State Network
- **LSTM**: 2-layer LSTM
- **Transformer**: 2-layer Transformer
- **SARIMA**: Statistical baseline

## Citation

```bibtex
@article{hpqrc2026,
  title={Hybrid Photonic-Quantum Reservoir Computing for EV Demand Forecasting},
  author={},
  year={2026}
}
```

## License

MIT
