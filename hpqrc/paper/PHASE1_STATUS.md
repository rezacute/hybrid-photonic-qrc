# Phase 1: Baseline Experiments

## Status

### Completed
- [x] 1.1 Data pipeline: ACN-Data (script created)
- [x] 1.2 Synthetic dataset (verified FFT)
- [x] 1.4 ExpandingWindowSplitter (implemented)
- [x] 1.5 Metrics, statistical tests, parameter counter
- [x] 1.6 Preprocessing (scalers, time features)
- [x] 1.7 Memory capacity benchmarks
- [x] 1.9 Pure QRC baseline
- [x] 1.10 ESN baseline

### In Progress
- [ ] 1.11 LSTM baseline
- [ ] 1.12 Transformer baseline
- [ ] 1.13-1.16 Baselines with multiple seeds

### GPU Status ⚠️
- Build completed but GPU not available
- Working on CPU (~2ms for 10-qubit circuits)
- Need CUDA runtime for GPU support

## Running Baselines

```bash
# Generate synthetic data
python scripts/generate_synthetic.py

# Run all baselines
python scripts/run_phase1_baselines.py

# Run specific model
python -c "
from src.models.qrc import PureQRC
model = PureQRC(n_qubits=8)
# ... train and evaluate
"
```

## Expected Results (to beat)

| Model | STM Capacity | RMSE (24h) |
|-------|-------------|------------|
| Pure QRC (8-qubit) | 6-8 | TBD |
| ESN (256 nodes) | 7-10 | TBD |
| LSTM | - | TBD |
| Transformer | - | TBD |

## Next Steps
1. Run full baseline experiments with 10 seeds
2. Compile baseline report
3. Begin HPQRC implementation
