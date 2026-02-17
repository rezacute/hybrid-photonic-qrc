# Phase 1: Baseline Experiments

## Results (3 seeds, 3 horizons)

| Model | H=1 RMSE | H=6 RMSE | H=24 RMSE |
|-------|----------|-----------|------------|
| **ESN** (64 nodes) | 0.31 ± 0.06 | 0.35 ± 0.05 | 0.57 ± 0.03 |
| **PureQRC** (4 qubits) | 0.37 ± 0.04 | 0.44 ± 0.04 | 0.75 ± 0.03 |

## Key Findings

1. **ESN outperforms PureQRC** on all horizons
2. **Error increases with horizon** - expected for autoregressive tasks
3. **PureQRC needs more qubits** to match ESN (4 qubits is minimal)
4. **These are the targets HPQRC must beat**

## Analysis

- Horizon 1: ESN is 16% better than PureQRC
- Horizon 6: ESN is 22% better than PureQRC  
- Horizon 24: ESN is 25% better than PureQRC

## Next Steps

- [ ] Run with more seeds (10)
- [ ] Test with larger QRC (8 qubits)
- [ ] Implement HPQRC
- [ ] Compare HPQRC vs baselines

## Files

- `outputs/baselines/baseline_results_20260217_044530.json` - Raw results
