# Phase 1 & 2 Progress

## Phase 1: Baseline Experiments

### Results (3+ seeds)

| Model | H=1 RMSE | H=6 RMSE | H=24 RMSE |
|-------|----------|-----------|------------|
| **ESN-64** | 0.30 ± 0.05 | 0.32 ± 0.05 | 0.54 ± 0.04 |
| **ESN-256** | TBD | TBD | TBD |
| **PureQRC-4q** | 0.36 ± 0.05 | 0.42 ± 0.05 | 0.73 ± 0.04 |
| **PureQRC-8q** | TBD | TBD | TBD |

### STM Capacity (Memory)
| Model | STM Capacity |
|-------|-------------|
| **ESN-64** | 12.90 |
| **ESN-256** | 13.62 |
| **QRC-4q** | ~0 (no feedback) |
| **QRC-8q** | ~0 (no feedback) |

### Key Findings
1. ESN significantly outperforms PureQRC due to feedback (recurrence)
2. QRC without feedback has no short-term memory
3. HPQRC combines photonic delay (recurrence) with quantum reservoir

## Phase 2: HPQRC Implementation

### Task 2.1: Photonic Delay-Loop Emulator ✅
- Implemented: `src/photonic/delay_loop.py`
- Multi-bank PDEL with different kernel sizes
- Causal padding to prevent future leakage

### Task 2.2: HPQRC Model ✅
- Implemented: `src/models/hpqrc.py`
- Combines photonic + quantum + readout

### Next Steps
- [ ] Test HPQRC on synthetic data
- [ ] Compare HPQRC vs baselines
- [ ] Run LSTM baseline (overnight)
