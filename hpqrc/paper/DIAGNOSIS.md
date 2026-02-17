# HPQRC Diagnosis Report

## Results

| Model | H=1 RMSE |
|-------|----------|
| ESN-64 | 0.41 |
| PureQRC-4q | 0.44 |
| Photonic-only | 0.48 |
| HPQRC (current) | 0.47 |

## Diagnosis

### Issue 1: Photonic layer lacks memory
- ESN has explicit feedback (recurrence) → STM ~13
- PDEL is feedforward Conv1D banks → no recurrence
- This is why photonic-only (0.48) is worse than ESN (0.41)

### Issue 2: Feature extraction loses temporal info
- Current: mean pool over time → loses temporal structure
- Better: use last timestep + mean (tested, RMSE 0.45)

### Issue 3: Quantum doesn't help
- QRC-only: 0.44
- HPQRC: 0.47
- Quantum features don't compensate for lack of memory

## Root Cause

**The PDEL architecture is wrong for reservoir computing.**

Real photonic reservoir computers use:
- Single nonlinear oscillator
- Delay line with multiple taps
- Feedback loop

My implementation uses parallel Conv1D banks which are just feature extractors, NOT reservoirs.

## Fix Options

1. **Add feedback to PDEL** - Add recurrence after convolution banks
2. **Use proper delay line** - Implement actual delay loop with feedback
3. **Focus on CNN+LSTM** - Since photonic isn't working, use classical temporal

## Recommendation

The HPQRC concept needs redesign. The photonic delay-loop must have feedback to have memory. Without recurrence, it's just a feature extractor.

Current HPQRC ≈ CNN feature extraction + quantum readout
This is NOT reservoir computing.
