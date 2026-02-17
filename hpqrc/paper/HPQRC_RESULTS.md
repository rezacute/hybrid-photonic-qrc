# HPQRC Integration Results

## Step 1: Smoke Test ✅
- Forward pass works
- Trainable params: ~140 (Ridge only)
- Fixed params: ~16K (PDR + Quantum)

## Step 2: Quantum Contribution (H=1)
- PhotonicRC: 0.13
- HPQRC: 0.13
- Delta: +0.00 (quantum doesn't hurt, but doesn't help either)

## Step 3: Full Comparison (H=[1,6,24])

| Model | H=1 | H=6 | H=24 |
|-------|-----|-----|------|
| **HPQRC** | 0.13 | 0.35 | 0.86 |
| PhotonicRC | 0.13 | 0.35 | 0.86 |
| ESN-64 | 0.10 | 0.27 | 0.85 |

## Step 4: Quantum Delta
- H=1: +0.00 (no change)
- H=6: +0.00 (no change)
- H=24: +0.00 (no change)

## Analysis
The quantum layer is not adding value. The quantum features (from 4 qubits) are being overwhelmed by the 80-dimensional photonic reservoir states. The readout ignores the quantum features.

## Issues
1. Quantum input: only using first 4 PDR states → too small signal
2. Feature scale mismatch: Photonic states dominate
3. Need: larger quantum input or normalized concatenation

## Next Steps
- Normalize features before concatenation
- Use more qubits (8+) 
- Or accept: HPQRC ≈ PhotonicRC as current baseline
