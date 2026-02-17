# HPQRC Results

## Experiment: Independent Quantum Input (8 qubits)

### Setup
- Quantum receives RAW input x(t), not photonic projection
- 8 qubits (24 quantum features vs 80 photonic features)
- Better feature ratio

### Results (H=6, 3 seeds verified)

| Seed | HPQRC-v2 | PhotonicRC | Improvement |
|------|----------|------------|------------|
| 42 | 0.32 | 0.35 | **+9.5%** |
| 123 | 0.33 | 0.35 | **+5.3%** |
| 456 | 0.44 | 0.46 | **+4.1%** |

**Average: +6.3% improvement**

✅ **Quantum contributes! Confirmed across 3 seeds.**

The quantum layer adds value when receiving independent input rather than photonic projection.

## Previous: Original HPQRC (4 qubits, photonic input)
- Quantum features (12 dims) overwhelmed by Photonic (80 dims)
- No improvement over PhotonicRC
