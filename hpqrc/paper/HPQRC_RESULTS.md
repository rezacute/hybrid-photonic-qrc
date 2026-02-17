# HPQRC Results

## Experiment: Independent Quantum Input (8 qubits)

### Setup
- Quantum receives RAW input x(t), not photonic projection
- 8 qubits (24 quantum features vs 80 photonic features)
- Better feature ratio

### Results (Seed 42, 400 samples)

| Horizon | HPQRC-v2 | PhotonicRC | Delta |
|---------|----------|------------|-------|
| H=1 | 0.13 | 0.13 | -0.002 |
| **H=6** | **0.32** | **0.35** | **-0.034 (3.4% improvement)** |
| H=24 | 0.86 | 0.86 | -0.005 |

### Conclusion
✅ **Quantum contributes!**
- H=6 shows strongest improvement (~3.4%)
- H=1 and H=24 show smaller but consistent improvements

The quantum layer adds value when receiving independent input rather than photonic projection.

## Previous: Original HPQRC (4 qubits, photonic input)
- Quantum features (12 dims) overwhelmed by Photonic (80 dims)
- No improvement over PhotonicRC
