# GPU Setup for HPQRC

## Current Status

### Hardware ✅
- 4x NVIDIA Tesla T4 (15GB each)
- CUDA 12.8 installed at `/usr/local/cuda-12.8`
- Compute capability: 7.5

### Software
- qiskit-aer: 0.17.2 (CPU-only)
- qiskit: 2.1.2 ✅
- GPU simulation: ❌ NOT AVAILABLE

## Why No GPU?

qiskit-aer from PyPI (`pip install qiskit-aer`) does NOT include GPU support.
The `qiskit-aer-gpu` package is stuck at version 0.15.1 and is **incompatible with Qiskit 2.x**.

## Solution: Build from Source

To enable GPU support, build qiskit-aer from source with CUDA:

```bash
# Install system dependencies (requires root)
sudo apt-get install libopenblas-dev liblapack-dev

# Install build tools
pip install scikit-build ninja "conan<2"

# Clone and build
git clone --depth 1 --branch 0.17.2 https://github.com/Qiskit/qiskit-aer.git
cd qiskit-aer

CMAKE_ARGS="-DCUDA=True -DCUDA_ARCH=7.5 -DOPENBLAS_ROOT=/path/to/openblas" \
pip install --break-system-packages . --no-build-isolation

# Verify GPU support
python3 -c "from qiskit_aer import AerSimulator; s=AerSimulator(); print('Devices:', s.available_devices())"
```

Expected output after build:
```
Devices: ('CPU', 'GPU')
```

## Workaround: Use CPU

The current CPU simulation is fast enough for development:
- 10-qubit circuit: ~1.5ms
- 8-qubit circuit: ~0.5ms

For production/large-scale experiments, either:
1. Build qiskit-aer with GPU support (above)
2. Use a cloud GPU instance with pre-built qiskit-aer-gpu
3. Use IBM Quantum cloud backend

## Notes

- Building takes ~30-60 minutes
- Requires ~10GB disk space
- May need to adjust CUDA_ARCH for your GPU
