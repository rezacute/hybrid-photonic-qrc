"""
Reproducibility and Environment Utilities

Set seeds, capture environment, config hashing.
"""

import hashlib
import json
import os
import platform
import random
from pathlib import Path
from typing import Any

import numpy as np


def set_global_seed(
    seed: int,
    deterministic: bool = True,
) -> None:
    """Set global random seeds for reproducibility.
    
    Args:
        seed: Random seed
        deterministic: If True, use deterministic algorithms where possible
    """
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    # TensorFlow
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass

    # Environment variables
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Deterministic algorithms
    if deterministic:
        try:
            import torch
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except ImportError:
            pass

        # TensorFlow
        try:
            import tensorflow as tf
            tf.config.experimental.enable_op_determinism()
        except ImportError:
            pass


def capture_environment() -> dict[str, Any]:
    """Capture environment information.
    
    Returns:
        Dictionary with environment details
    """
    env = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
    }

    # PyTorch
    try:
        import torch
        env["torch_version"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["cuda_version"] = torch.version.cuda
            env["cudnn_version"] = torch.backends.cudnn.version()
            env["gpu_count"] = torch.cuda.device_count()
            env["gpu_names"] = [
                torch.cuda.get_device_name(i)
                for i in range(torch.cuda.device_count())
            ]
    except ImportError:
        pass

    # Qiskit
    try:
        import qiskit
        env["qiskit_version"] = qiskit.__version__
    except ImportError:
        pass

    try:
        import qiskit_aer
        env["qiskit_aer_version"] = qiskit_aer.__version__
    except ImportError:
        pass

    # Other key packages
    for pkg in ["numpy", "pandas", "scipy", "scikit-learn", "wandb"]:
        try:
            mod = __import__(pkg)
            env[f"{pkg}_version"] = mod.__version__
        except ImportError:
            pass

    # Git hash
    try:
        import subprocess
        git_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=Path(__file__).parent.parent,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        env["git_hash"] = git_hash
    except Exception:
        pass

    return env


def config_hash(config: dict) -> str:
    """Generate deterministic hash for config.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        MD5 hash string
    """
    # Sort keys for deterministic ordering
    config_str = json.dumps(config, sort_keys=True, default=str)

    # Generate hash
    hash_obj = hashlib.md5(config_str.encode())

    return hash_obj.hexdigest()[:8]


def save_environment_info(output_dir: Path) -> Path:
    """Save environment info to file.
    
    Args:
        output_dir: Directory to save
    
    Returns:
        Path to saved file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = capture_environment()

    filepath = output_dir / "environment.json"
    with open(filepath, 'w') as f:
        json.dump(env, f, indent=2)

    return filepath


class ReproducibilityContext:
    """Context manager for reproducible execution."""

    def __init__(self, seed: int = 42, deterministic: bool = True):
        self.seed = seed
        self.deterministic = deterministic
        self.old_state = {}

    def __enter__(self):
        # Save current state
        self.old_state = {
            "random_state": random.getstate(),
            "numpy_state": np.random.get_state(),
        }

        try:
            import torch
            self.old_state["torch_state"] = torch.get_rng_state()
            if torch.cuda.is_available():
                self.old_state["torch_cuda_state"] = torch.cuda.get_rng_state_all()
        except ImportError:
            pass

        # Set seeds
        set_global_seed(self.seed, self.deterministic)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore state
        random.setstate(self.old_state.get("random_state"))
        np.random.set_state(self.old_state.get("numpy_state"))

        try:
            import torch
            torch.set_rng_state(self.old_state.get("torch_state"))
            if torch.cuda.is_available():
                torch.cuda.set_rng_state_all(self.old_state.get("torch_cuda_state"))
        except ImportError:
            pass

        return False


def get_gpu_info() -> dict[str, Any]:
    """Get GPU information.
    
    Returns:
        GPU info dictionary
    """
    info = {"available": False, "devices": []}

    try:
        import torch
        if torch.cuda.is_available():
            info["available"] = True
            info["count"] = torch.cuda.device_count()

            for i in range(torch.cuda.device_count()):
                info["devices"].append({
                    "id": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_total": torch.cuda.get_device_properties(i).total_memory,
                })
    except ImportError:
        pass

    return info
