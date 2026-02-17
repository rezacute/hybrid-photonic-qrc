"""
Training Callbacks

WandbCallback, CheckpointCallback, EarlyStoppingCallback.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
import json

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class Callback:
    """Base callback class."""
    
    def on_train_begin(self, logs: Dict = None):
        pass
    
    def on_train_end(self, logs: Dict = None):
        pass
    
    def on_epoch_begin(self, epoch: int, logs: Dict = None):
        pass
    
    def on_epoch_end(self, epoch: int, logs: Dict = None):
        pass
    
    def on_batch_begin(self, batch: int, logs: Dict = None):
        pass
    
    def on_batch_end(self, batch: int, logs: Dict = None):
        pass


class WandbCallback(Callback):
    """Callback for logging to Weights & Biases."""
    
    def __init__(self, metrics: Optional[List[str]] = None):
        """
        Args:
            metrics: List of metrics to log
        """
        self.metrics = metrics or ["loss", "val_loss"]
    
    def on_epoch_end(self, epoch: int, logs: Dict = None):
        """Log metrics to wandb."""
        if logs is None:
            return
        
        if not HAS_WANDB:
            return
        
        log_dict = {}
        for metric in self.metrics:
            if metric in logs:
                log_dict[metric] = logs[metric]
        
        if log_dict:
            wandb.log(log_dict, step=epoch)
    
    def on_train_end(self, logs: Dict = None):
        """Finish wandb run."""
        if HAS_WANDB:
            wandb.finish()


class CheckpointCallback(Callback):
    """Callback for saving model checkpoints."""
    
    def __init__(
        self,
        filepath: str,
        monitor: str = "val_loss",
        mode: str = "min",
        save_best_only: bool = True,
    ):
        """
        Args:
            filepath: Path to save checkpoint
            monitor: Metric to monitor
            mode: 'min' or 'max'
            save_best_only: Only save when metric improves
        """
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        
        if mode == "min":
            self.best = float('inf')
            self.is_better = lambda new, best: new < best
        else:
            self.best = float('-inf')
            self.is_better = lambda new, best: new > best
    
    def on_epoch_end(self, epoch: int, logs: Dict = None):
        """Save checkpoint if metric improved."""
        if logs is None:
            return
        
        current = logs.get(self.monitor)
        if current is None:
            return
        
        if self.save_best_only:
            if self.is_better(current, self.best):
                self.best = current
                self._save_checkpoint(epoch, logs)
        else:
            self._save_checkpoint(epoch, logs)
    
    def _save_checkpoint(self, epoch: int, logs: Dict):
        """Save checkpoint to file."""
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "epoch": epoch,
            "logs": logs,
        }
        
        # Add model state if available
        if "model_state" in logs:
            checkpoint["model_state"] = logs["model_state"]
        
        with open(self.filepath, 'w') as f:
            json.dump(checkpoint, f, indent=2)


class EarlyStoppingCallback(Callback):
    """Early stopping callback."""
    
    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 10,
        mode: str = "min",
        min_delta: float = 0.0,
        restore_best: bool = True,
    ):
        """
        Args:
            monitor: Metric to monitor
            patience: Number of epochs with no improvement
            mode: 'min' or 'max'
            min_delta: Minimum change to qualify as improvement
            restore_best: Restore model to best observed state
        """
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.restore_best = restore_best
        
        if mode == "min":
            self.best = float('inf')
            self.is_better = lambda new, best: new < best - min_delta
        else:
            self.best = float('-inf')
            self.is_better = lambda new, best: new > best + min_delta
        
        self.counter = 0
        self.should_stop = False
        self.best_epoch = 0
    
    def on_train_begin(self, logs: Dict = None):
        """Reset state."""
        self.counter = 0
        self.should_stop = False
        self.best_epoch = 0
    
    def on_epoch_end(self, epoch: int, logs: Dict = None):
        """Check for early stopping."""
        if logs is None:
            return
        
        current = logs.get(self.monitor)
        if current is None:
            return
        
        if self.is_better(current, self.best):
            self.best = current
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            self.should_stop = True
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            print(f"Best epoch: {self.best_epoch+1}, {self.monitor}: {self.best:.6f}")
    
    def get_best_epoch(self) -> int:
        """Get best epoch number."""
        return self.best_epoch


class LearningRateSchedulerCallback(Callback):
    """Learning rate scheduler callback."""
    
    def __init__(self, scheduler):
        """
        Args:
            scheduler: PyTorch learning rate scheduler
        """
        self.scheduler = scheduler
    
    def on_epoch_end(self, epoch: int, logs: Dict = None):
        """Step scheduler."""
        if logs is None:
            return
        
        val_loss = logs.get("val_loss")
        if val_loss is not None:
            if hasattr(self.scheduler, 'step'):
                self.scheduler.step(val_loss)
        else:
            if hasattr(self.scheduler, 'step'):
                self.scheduler.step()


class MetricHistoryCallback(Callback):
    """Track metrics history."""
    
    def __init__(self):
        self.history = {
            "train": {},
            "val": {},
        }
    
    def on_epoch_end(self, epoch: int, logs: Dict = None):
        """Record metrics."""
        if logs is None:
            return
        
        for key, value in logs.items():
            if key.startswith("val_"):
                metric_name = key[4:]
                if metric_name not in self.history["val"]:
                    self.history["val"][metric_name] = []
                self.history["val"][metric_name].append(value)
            else:
                if key not in self.history["train"]:
                    self.history["train"][key] = []
                self.history["train"][key].append(value)
    
    def get_history(self) -> Dict:
        """Get training history."""
        return self.history


class CallbackList:
    """Container for multiple callbacks."""
    
    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self.callbacks = callbacks or []
    
    def append(self, callback: Callback):
        self.callbacks.append(callback)
    
    def on_train_begin(self, logs: Dict = None):
        for cb in self.callbacks:
            cb.on_train_begin(logs)
    
    def on_train_end(self, logs: Dict = None):
        for cb in self.callbacks:
            cb.on_train_end(logs)
    
    def on_epoch_begin(self, epoch: int, logs: Dict = None):
        for cb in self.callbacks:
            cb.on_epoch_begin(epoch, logs)
    
    def on_epoch_end(self, epoch: int, logs: Dict = None):
        for cb in self.callbacks:
            cb.on_epoch_end(epoch, logs)
    
    def on_batch_begin(self, batch: int, logs: Dict = None):
        for cb in self.callbacks:
            cb.on_batch_begin(batch, logs)
    
    def on_batch_end(self, batch: int, logs: Dict = None):
        for cb in self.callbacks:
            cb.on_batch_end(batch, logs)
