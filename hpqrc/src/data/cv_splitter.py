"""
Time Series Cross-Validation Splitters

Provides time-aware CV splits for model evaluation.
"""

import numpy as np
from typing import Iterator, Tuple
from dataclasses import dataclass


@dataclass
class ExpandingWindowConfig:
    """Configuration for expanding window CV."""
    n_folds: int = 5
    test_size: int = 672  # 1 week at 15-min resolution
    val_ratio: float = 0.15
    gap: int = 0
    min_history: int = 2016  # 3 weeks minimum


class ExpandingWindowSplitter:
    """Time-series cross-validation with expanding training window.
    
    Guarantees no future data leakage.
    """
    
    def __init__(
        self,
        n_folds: int = 5,
        test_size: int = 672,
        val_ratio: float = 0.15,
        gap: int = 0,
        min_history: int = 2016,
    ):
        """
        Args:
            n_folds: Number of CV folds
            test_size: Size of test set
            val_ratio: Validation set ratio (relative to test)
            gap: Gap between train/val and test
            min_history: Minimum training history required
        """
        self.n_folds = n_folds
        self.test_size = test_size
        self.val_ratio = val_ratio
        self.gap = gap
        self.min_history = min_history
    
    def split(self, data: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Generate train/val/test splits.
        
        Args:
            data: Input data array
        
        Yields:
            (train_idx, val_idx, test_idx) tuples
        """
        n = len(data)
        val_size = int(self.test_size * self.val_ratio)
        
        # Calculate total samples needed per fold
        samples_per_fold = self.test_size + self.gap
        
        # Calculate start points for each fold
        # We work backwards from the end
        max_test_start = n - self.test_size
        
        # Ensure minimum history
        min_train_end = self.min_history
        
        if max_test_start < min_train_end:
            raise ValueError(
                f"Data too short: n={n}, min_history={self.min_history}, "
                f"test_size={self.test_size}"
            )
        
        # Calculate available folds
        available_folds = (max_test_start - min_train_end) // samples_per_fold + 1
        actual_folds = min(self.n_folds, available_folds)
        
        for fold in range(actual_folds):
            # Test end index
            test_end = n - fold * samples_per_fold
            
            # Test start index
            test_start = test_end - self.test_size
            
            # Validation start
            val_start = test_start - val_size - self.gap
            
            # Train end
            train_end = val_start - self.gap
            
            # Generate indices
            train_idx = np.arange(0, train_end)
            val_idx = np.arange(train_end + self.gap, val_start + val_size)
            test_idx = np.arange(test_start, test_end)
            
            yield train_idx, val_idx, test_idx
    
    def get_n_splits(self) -> int:
        """Get number of splits."""
        return self.n_folds


class SlidingWindowSplitter:
    """Fixed-size sliding window CV."""
    
    def __init__(
        self,
        n_folds: int = 5,
        window_size: int = 4032,  # 3 weeks
        test_size: int = 672,
        gap: int = 0,
    ):
        self.n_folds = n_folds
        self.window_size = window_size
        self.test_size = test_size
        self.gap = gap
    
    def split(self, data: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate train/test splits."""
        n = len(data)
        
        for fold in range(self.n_folds):
            # Calculate window positions
            window_end = n - fold * (self.test_size + self.gap)
            window_start = window_end - self.window_size
            
            if window_start < 0:
                break
            
            train_idx = np.arange(window_start, window_end - self.test_size)
            test_idx = np.arange(window_end - self.test_size, window_end)
            
            yield train_idx, test_idx


class SingleSplit:
    """Simple train/val/test split (no CV)."""
    
    def __init__(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        gap: int = 0,
    ):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.gap = gap
    
    def split(self, data: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Generate single split."""
        n = len(data)
        
        train_end = int(n * self.train_ratio)
        val_end = train_end + int(n * self.val_ratio)
        
        train_idx = np.arange(0, train_end)
        val_idx = np.arange(train_end + self.gap, val_end)
        test_idx = np.arange(val_end + self.gap, n)
        
        yield train_idx, val_idx, test_idx
