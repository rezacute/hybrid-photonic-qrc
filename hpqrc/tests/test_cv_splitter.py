"""
Tests for CV splitters.
"""

import pytest
import numpy as np

from src.data.cv_splitter import (
    ExpandingWindowSplitter,
    SlidingWindowSplitter,
    SingleSplit,
)


def test_expanding_window_no_overlap():
    """Test that train/val/test don't overlap."""
    splitter = ExpandingWindowSplitter(
        n_folds=3,
        test_size=100,
        val_ratio=0.2,
        gap=10,
    )
    
    data = np.arange(1000)
    
    for train_idx, val_idx, test_idx in splitter.split(data):
        # Check no overlap
        assert len(set(train_idx) & set(val_idx)) == 0
        assert len(set(train_idx) & set(test_idx)) == 0
        assert len(set(val_idx) & set(test_idx)) == 0


def test_expanding_window_fold_count():
    """Test correct number of folds."""
    splitter = ExpandingWindowSplitter(
        n_folds=5,
        test_size=100,
    )
    
    data = np.arange(1000)
    
    folds = list(splitter.split(data))
    
    assert len(folds) <= 5


def test_expanding_window_test_size():
    """Test that test set has correct size."""
    splitter = ExpandingWindowSplitter(
        n_folds=3,
        test_size=100,
    )
    
    data = np.arange(1000)
    
    for train_idx, val_idx, test_idx in splitter.split(data):
        assert len(test_idx) == 100


def test_sliding_window_no_overlap():
    """Test sliding window has no overlap."""
    splitter = SlidingWindowSplitter(
        n_folds=3,
        window_size=200,
        test_size=50,
    )
    
    data = np.arange(500)
    
    for train_idx, test_idx in splitter.split(data):
        assert len(set(train_idx) & set(test_idx)) == 0


def test_single_split_proportions():
    """Test single split proportions."""
    splitter = SingleSplit(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
    )
    
    data = np.arange(1000)
    
    splits = list(splitter.split(data))
    train_idx, val_idx, test_idx = splits[0]
    
    # Check proportions (approximately)
    assert len(train_idx) / 1000 == pytest.approx(0.7, abs=0.01)
    assert len(val_idx) / 1000 == pytest.approx(0.15, abs=0.01)
    assert len(test_idx) / 1000 == pytest.approx(0.15, abs=0.01)


def test_single_split_no_overlap():
    """Test single split has no overlap."""
    splitter = SingleSplit(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
    )
    
    data = np.arange(1000)
    
    train_idx, val_idx, test_idx = next(splitter.split(data))
    
    assert len(set(train_idx) & set(val_idx)) == 0
    assert len(set(train_idx) & set(test_idx)) == 0
    assert len(set(val_idx) & set(test_idx)) == 0
