"""
Logging Utilities

Centralized logging for HPQRC project.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """Setup logger with console and optional file handler.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional file path for file handler
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Format
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get logger by name.
    
    Args:
        name: Logger name
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# Default logger for HPQRC
hpqrc_logger = setup_logger("hpqrc")


def log_experiment_start(config: dict) -> None:
    """Log experiment start with config.
    
    Args:
        config: Experiment configuration
    """
    hpqrc_logger.info("=" * 50)
    hpqrc_logger.info("Starting experiment")
    hpqrc_logger.info(f"Config: {config}")
    hpqrc_logger.info("=" * 50)


def log_experiment_end(metrics: dict) -> None:
    """Log experiment end with metrics.
    
    Args:
        metrics: Results metrics
    """
    hpqrc_logger.info("=" * 50)
    hpqrc_logger.info("Experiment completed")
    hpqrc_logger.info(f"Metrics: {metrics}")
    hpqrc_logger.info("=" * 50)
