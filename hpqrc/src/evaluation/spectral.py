"""
Spectral Analysis of Residuals

Analyzes frequency content of prediction residuals.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import signal


def residual_spectral_analysis(
    residuals: np.ndarray,
    sampling_rate_hz: float = 1.0,
    expected_frequencies: Optional[List[float]] = None,
) -> Dict:
    """Analyze frequency content of residuals.
    
    Detects how well model captures periodic components.
    
    Args:
        residuals: Prediction residuals
        sampling_rate_hz: Sampling rate in Hz
        expected_frequencies: List of expected frequencies to check
    
    Returns:
        Dictionary with frequencies, power spectrum, detected peaks, captured flags
    """
    n = len(residuals)
    
    # Compute FFT
    fft_vals = np.fft.rfft(residuals)
    freqs = np.fft.rfftfreq(n, d=1.0 / sampling_rate_hz)
    power = np.abs(fft_vals) ** 2
    
    # Normalize power spectrum
    power_norm = power / (power.sum() + 1e-8)
    
    # Peak detection
    peaks, properties = signal.find_peaks(power, height=np.max(power) * 0.1)
    
    peak_freqs = freqs[peaks]
    peak_powers = power[peaks]
    
    # Default expected frequencies (1/day, 1/week in cycles per sample)
    if expected_frequencies is None:
        # Assuming 15-min data (96 samples/day)
        samples_per_day = 96
        samples_per_week = samples_per_day * 7
        
        expected_frequencies = [
            1.0 / samples_per_day,    # Daily
            1.0 / samples_per_week,   # Weekly
        ]
    
    # Check if expected frequencies are captured
    captured = {}
    tol = 0.1  # 10% tolerance
    
    for exp_freq in expected_frequencies:
        # Find closest peak
        if len(peak_freqs) > 0:
            idx = np.argmin(np.abs(peak_freqs - exp_freq))
            diff = np.abs(peak_freqs[idx] - exp_freq) / (exp_freq + 1e-8)
            
            captured[f"freq_{exp_freq:.6f}"] = diff < tol
        else:
            captured[f"freq_{exp_freq:.6f}"] = False
    
    # Total power in expected frequencies
    expected_power = 0.0
    for exp_freq in expected_frequencies:
        idx = np.argmin(np.abs(freqs - exp_freq))
        expected_power += power[idx]
    
    capture_ratio = expected_power / (power.sum() + 1e-8)
    
    return {
        "frequencies": freqs.tolist(),
        "power_spectrum": power.tolist(),
        "peak_frequencies": peak_freqs.tolist(),
        "peak_powers": peak_powers.tolist(),
        "expected_frequencies": expected_frequencies,
        "captured": captured,
        "capture_ratio": float(capture_ratio),
        "total_variance": float(np.var(residuals)),
        "spectral_centroid": float(np.sum(freqs * power_norm)),
    }


def spectral_entropy(power_spectrum: np.ndarray) -> float:
    """Compute spectral entropy.
    
    Args:
        power_spectrum: Power spectral density
    
    Returns:
        Spectral entropy
    """
    # Normalize
    p = power_spectrum / (power_spectrum.sum() + 1e-8)
    
    # Shannon entropy
    entropy = -np.sum(p * np.log2(p + 1e-8))
    
    return float(entropy)


def compare_spectral(
    residuals_model: np.ndarray,
    residuals_baseline: np.ndarray,
    sampling_rate_hz: float = 1.0,
) -> Dict:
    """Compare spectral properties of two residual sets.
    
    Args:
        residuals_model: Residuals from proposed model
        residuals_baseline: Residuals from baseline model
        sampling_rate_hz: Sampling rate
    
    Returns:
        Comparison dictionary
    """
    analysis_model = residual_spectral_analysis(residuals_model, sampling_rate_hz)
    analysis_baseline = residual_spectral_analysis(residuals_baseline, sampling_rate_hz)
    
    # Variance reduction
    var_reduction = 1 - (np.var(residuals_model) / (np.var(residuals_baseline) + 1e-8))
    
    # Spectral entropy comparison
    ent_model = spectral_entropy(np.array(analysis_model["power_spectrum"]))
    ent_baseline = spectral_entropy(np.array(analysis_baseline["power_spectrum"]))
    
    return {
        "variance_model": analysis_model["total_variance"],
        "variance_baseline": analysis_baseline["total_variance"],
        "variance_reduction": float(var_reduction),
        "entropy_model": ent_model,
        "entropy_baseline": ent_baseline,
        "entropy_change": ent_model - ent_baseline,
        "capture_ratio_model": analysis_model["capture_ratio"],
        "capture_ratio_baseline": analysis_baseline["capture_ratio"],
    }


def periodogram(data: np.ndarray, fs: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Compute periodogram (power spectral density estimate).
    
    Args:
        data: Time series data
        fs: Sampling frequency
    
    Returns:
        (frequencies, power)
    """
    f, Pxx = signal.periodogram(data, fs=fs)
    return f, Pxx


def spectrogram(
    data: np.ndarray,
    fs: float = 1.0,
    window: str = "hann",
    nperseg: int = 256,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute spectrogram.
    
    Args:
        data: Time series
        fs: Sampling frequency
        window: Window type
        nperseg: Segment length
    
    Returns:
        (frequencies, times, Sxx)
    """
    f, t, Sxx = signal.spectrogram(data, fs=fs, window=window, nperseg=nperseg)
    return f, t, Sxx
