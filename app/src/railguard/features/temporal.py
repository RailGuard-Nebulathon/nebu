"""Derivative, peak and temporal-shape features."""

import numpy as np
from scipy.signal import find_peaks

from railguard.signal.derivatives import derivative


def temporal_features(values: np.ndarray) -> dict[str, float]:
    x = np.asarray(values, dtype=float).reshape(-1)
    dx = derivative(x)
    peaks, _ = find_peaks(np.abs(x))
    signs = np.signbit(x)
    return {
        "derivative_mean": float(np.mean(dx)), "derivative_std": float(np.std(dx)),
        "derivative_max_abs": float(np.max(np.abs(dx))), "local_peak_count": float(len(peaks)),
        "zero_crossing_rate": float(np.mean(signs[1:] != signs[:-1])) if len(x) > 1 else 0.0,
        "lag1_autocorrelation": float(np.corrcoef(x[:-1], x[1:])[0, 1])
        if len(x) > 2 and np.std(x[:-1]) > 0 and np.std(x[1:]) > 0
        else 0.0,
    }
