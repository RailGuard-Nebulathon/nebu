"""Configurable zero-phase filters."""

import numpy as np
from scipy.signal import butter, sosfiltfilt


def butterworth(values: np.ndarray, sampling_rate: float, low: float | None = None, high: float | None = None, order: int = 4) -> np.ndarray:
    if low is None and high is None:
        return np.asarray(values, dtype=float).copy()
    nyquist = sampling_rate / 2
    if low is None:
        kind, cutoff = "lowpass", high / nyquist
    elif high is None:
        kind, cutoff = "highpass", low / nyquist
    else:
        kind, cutoff = "bandpass", [low / nyquist, high / nyquist]
    return sosfiltfilt(butter(order, cutoff, btype=kind, output="sos"), values, axis=0)

