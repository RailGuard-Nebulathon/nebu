"""Short-time Fourier transforms."""

import numpy as np
from scipy.signal import stft as scipy_stft


def stft_magnitude(values: np.ndarray, sampling_rate: float, nperseg: int = 128) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float).reshape(-1)
    frequencies, times, coefficients = scipy_stft(values, fs=sampling_rate, nperseg=min(nperseg, len(values)))
    return frequencies, times, np.abs(coefficients)

