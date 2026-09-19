"""One-sided FFT power spectra."""

import numpy as np


def power_spectrum(values: np.ndarray, sampling_rate: float) -> tuple[np.ndarray, np.ndarray]:
    signal = np.asarray(values, dtype=float).reshape(-1)
    if len(signal) < 2 or sampling_rate <= 0:
        raise ValueError("FFT requires at least two points and positive sampling_rate")
    centered = signal - np.mean(signal)
    spectrum = np.fft.rfft(centered)
    power = np.abs(spectrum) ** 2 / len(signal)
    return np.fft.rfftfreq(len(signal), d=1.0 / sampling_rate), power

