"""PSD and spectral summaries."""

import numpy as np
from scipy.signal import welch


def welch_psd(values: np.ndarray, sampling_rate: float) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float).reshape(-1)
    return welch(values, fs=sampling_rate, nperseg=min(256, len(values)))


def spectral_entropy(power: np.ndarray) -> float:
    power = np.maximum(np.asarray(power, dtype=float), 0)
    probabilities = power / max(float(power.sum()), np.finfo(float).eps)
    probabilities = probabilities[probabilities > 0]
    return float(-(probabilities * np.log2(probabilities)).sum() / max(np.log2(max(len(power), 2)), 1))


def band_power(frequencies: np.ndarray, power: np.ndarray, low: float, high: float) -> float:
    mask = (frequencies >= low) & (frequencies < high)
    return float(np.trapz(power[mask], frequencies[mask])) if mask.sum() > 1 else 0.0
