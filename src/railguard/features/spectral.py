"""Frequency-domain features with deterministic bands."""

import numpy as np

from railguard.signal.spectral import band_power, spectral_entropy, welch_psd


def spectral_features(values: np.ndarray, sampling_rate: float, bands: int = 4) -> dict[str, float]:
    frequencies, power = welch_psd(values, sampling_rate)
    total = float(np.trapz(power, frequencies)) if len(frequencies) > 1 else 0.0
    weighted = max(float(power.sum()), np.finfo(float).eps)
    centroid = float((frequencies * power).sum() / weighted)
    bandwidth = float(np.sqrt(((frequencies - centroid) ** 2 * power).sum() / weighted))
    output = {
        "dominant_frequency": float(frequencies[int(np.argmax(power))]), "spectral_centroid": centroid,
        "spectral_bandwidth": bandwidth, "spectral_entropy": spectral_entropy(power), "spectral_total_power": total,
        "high_frequency_ratio": float(power[len(power) // 2 :].sum() / weighted),
    }
    edges = np.linspace(0, sampling_rate / 2, bands + 1)
    output.update({f"band_power_{i}": band_power(frequencies, power, edges[i], edges[i + 1]) for i in range(bands)})
    return output
