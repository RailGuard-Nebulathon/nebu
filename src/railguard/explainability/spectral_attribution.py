"""Frequency-band occlusion evidence."""

from collections.abc import Callable

import numpy as np


def spectral_band_attribution(signal: np.ndarray, sampling_rate: float, score: Callable[[np.ndarray], float], bands: list[tuple[float, float]]) -> dict[str, float]:
    x = np.asarray(signal, dtype=float)
    spectrum = np.fft.rfft(x)
    frequencies = np.fft.rfftfreq(len(x), 1 / sampling_rate)
    baseline = float(score(x))
    output = {}
    for low, high in bands:
        modified = spectrum.copy()
        modified[(frequencies >= low) & (frequencies < high)] = 0
        output[f"{low:g}-{high:g}Hz"] = baseline - float(score(np.fft.irfft(modified, n=len(x))))
    return output

