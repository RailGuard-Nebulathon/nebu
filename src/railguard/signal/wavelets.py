"""Optional wavelet-energy features."""

import numpy as np


def wavelet_energies(values: np.ndarray, wavelet: str = "db4", level: int = 3) -> dict[str, float]:
    try:
        import pywt
    except ImportError:
        return {}
    coefficients = pywt.wavedec(np.asarray(values, dtype=float), wavelet, level=level)
    return {f"wavelet_energy_{index}": float(np.square(coefficient).sum()) for index, coefficient in enumerate(coefficients)}

