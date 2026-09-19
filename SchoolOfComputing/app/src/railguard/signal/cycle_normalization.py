"""Phase-normalised cycles with masks."""

import numpy as np

from railguard.signal.resampling import resample_length


def phase_normalize(values: np.ndarray, length: int = 192) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normalized = resample_length(values, length).astype(np.float32)
    phase = np.linspace(0.0, 1.0, length, dtype=np.float32)
    mask = np.ones(length, dtype=bool)
    return normalized, phase, mask

