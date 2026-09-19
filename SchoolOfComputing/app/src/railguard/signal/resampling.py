"""Deterministic sequence resampling."""

import numpy as np


def resample_length(values: np.ndarray, length: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if length < 2 or len(array) < 2:
        raise ValueError("Source and destination lengths must be at least two")
    old = np.linspace(0.0, 1.0, len(array))
    new = np.linspace(0.0, 1.0, length)
    if array.ndim == 1:
        return np.interp(new, old, array)
    return np.column_stack([np.interp(new, old, array[:, column]) for column in range(array.shape[1])])

