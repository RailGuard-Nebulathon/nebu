"""Numerically stable temporal derivatives."""

import numpy as np


def derivative(values: np.ndarray, dt: float | np.ndarray = 1.0, order: int = 1) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    for _ in range(order):
        result = np.gradient(result, dt, axis=0)
    return result

