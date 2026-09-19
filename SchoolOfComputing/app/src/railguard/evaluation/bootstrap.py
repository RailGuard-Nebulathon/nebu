"""Sample-level bootstrap confidence intervals."""

import numpy as np


def bootstrap_interval(y_true: np.ndarray, y_pred: np.ndarray, metric, samples: int = 1000, confidence: float = 0.95, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed); values = []
    for _ in range(samples):
        indexes = rng.integers(0, len(y_true), len(y_true)); values.append(metric(y_true[indexes], y_pred[indexes]))
    alpha = (1 - confidence) / 2
    return float(np.quantile(values, alpha)), float(np.quantile(values, 1 - alpha))

