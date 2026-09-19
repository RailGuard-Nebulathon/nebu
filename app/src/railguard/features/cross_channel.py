"""Cross-channel correlation summaries."""

import numpy as np


def correlation_features(values: np.ndarray) -> dict[str, float]:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] < 2:
        return {"correlation_abs_mean": 0.0, "correlation_abs_max": 0.0}
    correlation = np.nan_to_num(np.corrcoef(matrix, rowvar=False))
    upper = np.abs(correlation[np.triu_indices(matrix.shape[1], 1)])
    return {"correlation_abs_mean": float(upper.mean()), "correlation_abs_max": float(upper.max())}

