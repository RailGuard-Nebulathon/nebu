"""Explicit finite-value cleaning policies."""

import numpy as np


def clean_signal(values: np.ndarray, max_missing_fraction: float = 0.2) -> np.ndarray:
    array = np.asarray(values, dtype=float).copy()
    array[~np.isfinite(array)] = np.nan
    if array.ndim == 1:
        array = array[:, None]
        squeeze = True
    else:
        squeeze = False
    for column in range(array.shape[1]):
        series = array[:, column]
        missing = np.isnan(series)
        if missing.mean() > max_missing_fraction:
            raise ValueError(f"Channel {column} missing fraction {missing.mean():.3f} exceeds limit")
        if missing.any():
            valid = np.flatnonzero(~missing)
            if not len(valid):
                raise ValueError(f"Channel {column} contains no finite observations")
            series[missing] = np.interp(np.flatnonzero(missing), valid, series[valid])
    return array[:, 0] if squeeze else array

