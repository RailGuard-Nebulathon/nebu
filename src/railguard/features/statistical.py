"""Robust univariate statistics."""

import numpy as np
from scipy.stats import kurtosis, skew


def statistical_features(values: np.ndarray) -> dict[str, float]:
    x = np.asarray(values, dtype=float).reshape(-1)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError("Statistical features require a non-empty finite signal")
    mean_abs = float(np.mean(np.abs(x)))
    rms = float(np.sqrt(np.mean(np.square(x))))
    peak = float(np.max(np.abs(x)))
    median = float(np.median(x))
    is_constant = float(np.std(x)) <= np.finfo(float).eps
    return {
        "mean": float(np.mean(x)), "std": float(np.std(x)), "median": median,
        "mad": float(np.median(np.abs(x - median))), "min": float(np.min(x)), "max": float(np.max(x)),
        "range": float(np.ptp(x)), "rms": rms, "energy": float(np.square(x).sum()),
        "integral": float(np.trapz(x)), "positive_area": float(np.maximum(x, 0).sum()),
        "negative_area": float(np.minimum(x, 0).sum()), "peak_magnitude": peak,
        "peak_fraction": float(np.argmax(np.abs(x)) / max(len(x) - 1, 1)),
        "q05": float(np.quantile(x, 0.05)), "q25": float(np.quantile(x, 0.25)),
        "q75": float(np.quantile(x, 0.75)), "q95": float(np.quantile(x, 0.95)),
        "skew": 0.0 if is_constant else float(np.nan_to_num(skew(x), nan=0.0)),
        "kurtosis": 0.0 if is_constant else float(np.nan_to_num(kurtosis(x), nan=0.0)),
        "crest_factor": peak / max(rms, np.finfo(float).eps),
        "impulse_factor": peak / max(mean_abs, np.finfo(float).eps),
        "shape_factor": rms / max(mean_abs, np.finfo(float).eps),
    }
