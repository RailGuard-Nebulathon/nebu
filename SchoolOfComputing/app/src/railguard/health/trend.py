"""Robust health trend estimation."""

import numpy as np
from scipy.stats import theilslopes


def health_trend(times: np.ndarray, scores: np.ndarray) -> dict[str, float | str]:
    if len(times) < 2:
        return {"slope": 0.0, "direction": "insufficient_history"}
    slope = float(theilslopes(scores, times).slope)
    direction = "degrading" if slope < 0 else "improving" if slope > 0 else "stable"
    return {"slope": slope, "direction": direction}

