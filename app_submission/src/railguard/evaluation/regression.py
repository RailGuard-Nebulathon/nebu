"""Regression metrics including the official SHM score."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score


def regression_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    truth, prediction = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    safe = np.abs(truth) > np.finfo(float).eps
    mape = float(np.mean(np.abs(truth[safe] - prediction[safe]) / np.abs(truth[safe]))) if safe.any() else float("nan")
    return {
        "mae": float(mean_absolute_error(truth, prediction)), "rmse": float(np.sqrt(mean_squared_error(truth, prediction))),
        "r2": float(r2_score(truth, prediction)), "median_absolute_error": float(median_absolute_error(truth, prediction)),
        "spearman": float(spearmanr(truth, prediction).statistic), "mape": mape, "shm_score": max(0.0, 1.0 - mape),
    }

