"""Finite-sample split conformal regression intervals."""

import numpy as np


class SplitConformalRegressor:
    def __init__(self, coverage: float = 0.9) -> None:
        if not 0 < coverage < 1:
            raise ValueError("coverage must be between zero and one")
        self.coverage, self.quantile = coverage, None

    def fit(self, calibration_truth: np.ndarray, calibration_prediction: np.ndarray) -> "SplitConformalRegressor":
        residuals = np.abs(np.asarray(calibration_truth) - np.asarray(calibration_prediction))
        level = min(1.0, np.ceil((len(residuals) + 1) * self.coverage) / len(residuals))
        self.quantile = float(np.quantile(residuals, level, method="higher"))
        return self

    def predict_interval(self, prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.quantile is None:
            raise RuntimeError("Conformal interval is not calibrated")
        prediction = np.asarray(prediction)
        return prediction - self.quantile, prediction + self.quantile

