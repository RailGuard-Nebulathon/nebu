"""Binary isotonic calibration hook with minimum-data guard."""

import numpy as np
from sklearn.isotonic import IsotonicRegression


class IsotonicCalibrator:
    def __init__(self, minimum_samples: int = 30) -> None:
        self.minimum_samples = minimum_samples
        self.model = IsotonicRegression(out_of_bounds="clip")
        self.fitted = False

    def fit(self, probabilities: np.ndarray, targets: np.ndarray) -> "IsotonicCalibrator":
        if len(probabilities) < self.minimum_samples or len(np.unique(targets)) < 2:
            raise ValueError("Insufficient held-out calibration data for isotonic regression")
        self.model.fit(probabilities, targets); self.fitted = True
        return self

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Isotonic calibrator is not fitted")
        return self.model.predict(probabilities)

