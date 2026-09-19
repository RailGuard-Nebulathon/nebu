"""Calibration metrics."""

import numpy as np
from sklearn.metrics import brier_score_loss


def expected_calibration_error(probabilities: np.ndarray, targets: np.ndarray, bins: int = 10) -> float:
    probabilities, targets = np.asarray(probabilities), np.asarray(targets)
    confidence, prediction = probabilities.max(axis=1), probabilities.argmax(axis=1)
    edges = np.linspace(0, 1, bins + 1)
    error = 0.0
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        selected = (confidence > low) & (confidence <= high)
        if selected.any():
            error += selected.mean() * abs(float((prediction[selected] == targets[selected]).mean()) - float(confidence[selected].mean()))
    return float(error)


def binary_brier(probabilities: np.ndarray, targets: np.ndarray) -> float:
    return float(brier_score_loss(targets, probabilities))

