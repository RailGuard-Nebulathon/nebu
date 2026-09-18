import numpy as np

from railguard.calibration.conformal import SplitConformalRegressor
from railguard.calibration.metrics import expected_calibration_error
from railguard.uncertainty.ensembles import classification_ensemble, regression_ensemble
from railguard.uncertainty.ood import EmbeddingOOD


def test_calibration_ensemble_and_ood() -> None:
    probabilities = np.array([[0.8, 0.2], [0.3, 0.7]])
    assert 0 <= expected_calibration_error(probabilities, np.array([0, 1])) <= 1
    assert classification_ensemble([probabilities, probabilities])[0].shape == probabilities.shape
    assert regression_ensemble([np.array([1.0]), np.array([1.2])])[1][0] > 0
    conformal = SplitConformalRegressor().fit(np.array([1.0, 2.0, 3.0]), np.array([0.9, 2.2, 2.8]))
    lower, upper = conformal.predict_interval(np.array([2.0]))
    assert lower < 2 < upper
    training = np.random.default_rng(42).normal(size=(30, 4))
    ood = EmbeddingOOD().fit(training)
    assert ood.is_ood(np.array([[100, 100, 100, 100]])).item()

