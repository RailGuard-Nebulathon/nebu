import numpy as np
import pandas as pd

from railguard.features.shm import shm_features
from railguard.models.acv import ACVRanker
from railguard.models.classical import ClassicalRegressor
from railguard.training.competition import _mape_scale
from railguard.types import SequenceSample


def test_log_regressor_and_mape_scale_are_positive() -> None:
    x = pd.DataFrame({"x": np.linspace(0, 1, 20), "z": np.linspace(1, 2, 20)})
    y = np.exp(x["x"].to_numpy())
    model = ClassicalRegressor("ridge", target_transform="log").fit(x, y)
    prediction = model.predict(x)
    assert np.all(prediction > 0)
    assert _mape_scale(y, prediction) > 0


def test_acv_ranker_combines_probabilities_and_peer_deviation() -> None:
    x = pd.DataFrame(
        {
            "absolute__temperature_mean": [20.0, 20.1, 19.9, 25.0] * 3,
            "peer_residual__temperature_mean": [0.0, 0.1, -0.1, 5.0] * 3,
            "peer_residual__target_error_mean": [0.0, 0.2, -0.2, 4.0] * 3,
        }
    )
    y = np.asarray([False, False, False, True] * 3)
    model = ACVRanker("logreg", supervised_weight=0.5).fit(x, y)
    probabilities = model.predict_proba(x.iloc[:4])
    assert probabilities.shape == (4, 2)
    assert int(np.argmax(probabilities[:, 1])) == 3


def test_shm_features_include_learnable_fatigue_exponents() -> None:
    time = np.linspace(0, 20 * np.pi, 4000)
    values = (2.0 * np.sin(time) + 0.25 * np.sin(7 * time))[:, None]
    features = shm_features(SequenceSample("stress", values, None, ["stress_0"]))
    assert all(np.isfinite(list(features.values())))
    assert "stress_0__fatigue_log_proxy_m8" in features
    assert features["stress_0__rainflow_cycle_count"] > 0
