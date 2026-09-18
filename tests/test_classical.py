import numpy as np
import pytest

from railguard.data.synthetic import synthetic_features
from railguard.models import create_model


@pytest.mark.parametrize("task", ["door", "acv", "corrugation"])
def test_synthetic_classifiers(task: str) -> None:
    x, y = synthetic_features(task, 36)
    model = create_model(task, "logreg").fit(x, y)
    assert model.predict(x).shape == (36,)
    assert model.predict_proba(x).shape[0] == 36
    with pytest.raises(ValueError, match="schema"):
        model.predict(x[x.columns[::-1]])


def test_synthetic_regressor() -> None:
    x, y = synthetic_features("shm", 36)
    model = create_model("shm", "ridge", target_transform="log1p").fit(x, y)
    assert np.isfinite(model.predict(x)).all()

