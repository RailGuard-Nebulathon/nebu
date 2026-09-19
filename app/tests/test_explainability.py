import numpy as np

from railguard.data.synthetic import synthetic_features
from railguard.explainability.feature_importance import feature_importance
from railguard.explainability.neighbours import NeighbourIndex
from railguard.explainability.spectral_attribution import spectral_band_attribution
from railguard.models import create_model


def test_explanation_components() -> None:
    x, y = synthetic_features("door", 24)
    model = create_model("door", "extra_trees").fit(x, y)
    assert list(feature_importance(model, list(x.columns)))
    embeddings = np.random.default_rng(42).normal(size=(6, 3))
    index = NeighbourIndex(2).fit(embeddings, [{"sample_id": str(i)} for i in range(6)])
    assert len(index.query(embeddings[:1])[0]) == 2
    signal = np.sin(np.linspace(0, 10, 128))
    evidence = spectral_band_attribution(signal, 128, lambda value: float(np.var(value)), [(0, 10), (10, 64)])
    assert set(evidence) == {"0-10Hz", "10-64Hz"}

