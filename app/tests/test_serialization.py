from pathlib import Path

from railguard.data.synthetic import synthetic_features
from railguard.models import create_model
from railguard.models.serialization import load_classical_bundle, save_classical_bundle


def test_bundle_save_load_and_schema_rejection() -> None:
    x, y = synthetic_features("door", 24)
    model = create_model("door", "logreg").fit(x, y)
    target = Path("data/manifests/test_bundle")
    save_classical_bundle(target, model, {}, {"feature_names": list(x.columns)}, {"task": "door"})
    loaded, metadata = load_classical_bundle(target)
    assert metadata["task"] == "door"
    assert (loaded.predict(x) == model.predict(x)).all()

