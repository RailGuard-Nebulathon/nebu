from pathlib import Path

import pandas as pd

from railguard.data.manifest import build_manifest
from railguard.data.synthetic import synthetic_features
from railguard.evaluation import classification_metrics, regression_metrics
from railguard.models import create_model


def test_all_tasks_end_to_end_synthetic() -> None:
    manifest = build_manifest([Path("tests/fixtures/sample.csv")], "shm", "train", {"sample.csv": 0.2})
    assert manifest.loc[0, "label"] == 0.2
    rows = []
    for task in ("door", "acv", "corrugation", "shm"):
        features, target = synthetic_features(task, 30)
        name = "ridge" if task == "shm" else "logreg"
        model = create_model(task, name).fit(features, target)
        prediction = model.predict(features)
        metrics = regression_metrics(target, prediction) if task == "shm" else classification_metrics(target, prediction, model.predict_proba(features))
        assert metrics
        rows.extend({"task": task, "sample_id": str(index), "prediction": value} for index, value in enumerate(prediction))
    output = Path("data/manifests/synthetic_predictions.csv")
    pd.DataFrame(rows).to_csv(output, index=False)
    assert output.exists() and len(pd.read_csv(output)) == 120

