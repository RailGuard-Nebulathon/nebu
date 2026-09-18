"""Transparent directory bundles for trusted locally-created models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import yaml


def save_classical_bundle(path: str | Path, model: Any, config: dict[str, Any], schema: dict[str, Any], metadata: dict[str, Any]) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    (target / "metadata.json").write_text(json.dumps(metadata | model.metadata(), indent=2, default=str), encoding="utf-8")
    (target / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (target / "schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    (target / "feature_names.json").write_text(json.dumps(model.feature_names_, indent=2), encoding="utf-8")
    joblib.dump(model.pipeline.named_steps["imputer"], target / "preprocessor.joblib")
    joblib.dump(model, target / "model.joblib")
    return target


def load_classical_bundle(path: str | Path) -> tuple[Any, dict[str, Any]]:
    source = Path(path).resolve()
    required = ["metadata.json", "schema.json", "feature_names.json", "model.joblib"]
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete model bundle; missing {missing}")
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    model = joblib.load(source / "model.joblib")
    return model, metadata

