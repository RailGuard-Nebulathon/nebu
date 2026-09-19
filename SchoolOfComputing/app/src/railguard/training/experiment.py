"""Local, hosted-service-free experiment artifacts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from railguard.utils.environment import environment_metadata


class ExperimentRun:
    def __init__(self, root: str | Path, task: str, model: str, seed: int) -> None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self.experiment_id = f"{task}-{model}-{stamp}-s{seed}"
        self.path = Path(root) / self.experiment_id
        (self.path / "plots").mkdir(parents=True, exist_ok=True)
        self.metadata = {"experiment_id": self.experiment_id, "task": task, "model": model, "seed": seed} | environment_metadata()

    def write(self, config: dict[str, Any], metrics: dict[str, Any], folds: list[dict[str, Any]] | None = None, predictions: pd.DataFrame | None = None, schema_hashes: dict[str, str] | None = None, feature_names: list[str] | None = None) -> Path:
        (self.path / "config_resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        for name, value in (("metadata.json", self.metadata), ("metrics.json", metrics), ("environment.json", environment_metadata()), ("schema_hashes.json", schema_hashes or {}), ("feature_names.json", feature_names or [])):
            (self.path / name).write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
        pd.DataFrame(folds or []).to_csv(self.path / "fold_metrics.csv", index=False)
        (predictions if predictions is not None else pd.DataFrame()).to_csv(self.path / "predictions.csv", index=False)
        return self.path

