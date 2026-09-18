"""High-level trusted-bundle inference API."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from railguard.data.adapters.acv import split_car_columns
from railguard.data.adapters.corrugation import SPEED_COLUMN
from railguard.data.adapters.door import (
    CONTINUOUS_CHANNELS,
    STATE_CHANNELS,
    infer_cycle_boundaries,
    validate_door_schema,
)
from railguard.data.timestamp import parse_door_timestamps
from railguard.features.acv import acv_candidate_features
from railguard.features.corrugation import corrugation_features
from railguard.features.door import door_feature_table
from railguard.features.shm import shm_features
from railguard.models.serialization import load_classical_bundle
from railguard.types import Prediction, SequenceSample


class RailGuardPredictor:
    def __init__(self, model, metadata: dict[str, object], schema: dict[str, object]) -> None:
        self.model, self.metadata, self.schema = model, metadata, schema
        self.task = str(metadata.get("task", ""))

    @classmethod
    def from_bundle(cls, path: str | Path) -> RailGuardPredictor:
        source = Path(path)
        model, metadata = load_classical_bundle(source)
        schema = json.loads((source / "schema.json").read_text(encoding="utf-8"))
        if schema.get("feature_names") and schema["feature_names"] != model.feature_names_:
            raise ValueError("Bundle schema feature order is incompatible with model metadata")
        return cls(model, metadata, schema)

    def _frame(self, input_path: str | Path) -> tuple[pd.DataFrame, list[dict[str, object]]]:
        path = Path(input_path)
        if self.task == "door":
            raw = pd.read_csv(path)
            validate_door_schema(raw)
            timestamps, errors = parse_door_timestamps(raw["Datetime"], strict=False)
            if errors:
                raise ValueError(errors[:3])
            samples, records = [], []
            channels = CONTINUOUS_CHANNELS + STATE_CHANNELS
            for number, (start, end, operation) in enumerate(infer_cycle_boundaries(raw), 1):
                samples.append(
                    SequenceSample(
                        f"segment_{number}",
                        raw.iloc[start : end + 1][channels].to_numpy(np.float32),
                        timestamps[start : end + 1],
                        channels,
                        {"operation": operation},
                    )
                )
                records.append(
                    {
                        "sample_id": f"segment_{number}",
                        "start_time": raw.iloc[start]["Datetime"],
                        "end_time": raw.iloc[end]["Datetime"],
                    }
                )
            return door_feature_table(samples), records
        if self.task == "acv":
            raw = pd.read_excel(path)
            _, cars = split_car_columns(raw)
            frame = acv_candidate_features(raw, path.name)
            return frame.drop(columns=["case_id", "car_id"]), [
                {"sample_id": car, "file_id": path.name} for car in frame["car_id"]
            ]
        if self.task == "corrugation":
            raw = pd.read_csv(path)
            channels = [str(c) for c in raw if c != SPEED_COLUMN]
            sample = SequenceSample(
                path.name,
                raw[channels].to_numpy(np.float32),
                None,
                channels,
                {
                    "sampling_rate_hz": 10000,
                    "rotating_speed": raw[SPEED_COLUMN].to_numpy(np.float32),
                },
            )
            return pd.DataFrame([corrugation_features(sample)]), [
                {"sample_id": path.name, "file_id": path.name}
            ]
        if self.task == "shm":
            raw = pd.read_csv(path, header=None)
            channels = [f"stress_{i}" for i in range(raw.shape[1])]
            sample = SequenceSample(path.name, raw.to_numpy(np.float32), None, channels)
            return pd.DataFrame([shm_features(sample)]), [
                {"sample_id": path.name, "file_id": path.name}
            ]
        raise ValueError(f"Unsupported bundle task: {self.task}")

    def predict(self, task: str, input_path: str | Path) -> list[Prediction]:
        if task != self.task:
            raise ValueError(f"Bundle task is {self.task}, not {task}")
        features, records = self._frame(input_path)
        features = features.reindex(columns=self.model.feature_names_)
        if features.isna().all(axis=0).any():
            missing = features.columns[features.isna().all()].tolist()
            raise ValueError(f"Input cannot provide required bundle features: {missing[:10]}")
        prediction = self.model.predict(features)
        probabilities = (
            self.model.predict_proba(features) if hasattr(self.model, "predict_proba") else None
        )
        classes = []
        if probabilities is not None:
            classes = list(getattr(self.model, "classes_", []))
            if not classes and hasattr(self.model, "pipeline"):
                classes = list(self.model.pipeline.named_steps["model"].classes_)
        return [
            Prediction(
                sample_id=str(records[i]["sample_id"]),
                task=task,
                prediction=value,
                probabilities={
                    str(label): float(probabilities[i, j]) for j, label in enumerate(classes)
                }
                if probabilities is not None
                else None,
                confidence=float(probabilities[i].max()) if probabilities is not None else None,
                metadata=records[i],
            )
            for i, value in enumerate(prediction)
        ]
