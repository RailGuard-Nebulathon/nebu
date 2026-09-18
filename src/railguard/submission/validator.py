"""Strict official output and archive validation."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from railguard.data.timestamp import parse_door_timestamp
from railguard.submission.schemas import OFFICIAL_COLUMNS, OFFICIAL_FILES


def _timestamp(value: str):
    try:
        return parse_door_timestamp(str(value))
    except ValueError:
        parsed = pd.to_datetime(value, errors="raise")
        return parsed.to_pydatetime()


def validate_prediction_frame(task: str, frame: pd.DataFrame, expected_ids: Iterable[str] | None = None, expected_cars: dict[str, set[str]] | None = None) -> None:
    required = OFFICIAL_COLUMNS[task]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"{task} submission missing columns: {missing}")
    if task == "door":
        if frame.empty:
            raise ValueError("Door submission must contain predicted segments")
        for row in frame.itertuples(index=False):
            if row.prediction not in {"Normal", "Abnormal resistance"}:
                raise ValueError(f"Invalid Door label: {row.prediction}")
            if _timestamp(row.start_time) >= _timestamp(row.end_time):
                raise ValueError("Door segment start_time must precede end_time")
        if frame[["start_time", "end_time"]].duplicated().any():
            raise ValueError("Duplicate Door segment boundaries")
        return
    if frame["file_id"].duplicated().any():
        raise ValueError(f"Duplicate {task} file_id values")
    if expected_ids is not None:
        expected, actual = set(expected_ids), set(frame["file_id"].astype(str))
        if missing_ids := expected - actual:
            raise ValueError(f"Missing IDs: {sorted(missing_ids)}")
        if extra_ids := actual - expected:
            raise ValueError(f"Extra IDs: {sorted(extra_ids)}")
    if task == "acv":
        for row in frame.itertuples(index=False):
            ranking = str(row.ranked_cars).split("|")
            if any(not car for car in ranking) or len(ranking) != len(set(ranking)):
                raise ValueError(f"ACV ranking for {row.file_id} must list unique non-empty car IDs")
            if expected_cars and set(ranking) != expected_cars[str(row.file_id)]:
                raise ValueError(f"ACV ranking for {row.file_id} does not contain every exact car identifier")
    elif task == "corrugation":
        invalid = set(frame["prediction"]) - {"Normal", "Side I", "Side II"}
        if invalid:
            raise ValueError(f"Invalid corrugation labels: {sorted(invalid)}")
    elif task == "shm":
        numeric = pd.to_numeric(frame["prediction"], errors="coerce")
        if numeric.isna().any() or not np.isfinite(numeric).all():
            raise ValueError("SHM predictions must be finite numeric values")


def validate_prediction_file(path: str | Path, task: str | None = None, **kwargs) -> pd.DataFrame:
    source = Path(path)
    if task is None:
        reverse = {value: key for key, value in OFFICIAL_FILES.items()}
        try:
            task = reverse[source.name]
        except KeyError as exc:
            raise ValueError(f"Unknown official prediction filename: {source.name}") from exc
    frame = pd.read_csv(source, dtype={"file_id": str, "ranked_cars": str})
    validate_prediction_frame(task, frame, **kwargs)
    return frame


def validate_zip(path: str | Path) -> list[str]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if not names or any("/" in name or "\\" in name for name in names):
            raise ValueError("predictions.zip must contain CSVs directly at the top level")
        allowed = set(OFFICIAL_FILES.values())
        if invalid := set(names) - allowed:
            raise ValueError(f"Unexpected archive files: {sorted(invalid)}")
        reverse = {value: key for key, value in OFFICIAL_FILES.items()}
        for name in names:
            frame = pd.read_csv(io.BytesIO(archive.read(name)), dtype={"file_id": str, "ranked_cars": str})
            validate_prediction_frame(reverse[name], frame)
    return names

