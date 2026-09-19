"""Template-driven, row-order-preserving submission writer."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from railguard.submission.schemas import OFFICIAL_COLUMNS, OFFICIAL_FILES
from railguard.submission.validator import validate_prediction_frame


def write_submission(task: str, predictions: pd.DataFrame, output_dir: str | Path, template: str | Path | None = None) -> Path:
    if task not in OFFICIAL_FILES:
        raise ValueError(f"Unknown submission task: {task}")
    frame = predictions.copy()
    if template is not None:
        official = pd.read_csv(template, dtype={"file_id": str})
        if task in {"acv", "corrugation", "shm"}:
            if "file_id" not in official:
                raise ValueError("Official template lacks file_id")
            expected = official["file_id"].astype(str).tolist()
            validate_prediction_frame(task, frame, expected_ids=expected)
            frame = official[["file_id"]].merge(frame, on="file_id", how="left", validate="one_to_one")
    validate_prediction_frame(task, frame)
    target = Path(output_dir) / OFFICIAL_FILES[task]
    target.parent.mkdir(parents=True, exist_ok=True)
    frame[OFFICIAL_COLUMNS[task]].to_csv(target, index=False)
    return target

