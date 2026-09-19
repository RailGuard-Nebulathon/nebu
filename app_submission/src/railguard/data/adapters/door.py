"""Door continuous-stream adapter and configurable cycle segmentation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from railguard.data.adapters.base import BaseAdapter
from railguard.data.registry import register_adapter
from railguard.data.timestamp import parse_door_timestamps, timestamp_diagnostics
from railguard.types import SequenceSample

LOGGER = logging.getLogger(__name__)
DATETIME = "Datetime"
CONTINUOUS_CHANNELS = [
    "Motor current(mA)", "Motor Voltage(10mV)", "Motor electrodynamic force",
    "Door opening time(.1s)", "Door closing time(.1s)", "Door leaf position",
]
STATE_CHANNELS = [
    "Close command", "Open command", "DCSR", "DCSL", "DLSR", "DLSL",
    "Door Opened", "Door Locked", "Door is opening", "Door is closing",
]
REQUIRED_COLUMNS = [DATETIME, *CONTINUOUS_CHANNELS, *STATE_CHANNELS]


def validate_door_schema(frame: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Door CSV is missing required columns: {missing}")


def infer_cycle_boundaries(
    frame: pd.DataFrame, *, min_rows: int = 20, merge_gap_rows: int = 3, timestamp_gap_factor: float = 5.0
) -> list[tuple[int, int, str]]:
    """Find active open/close runs without using status labels."""
    validate_door_schema(frame)
    opening = frame["Door is opening"].fillna(0).to_numpy(dtype=float) > 0
    closing = frame["Door is closing"].fillna(0).to_numpy(dtype=float) > 0
    timestamps, errors = parse_door_timestamps(frame[DATETIME], strict=False)
    if not errors and len(timestamps) > 1:
        milliseconds = timestamps.astype("datetime64[ms]").astype(np.int64)
        deltas = np.diff(milliseconds)
        positive = deltas[deltas > 0]
        nominal = float(np.median(positive)) if len(positive) else 0.0
        split_points = np.flatnonzero(deltas > max(nominal * timestamp_gap_factor, nominal + 1)) + 1
        if len(split_points):
            edges = np.r_[0, split_points, len(frame)]
            boundaries = []
            for start, stop in zip(edges[:-1], edges[1:], strict=True):
                end = int(stop - 1)
                if end - int(start) + 1 < min_rows:
                    continue
                open_votes = int(opening[start : end + 1].sum())
                close_votes = int(closing[start : end + 1].sum())
                boundaries.append((int(start), end, "Open" if open_votes > close_votes else "Close"))
            return boundaries
    command = (frame["Open command"].fillna(0).to_numpy(dtype=float) > 0) | (
        frame["Close command"].fillna(0).to_numpy(dtype=float) > 0
    )
    active = opening | closing | command
    indexes = np.flatnonzero(active)
    if not len(indexes):
        return []
    groups: list[list[int]] = [[int(indexes[0])]]
    for index in indexes[1:]:
        if int(index) - groups[-1][-1] <= merge_gap_rows + 1:
            groups[-1].append(int(index))
        else:
            groups.append([int(index)])
    boundaries = []
    for group in groups:
        start, end = group[0], group[-1]
        if end - start + 1 < min_rows:
            continue
        open_votes = int(opening[start : end + 1].sum())
        close_votes = int(closing[start : end + 1].sum())
        operation = "Open" if open_votes > close_votes else "Close"
        boundaries.append((start, end, operation))
    return boundaries


class DoorAdapter(BaseAdapter):
    task = "door"

    def _read_stream(self, split: str) -> tuple[pd.DataFrame, np.ndarray]:
        path = self.root / ("Train.csv" if split == "train" else "Test.csv")
        if not path.exists():
            raise FileNotFoundError(f"Door {split} stream not found: {path}")
        frame = pd.read_csv(path)
        validate_door_schema(frame)
        parsed, errors = parse_door_timestamps(frame[DATETIME], strict=False)
        if errors:
            raise ValueError(f"Door stream contains malformed timestamps: {errors[:3]}")
        diagnostics = timestamp_diagnostics(parsed)
        if not diagnostics["monotonic"]:
            raise ValueError("Door timestamps are not monotonic")
        LOGGER.info("Door timestamp diagnostics: %s", diagnostics)
        return frame, parsed

    def samples(self, split: str = "train") -> Iterable[SequenceSample]:
        frame, timestamps = self._read_stream(split)
        channels = CONTINUOUS_CHANNELS + STATE_CHANNELS
        if split == "train":
            labels_path = self.root / "Train_Segments_Answer.csv"
            if not labels_path.exists():
                raise FileNotFoundError(f"Door segment labels not found: {labels_path}")
            labels = pd.read_csv(labels_path)
            lookup = {value: index for index, value in enumerate(frame[DATETIME].astype(str))}
            diagnostics: list[dict[str, object]] = []
            for row in labels.itertuples(index=False):
                if row.start_time not in lookup or row.end_time not in lookup:
                    raise ValueError(f"Exact Door segment boundary not present for {row.segment_id}")
                start, end = lookup[row.start_time], lookup[row.end_time]
                actual = end - start + 1
                match = actual == int(row.n_rows)
                diagnostics.append({"segment_id": row.segment_id, "expected": int(row.n_rows), "actual": actual, "exact": match})
                if not match:
                    LOGGER.warning("Door segment row-count mismatch: %s", diagnostics[-1])
                segment = frame.iloc[start : end + 1]
                yield SequenceSample(
                    sample_id=str(row.segment_id), values=segment[channels].to_numpy(dtype=np.float32),
                    timestamps=timestamps[start : end + 1], channel_names=channels,
                    metadata={"operation": row.operation, "start_time": row.start_time, "end_time": row.end_time, "row_count_diagnostic": diagnostics[-1]},
                    target=str(row.status),
                )
        else:
            for number, (start, end, operation) in enumerate(infer_cycle_boundaries(frame), 1):
                segment = frame.iloc[start : end + 1]
                yield SequenceSample(
                    sample_id=f"test_seg_{number:03d}", values=segment[channels].to_numpy(dtype=np.float32),
                    timestamps=timestamps[start : end + 1], channel_names=channels,
                    metadata={"operation": operation, "start_time": frame.iloc[start][DATETIME], "end_time": frame.iloc[end][DATETIME], "inferred": True},
                )


register_adapter("door", DoorAdapter)
