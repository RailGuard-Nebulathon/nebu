"""Parser for the official non-zero-padded Door timestamp format."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Iterable

import numpy as np

DOOR_TIMESTAMP = re.compile(
    r"^(?P<year>\d{4})-(?P<month>\d{1,2})-(?P<day>\d{1,2})-"
    r"(?P<hour>\d{1,2})-(?P<minute>\d{1,2})-(?P<second>\d{1,2})-(?P<millisecond>\d{1,3})$"
)


def parse_door_timestamp(value: str) -> datetime:
    match = DOOR_TIMESTAMP.fullmatch(str(value).strip())
    if not match:
        raise ValueError(f"Malformed Door timestamp: {value!r}; expected YYYY-M-D-H-M-S-ms")
    parts = {key: int(item) for key, item in match.groupdict().items()}
    milliseconds = parts.pop("millisecond")
    try:
        return datetime(**parts, microsecond=milliseconds * 1000)
    except ValueError as exc:
        raise ValueError(f"Invalid Door timestamp {value!r}: {exc}") from exc


def parse_door_timestamps(values: Iterable[str], strict: bool = True) -> tuple[np.ndarray, list[dict[str, object]]]:
    parsed: list[np.datetime64] = []
    errors: list[dict[str, object]] = []
    for index, value in enumerate(values):
        try:
            parsed.append(np.datetime64(parse_door_timestamp(str(value)), "ms"))
        except ValueError as exc:
            errors.append({"row": index, "value": value, "error": str(exc)})
            parsed.append(np.datetime64("NaT"))
    if strict and errors:
        raise ValueError(f"Failed to parse {len(errors)} Door timestamps; first error: {errors[0]}")
    return np.asarray(parsed), errors


def elapsed_seconds(timestamps: np.ndarray) -> np.ndarray:
    valid = timestamps[~np.isnat(timestamps)]
    if not len(valid):
        return np.full(len(timestamps), np.nan)
    return (timestamps - valid[0]).astype("timedelta64[ms]").astype(float) / 1000.0


def timestamp_diagnostics(timestamps: np.ndarray) -> dict[str, object]:
    valid = timestamps[~np.isnat(timestamps)]
    delta = np.diff(valid).astype("timedelta64[ms]").astype(float) if len(valid) > 1 else np.array([])
    return {
        "n_values": int(len(timestamps)),
        "n_malformed": int(np.isnat(timestamps).sum()),
        "monotonic": bool(np.all(delta >= 0)),
        "n_duplicates": int(np.sum(delta == 0)),
        "sampling_interval_ms": {
            "median": float(np.median(delta)) if len(delta) else None,
            "min": float(np.min(delta)) if len(delta) else None,
            "max": float(np.max(delta)) if len(delta) else None,
        },
    }

