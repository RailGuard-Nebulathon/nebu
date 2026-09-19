"""Deterministic metadata extraction for the four PS3 upload formats."""

from __future__ import annotations

import io
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel

TaskName = Literal["door", "acv", "corrugation", "shm"]
MetadataSource = Literal["embedded", "filename", "file_modified"]


class MetadataSuggestion(BaseModel):
    asset_id: str
    component_info: str
    measurement_time: datetime
    asset_source: MetadataSource
    component_source: MetadataSource
    measurement_time_source: MetadataSource
    warnings: list[str]


def _fallback_time(file_modified_ms: float | None) -> datetime:
    if file_modified_ms is None:
        raise ValueError("The file has no embedded timestamp and no file modification time was supplied")
    return datetime.fromtimestamp(file_modified_ms / 1000, tz=timezone.utc)


def _sample_id(task: TaskName, filename: str) -> str:
    stem = Path(filename).stem.strip() or "upload"
    prefix = {"door": "Door dataset", "corrugation": "Rail sample", "shm": "SHM sample"}[task]
    return f"{prefix} {stem}"


def _door_time(value: Any) -> datetime:
    text = str(value).strip()
    match = re.fullmatch(
        r"(\d{4})-(\d{1,2})-(\d{1,2})-(\d{1,2})-(\d{1,2})-(\d{1,2})-(\d{1,6})",
        text,
    )
    if not match:
        parsed = pd.to_datetime(text, errors="raise")
        return parsed.to_pydatetime()
    year, month, day, hour, minute, second, fraction = match.groups()
    microsecond = int(fraction.ljust(3, "0")[:3]) * 1000
    return datetime(
        int(year), int(month), int(day), int(hour), int(minute), int(second), microsecond
    )


def _door(payload: bytes, filename: str) -> MetadataSuggestion:
    frame = pd.read_csv(io.BytesIO(payload), usecols=["Datetime"], nrows=1)
    if frame.empty:
        raise ValueError("Door file contains no data rows")
    return MetadataSuggestion(
        asset_id=_sample_id("door", filename),
        component_info="Door system",
        measurement_time=_door_time(frame.iloc[0]["Datetime"]),
        asset_source="filename",
        component_source="embedded",
        measurement_time_source="embedded",
        warnings=["No train or door identifier exists in this file; the filename is used as the asset ID."],
    )


def _acv(payload: bytes) -> MetadataSuggestion:
    workbook = io.BytesIO(payload)
    frame = pd.read_excel(workbook, nrows=1)
    required = {"Train number", "Time"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"ACV workbook is missing columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("ACV workbook contains no data rows")
    train_number = frame.iloc[0]["Train number"]
    if pd.isna(train_number):
        raise ValueError("ACV workbook has no train number in its first data row")
    train_text = str(int(train_number)) if isinstance(train_number, (int, float)) and float(train_number).is_integer() else str(train_number).strip()
    car_numbers = sorted(
        {int(match.group(1)) for column in frame.columns if (match := re.match(r"Car\s+(\d+)", str(column)))}
    )
    components = "ACV"
    if car_numbers:
        components += f" · Cars {min(car_numbers):02d}–{max(car_numbers):02d}"
    if "Car model" in frame.columns and not pd.isna(frame.iloc[0]["Car model"]):
        components += f" · Model {str(frame.iloc[0]['Car model']).strip()}"
    return MetadataSuggestion(
        asset_id=f"Train {train_text}",
        component_info=components,
        measurement_time=pd.to_datetime(frame.iloc[0]["Time"], errors="raise").to_pydatetime(),
        asset_source="embedded",
        component_source="embedded",
        measurement_time_source="embedded",
        warnings=[],
    )


def _corrugation(payload: bytes, filename: str, file_modified_ms: float | None) -> MetadataSuggestion:
    columns = pd.read_csv(io.BytesIO(payload), nrows=0).columns
    locations = [
        (int(match.group(1)), int(match.group(2)))
        for column in columns
        if (match := re.search(r"position\s+(\d+)\s+of\s+car\s+(\d+)", str(column), re.IGNORECASE))
    ]
    if not locations:
        raise ValueError("Rail file has no recognised bearing-position columns")
    positions = [position for position, _ in locations]
    cars = [car for _, car in locations]
    return MetadataSuggestion(
        asset_id=_sample_id("corrugation", filename),
        component_info=f"Bearing sensors · Cars {min(cars)}–{max(cars)} · Positions {min(positions)}–{max(positions)}",
        measurement_time=_fallback_time(file_modified_ms),
        asset_source="filename",
        component_source="embedded",
        measurement_time_source="file_modified",
        warnings=[
            "No train identifier exists in this file; the filename is used as the asset ID.",
            "No measurement timestamp exists in this file; the file modification time is used.",
        ],
    )


def _shm(payload: bytes, filename: str, file_modified_ms: float | None) -> MetadataSuggestion:
    first_line = payload.splitlines()[0].decode("utf-8-sig", errors="replace") if payload else ""
    values = [value.strip() for value in first_line.split(",") if value.strip()]
    if not values:
        raise ValueError("SHM file contains no stress values")
    try:
        [float(value) for value in values]
    except ValueError as exc:
        raise ValueError("SHM file must begin with numeric stress values and no header") from exc
    channels = len(values)
    return MetadataSuggestion(
        asset_id=_sample_id("shm", filename),
        component_info=f"Structural stress · {channels} channel{'s' if channels != 1 else ''}",
        measurement_time=_fallback_time(file_modified_ms),
        asset_source="filename",
        component_source="embedded",
        measurement_time_source="file_modified",
        warnings=[
            "No train or structure identifier exists in this file; the filename is used as the asset ID.",
            "No measurement timestamp exists in this file; the file modification time is used.",
        ],
    )


def extract_metadata(
    task: TaskName,
    payload: bytes,
    filename: str,
    file_modified_ms: float | None,
) -> MetadataSuggestion:
    if task == "door":
        return _door(payload, filename)
    if task == "acv":
        return _acv(payload)
    if task == "corrugation":
        return _corrugation(payload, filename, file_modified_ms)
    return _shm(payload, filename, file_modified_ms)
