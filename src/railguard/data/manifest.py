"""Dataset manifests with source and schema identity."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from railguard.data.schema_inspector import inspect_file
from railguard.utils.hashing import file_sha256, stable_hash


@dataclass(slots=True)
class ManifestRecord:
    sample_id: str
    task: str
    source_path: str
    split: str
    label: str | float | None
    group_id: str | None
    n_rows: int | None
    n_channels: int | None
    schema_hash: str
    file_hash: str
    metadata_json: str = "{}"


def build_manifest(
    files: Iterable[str | Path], task: str, split: str, labels: dict[str, Any] | None = None
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for raw_path in files:
        path = Path(raw_path)
        report = inspect_file(path, sample_rows=128)
        primary = report["sheets"][0] if "sheets" in report else report
        record = ManifestRecord(
            sample_id=path.name,
            task=task,
            source_path=str(path),
            split=split,
            label=(labels or {}).get(path.name),
            group_id=path.stem,
            n_rows=primary.get("n_rows"),
            n_channels=primary.get("n_columns"),
            schema_hash=primary.get("schema_hash", stable_hash(primary.get("columns", []))),
            file_hash=file_sha256(path),
            metadata_json=json.dumps({"extension": path.suffix.lower()}),
        )
        records.append(asdict(record))
    return pd.DataFrame(records, columns=list(ManifestRecord.__annotations__))


def manifest_hash(frame: pd.DataFrame) -> str:
    return stable_hash(frame.sort_values("sample_id").to_dict(orient="records"))

