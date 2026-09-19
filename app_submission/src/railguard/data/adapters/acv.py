"""ACV workbook discovery and variable-schema loading."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from railguard.data.adapters.base import BaseAdapter
from railguard.data.registry import register_adapter
from railguard.types import SequenceSample
from railguard.utils.io import write_yaml

CAR_COLUMN = re.compile(r"^Car (?P<car>\d{2}) - (?P<parameter>.+)$")


def discover_acv_mapping(path: str | Path) -> dict[str, object]:
    source = Path(path)
    excel = pd.ExcelFile(source)
    sheets: dict[str, object] = {}
    for sheet in excel.sheet_names:
        columns = list(pd.read_excel(source, sheet_name=sheet, nrows=0).columns.astype(str))
        cars: dict[str, list[str]] = {}
        identifiers = []
        for column in columns:
            match = CAR_COLUMN.match(column)
            if match:
                cars.setdefault(match.group("car"), []).append(match.group("parameter"))
            else:
                identifiers.append(column)
        sheets[sheet] = {"identifier_columns": identifiers, "cars": cars, "column_count": len(columns)}
    return {"source": source.name, "layout": "prefixed_columns", "sheets": sheets}


def write_acv_schema_mapping(files: list[str | Path], output: str | Path) -> Path:
    return write_yaml(output, {Path(path).name: discover_acv_mapping(path) for path in files})


def split_car_columns(frame: pd.DataFrame) -> tuple[list[str], dict[str, pd.DataFrame]]:
    identifiers, selected = [], {}
    for column in frame.columns.astype(str):
        match = CAR_COLUMN.match(column)
        if match:
            selected.setdefault(match.group("car"), {})[match.group("parameter")] = frame[column]
        else:
            identifiers.append(column)
    if not selected:
        raise ValueError("No ACV per-car columns matched 'Car NN - parameter'; configure another layout")
    return identifiers, {car: pd.DataFrame(parameters) for car, parameters in sorted(selected.items())}


class ACVAdapter(BaseAdapter):
    task = "acv"

    def files(self, split: str) -> list[Path]:
        folder = self.root / ("Train" if split == "train" else "Test")
        if not folder.exists():
            raise FileNotFoundError(f"ACV {split} folder not found: {folder}")
        return sorted(folder.glob("*.xlsx"))

    def samples(self, split: str = "train") -> Iterable[SequenceSample]:
        labels: dict[str, str] = {}
        label_path = self.root / "Train_Labels.csv"
        if split == "train" and label_path.exists():
            label_frame = pd.read_csv(label_path, dtype={"faulty_car": str})
            labels = dict(zip(label_frame["filename"], label_frame["faulty_car"].str.zfill(2), strict=True))
        for path in self.files(split):
            frame = pd.read_excel(path)
            identifiers, cars = split_car_columns(frame)
            numeric = frame.select_dtypes(include=np.number)
            yield SequenceSample(
                sample_id=path.name, values=numeric.to_numpy(dtype=np.float32), timestamps=None,
                channel_names=list(numeric.columns.astype(str)),
                metadata={"source_path": str(path), "identifiers": identifiers, "cars": list(cars), "mapping": discover_acv_mapping(path)},
                target=labels.get(path.name),
            )


register_adapter("acv", ACVAdapter)

