"""Headerless dynamic-stress SHM file adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from railguard.data.adapters.base import BaseAdapter
from railguard.data.registry import register_adapter
from railguard.types import SequenceSample


class SHMAdapter(BaseAdapter):
    task = "shm"

    def files(self, split: str) -> list[Path]:
        folder = self.root / ("Train" if split == "train" else "Test")
        if not folder.exists():
            raise FileNotFoundError(f"SHM {split} folder not found: {folder}")
        return sorted(folder.glob("*.csv"), key=lambda path: int("".join(filter(str.isdigit, path.stem))))

    def samples(self, split: str = "train") -> Iterable[SequenceSample]:
        labels: dict[str, float] = {}
        label_path = self.root / "Train_Labels.csv"
        if split == "train" and label_path.exists():
            label_frame = pd.read_csv(label_path)
            labels = dict(zip(label_frame["filename"], label_frame["damage"].astype(float), strict=True))
        for path in self.files(split):
            frame = pd.read_csv(path, header=None)
            numeric = frame.apply(pd.to_numeric, errors="coerce")
            if numeric.isna().any().any():
                raise ValueError(f"SHM file contains non-numeric/missing stress values: {path}")
            channels = [f"stress_{index}" for index in range(numeric.shape[1])]
            yield SequenceSample(
                sample_id=path.name, values=numeric.to_numpy(dtype=np.float32), timestamps=None,
                channel_names=channels, metadata={"source_path": str(path), "schema": "headerless_numeric_stress"}, target=labels.get(path.name),
            )


register_adapter("shm", SHMAdapter)

