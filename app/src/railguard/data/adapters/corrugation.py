"""Rail-corrugation per-file adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from railguard.data.adapters.base import BaseAdapter
from railguard.data.registry import register_adapter
from railguard.types import SequenceSample

SPEED_COLUMN = "Rotating speed"


class CorrugationAdapter(BaseAdapter):
    task = "corrugation"

    def files(self, split: str) -> list[Path]:
        folder = self.root / ("Train" if split == "train" else "Test")
        if not folder.exists():
            raise FileNotFoundError(f"Corrugation {split} folder not found: {folder}")
        return sorted(folder.glob("*.csv"), key=lambda path: int("".join(filter(str.isdigit, path.stem))))

    def samples(self, split: str = "train") -> Iterable[SequenceSample]:
        labels: dict[str, str] = {}
        label_path = self.root / "Train_Labels.csv"
        if split == "train" and label_path.exists():
            label_frame = pd.read_csv(label_path)
            labels = dict(zip(label_frame["filename"], label_frame["label"], strict=True))
        for path in self.files(split):
            frame = pd.read_csv(path)
            if SPEED_COLUMN not in frame or frame.shape[1] != 129:
                raise ValueError(f"Unexpected corrugation schema in {path}: {frame.shape}")
            channels = [str(c) for c in frame.columns if c != SPEED_COLUMN]
            yield SequenceSample(
                sample_id=path.name, values=frame[channels].to_numpy(dtype=np.float32), timestamps=None,
                channel_names=channels, metadata={"source_path": str(path), "sampling_rate_hz": 10000, "rotating_speed": frame[SPEED_COLUMN].to_numpy(dtype=np.float32)},
                target=labels.get(path.name),
            )


register_adapter("corrugation", CorrugationAdapter)

