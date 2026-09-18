"""Adapter abstraction that never mutates source data."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

from railguard.types import SequenceSample


class BaseAdapter(ABC):
    task: str

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def require_root(self) -> None:
        if not self.root.exists():
            raise FileNotFoundError(f"{self.task} data root not found: {self.root}")

    @abstractmethod
    def samples(self, split: str = "train") -> Iterable[SequenceSample]:
        """Yield canonical samples for a split."""

