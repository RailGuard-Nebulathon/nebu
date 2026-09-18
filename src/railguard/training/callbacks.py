"""Minimal callback protocol."""

from typing import Protocol


class Callback(Protocol):
    def on_epoch_end(self, metrics: dict[str, float]) -> None: ...

