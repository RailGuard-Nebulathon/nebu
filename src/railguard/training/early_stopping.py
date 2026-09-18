"""Stateful early stopping."""

from dataclasses import dataclass


@dataclass
class EarlyStopping:
    patience: int = 12
    mode: str = "min"
    best: float | None = None
    stale_epochs: int = 0

    def update(self, value: float) -> bool:
        improved = self.best is None or (value < self.best if self.mode == "min" else value > self.best)
        if improved:
            self.best, self.stale_epochs = value, 0
        else:
            self.stale_epochs += 1
        return self.stale_epochs >= self.patience

