"""Dashboard mode state."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class DashboardState:
    mode: str = "DEMO"
    predictions_root: Path = Path("outputs/predictions")

