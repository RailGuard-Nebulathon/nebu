"""Ablation table persistence."""

from pathlib import Path
import pandas as pd


def write_ablation(rows: list[dict[str, object]], path: str | Path) -> Path:
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True); pd.DataFrame(rows).to_csv(target, index=False); return target

