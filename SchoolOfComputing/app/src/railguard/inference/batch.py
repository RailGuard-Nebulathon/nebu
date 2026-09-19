"""Batch prediction over independent files."""

from pathlib import Path


def input_files(path: str | Path) -> list[Path]:
    source = Path(path)
    if source.is_file():
        return [source]
    return sorted(p for p in source.iterdir() if p.suffix.lower() in {".csv", ".xlsx", ".txt"})

