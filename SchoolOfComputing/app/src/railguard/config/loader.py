"""YAML configuration merging: defaults < YAML files < dotted overrides."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from railguard.config.schema import AppConfig


def _merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(left)
    for key, value in right.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    return data


def load_config(
    *paths: str | Path, overrides: dict[str, Any] | None = None, base: str | Path = "configs/base.yaml"
) -> AppConfig:
    data = _read_yaml(Path(base))
    for path in paths:
        data = _merge(data, _read_yaml(Path(path)))
    for dotted, value in (overrides or {}).items():
        cursor = data
        parts = dotted.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return AppConfig.model_validate(data)

