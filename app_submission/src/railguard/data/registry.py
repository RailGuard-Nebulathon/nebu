"""Lazy task adapter registry."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Callable

from railguard.data.adapters.base import BaseAdapter

ADAPTER_REGISTRY: dict[str, Callable[[str | Path], BaseAdapter]] = {}


def register_adapter(task: str, adapter: Callable[[str | Path], BaseAdapter]) -> None:
    ADAPTER_REGISTRY[task] = adapter


def get_adapter(task: str, root: str | Path) -> BaseAdapter:
    if task not in ADAPTER_REGISTRY:
        if task not in {"door", "acv", "corrugation", "shm"}:
            raise KeyError(f"Unknown task '{task}'. Registered: {sorted(ADAPTER_REGISTRY)}")
        importlib.import_module(f"railguard.data.adapters.{task}")
    try:
        return ADAPTER_REGISTRY[task](root)
    except KeyError as exc:
        raise KeyError(f"Unknown task '{task}'. Registered: {sorted(ADAPTER_REGISTRY)}") from exc
