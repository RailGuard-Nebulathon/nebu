"""Explicit leakage checks."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def assert_disjoint(values_a: Iterable[str], values_b: Iterable[str], name: str) -> None:
    overlap = set(values_a) & set(values_b)
    if overlap:
        raise ValueError(f"{name} overlap detected: {sorted(overlap)[:5]}")


def validate_split_leakage(train: pd.DataFrame, validation: pd.DataFrame, grouped: bool = False) -> None:
    assert_disjoint(train["file_hash"], validation["file_hash"], "source hash")
    if grouped:
        assert_disjoint(train["group_id"].dropna(), validation["group_id"].dropna(), "group")


def reject_target_columns(frame: pd.DataFrame, target_names: Iterable[str]) -> None:
    leaked = set(frame.columns) & set(target_names)
    if leaked:
        raise ValueError(f"Target leakage columns in feature table: {sorted(leaked)}")

