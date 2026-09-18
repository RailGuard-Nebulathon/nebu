"""Within-case relational ACV candidate features."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from railguard.data.adapters.acv import split_car_columns


# The supplied ACV workbooks contain two vendor schemas.  These aliases map
# like-for-like signals without relying on a brittle intersection of headers.
ACV_SEQUENCE_SCHEMA: dict[str, tuple[str, tuple[str, ...]]] = {
    "cabin_temperature": (
        "numeric",
        ("Indoor Average Temperature", "Passenger Cabin Temperature Detected Value"),
    ),
    "outdoor_temperature": (
        "numeric",
        ("Outdoor Average Temperature", "Outside Temperature Sensor Reading", "Fresh Air Temperature Detected Value"),
    ),
    "target_temperature": (
        "numeric",
        ("ACV Control Temperature (Cooling)", "Target Temperature Value"),
    ),
    "running_mode": ("categorical", ("ACV Running Mode",)),
    "control_mode": ("categorical", ("ACV Setting Mode", "ACV Control Mode")),
    "load_shedding": ("categorical", ("Load Halved", "Load Shedding")),
    "information_valid": ("categorical", ("ACV Information Valid",)),
}


def _normalise_category(value: object, signal: str) -> str:
    text = " ".join(str(value).strip().lower().replace("-", " ").split())
    aliases = {
        ("running_mode", "stop"): "stopped",
        ("load_shedding", "normal"): "no load shedding",
    }
    return aliases.get((signal, text), text)


class ACVSequenceCanonicalizer:
    """Fit and apply a fixed, mask-aware sequence schema across ACV vendors."""

    def __init__(self) -> None:
        self.category_maps: dict[str, dict[str, int]] = {}

    @staticmethod
    def _source(frame: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
        return next((column for column in aliases if column in frame.columns), None)

    def fit(self, frames: Iterable[pd.DataFrame]) -> "ACVSequenceCanonicalizer":
        categories = {name: set() for name, (kind, _) in ACV_SEQUENCE_SCHEMA.items() if kind == "categorical"}
        for frame in frames:
            for name in categories:
                source = self._source(frame, ACV_SEQUENCE_SCHEMA[name][1])
                if source is not None:
                    values = frame[source].dropna().map(lambda value: _normalise_category(value, name))
                    categories[name].update(value for value in values if value)
        self.category_maps = {
            name: {value: index + 1 for index, value in enumerate(sorted(values))}
            for name, values in categories.items()
        }
        return self

    @property
    def channel_names(self) -> list[str]:
        names = list(ACV_SEQUENCE_SCHEMA)
        return names + [f"{name}__available" for name in names]

    def transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, str | None]]:
        if not self.category_maps:
            raise RuntimeError("ACVSequenceCanonicalizer must be fit before transform")
        values: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        sources: dict[str, str | None] = {}
        for name, (kind, aliases) in ACV_SEQUENCE_SCHEMA.items():
            source = self._source(frame, aliases)
            sources[name] = source
            available = source is not None
            if kind == "numeric" and source is not None:
                series = pd.to_numeric(frame[source], errors="coerce").replace([np.inf, -np.inf], np.nan)
                available = bool(series.notna().any())
                series = series.interpolate(limit_direction="both").fillna(0.0)
            elif kind == "categorical" and source is not None:
                mapping = self.category_maps[name]
                series = frame[source].map(
                    lambda value: mapping.get(_normalise_category(value, name), 0) if pd.notna(value) else 0
                )
                available = bool(frame[source].notna().any())
            else:
                series = pd.Series(np.zeros(len(frame), dtype=float), index=frame.index)
            values.append(series.to_numpy(dtype=float))
            masks.append(np.full(len(frame), float(available)))
        return np.column_stack(values + masks), sources


def _slope(values: np.ndarray) -> float:
    valid = np.isfinite(values)
    return float(np.polyfit(np.flatnonzero(valid), values[valid], 1)[0]) if valid.sum() > 2 else 0.0


def _car_summary(frame: pd.DataFrame) -> dict[str, float]:
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    temperature = numeric[[c for c in numeric if "temperature" in c.lower()]]
    control = numeric[[c for c in numeric if any(word in c.lower() for word in ("mode", "status", "control", "relay", "command"))]]
    values = numeric.to_numpy(dtype=float)
    temp_values = temperature.to_numpy(dtype=float) if not temperature.empty else values
    slopes = [_slope(numeric[c].to_numpy(dtype=float)) for c in numeric]
    return {
        "absolute_mean": float(np.nanmean(values)), "absolute_std": float(np.nanstd(values)),
        "absolute_min": float(np.nanmin(values)), "absolute_max": float(np.nanmax(values)),
        "temperature_mean": float(np.nanmean(temp_values)), "temperature_std": float(np.nanstd(temp_values)),
        "temperature_range": float(np.nanmax(temp_values) - np.nanmin(temp_values)),
        "mean_slope": float(np.mean(slopes)), "max_abs_slope": float(np.max(np.abs(slopes))),
        "control_change_rate": float(np.nanmean(np.diff(control.to_numpy(dtype=float), axis=0) != 0)) if not control.empty and len(control) > 1 else 0.0,
        "missing_fraction": float(numeric.isna().mean().mean()), "parameter_count": float(numeric.shape[1]),
    }


def acv_candidate_features(frame: pd.DataFrame, case_id: str, faulty_car: str | None = None) -> pd.DataFrame:
    _, cars = split_car_columns(frame)
    absolute = pd.DataFrame.from_dict({car: _car_summary(values) for car, values in cars.items()}, orient="index").sort_index()
    peer_median = absolute.median(axis=0)
    relative = absolute - peer_median
    relative.columns = [f"peer_residual__{column}" for column in relative.columns]
    output = pd.concat([absolute.add_prefix("absolute__"), relative], axis=1)
    output.insert(0, "car_id", output.index)
    output.insert(0, "case_id", case_id)
    if faulty_car is not None:
        output["is_faulty"] = output["car_id"] == str(faulty_car).zfill(2)
    return output.reset_index(drop=True)


def robust_rule_score(candidates: pd.DataFrame) -> np.ndarray:
    columns = [c for c in candidates if c.startswith("peer_residual__") and c != "peer_residual__parameter_count"]
    matrix = candidates[columns].to_numpy(dtype=float)
    median = np.nanmedian(matrix, axis=0)
    mad = np.nanmedian(np.abs(matrix - median), axis=0)
    z = np.abs((matrix - median) / np.maximum(mad * 1.4826, 1e-9))
    return np.nanmean(z, axis=1)
