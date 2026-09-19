"""Stress statistics, turning points, and generic fatigue-cycle proxies."""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from railguard.features.statistical import statistical_features
from railguard.types import SequenceSample


def turning_points(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float).reshape(-1)
    if len(x) < 3:
        return x.copy()
    differences = np.diff(x)
    nonzero = np.flatnonzero(differences != 0)
    if not len(nonzero):
        return x[:1]
    compact = np.r_[x[0], x[nonzero + 1]]
    signs = np.sign(np.diff(compact))
    changes = np.flatnonzero(signs[1:] != signs[:-1]) + 1
    return compact[np.r_[0, changes, len(compact) - 1]]


def rainflow_ranges(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return cycle ranges and 0.5/1.0 counts using a stack rainflow algorithm."""
    points = turning_points(values)
    stack: list[float] = []
    ranges: list[float] = []
    counts: list[float] = []
    for point in points:
        stack.append(float(point))
        while len(stack) >= 3:
            previous = abs(stack[-2] - stack[-3])
            latest = abs(stack[-1] - stack[-2])
            if latest < previous:
                break
            if len(stack) == 3:
                ranges.append(previous)
                counts.append(0.5)
                stack.pop(0)
            else:
                ranges.append(previous)
                counts.append(1.0)
                last = stack.pop()
                stack.pop()
                stack.pop()
                stack.append(last)
    for first, second in zip(stack[:-1], stack[1:], strict=False):
        ranges.append(abs(second - first))
        counts.append(0.5)
    return np.asarray(ranges), np.asarray(counts)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    if not len(values) or float(weights.sum()) <= 0:
        return 0.0
    order = np.argsort(values)
    ordered_values, ordered_weights = values[order], weights[order]
    threshold = quantile * float(ordered_weights.sum())
    index = min(
        int(np.searchsorted(np.cumsum(ordered_weights), threshold, side="left")), len(values) - 1
    )
    return float(ordered_values[index])


def shm_features(sample: SequenceSample) -> dict[str, float]:
    output: dict[str, float] = {}
    for index, channel in enumerate(sample.channel_names):
        x = sample.values[:, index].astype(float)
        output.update(
            {f"{channel}__{key}": value for key, value in statistical_features(x).items()}
        )
        derivative = np.diff(x)
        ranges, counts = rainflow_ranges(x)
        weighted = ranges * counts if len(ranges) else np.array([0.0])
        positive, _ = find_peaks(x)
        negative, _ = find_peaks(-x)
        output.update(
            {
                f"{channel}__derivative_std": float(np.std(derivative)),
                f"{channel}__derivative_max_abs": float(np.max(np.abs(derivative))),
                f"{channel}__positive_excursions": float(len(positive)),
                f"{channel}__negative_excursions": float(len(negative)),
                f"{channel}__rainflow_cycle_count": float(counts.sum()),
                f"{channel}__cycle_range_mean": float(np.average(ranges, weights=counts))
                if counts.sum()
                else 0.0,
                f"{channel}__cycle_range_max": float(ranges.max()) if len(ranges) else 0.0,
                f"{channel}__fatigue_proxy_range2": float(np.sum(counts * ranges**2)),
                f"{channel}__fatigue_proxy_range3": float(np.sum(counts * ranges**3)),
                f"{channel}__fatigue_proxy_weighted_mean": float(np.mean(weighted)),
            }
        )
        if len(ranges):
            scale = max(float(np.max(ranges)), 1e-9)
            histogram, _ = np.histogram(ranges, bins=8, range=(0, scale), weights=counts)
            histogram_total = max(float(histogram.sum()), 1e-9)
            output.update(
                {
                    f"{channel}__cycle_hist_{i}": float(value / histogram_total)
                    for i, value in enumerate(histogram)
                }
            )
            for quantile in (0.5, 0.75, 0.9, 0.95, 0.99):
                output[f"{channel}__cycle_range_q{int(quantile * 100):02d}"] = _weighted_quantile(
                    ranges, counts, quantile
                )
            # The reference target is produced from rainflow counting and an undisclosed S-N
            # curve.  A small bank of Basquin-style exponents lets the downstream regressor
            # learn that curve without fabricating material constants.
            for exponent in range(2, 9):
                proxy = float(np.sum(counts * np.power(ranges / 2.0, exponent)))
                output[f"{channel}__fatigue_log_proxy_m{exponent}"] = float(
                    np.log1p(max(proxy, 0.0))
                )
    return output


def miner_damage_proxy(
    ranges: np.ndarray, counts: np.ndarray, sn_m: float | None = None, sn_c: float | None = None
) -> float:
    if sn_m is None or sn_c is None:
        raise ValueError("Miner damage requires configured material S-N constants m and C")
    amplitudes = np.asarray(ranges, dtype=float) / 2
    return float(np.sum(np.asarray(counts) * amplitudes**sn_m / sn_c))
