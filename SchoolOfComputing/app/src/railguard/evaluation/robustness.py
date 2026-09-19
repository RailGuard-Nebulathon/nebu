"""Copy-safe sensor corruptions and metric-retention reporting."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def corrupt(values: np.ndarray, name: str, severity: float, seed: int = 42) -> np.ndarray:
    x = np.asarray(values, dtype=float).copy()
    rng = np.random.default_rng(seed)
    scale = np.std(x, axis=0, keepdims=True)
    if name == "gaussian_noise":
        x += rng.normal(size=x.shape) * scale * severity
    elif name == "amplitude_scale":
        x *= 1 + severity
    elif name == "constant_bias":
        x += scale * severity
    elif name == "random_point_dropout":
        x[rng.random(x.shape) < severity] = 0
    elif name == "contiguous_dropout":
        width = max(1, int(len(x) * severity)); start = int(rng.integers(0, max(len(x) - width + 1, 1))); x[start : start + width] = 0
    elif name == "downsample_upsample":
        factor = max(2, int(round(1 / max(severity, 1e-3))))
        selected = np.arange(0, len(x), factor); target = np.arange(len(x))
        x = np.column_stack([np.interp(target, selected, x[selected, channel]) for channel in range(x.shape[1])])
    elif name == "single_channel_dropout":
        x[:, int(rng.integers(0, x.shape[1]))] = 0
    else:
        raise ValueError(f"Unknown corruption: {name}")
    return x


def jitter_timestamps(timestamps: np.ndarray, severity_seconds: float, seed: int = 42) -> np.ndarray:
    output = np.asarray(timestamps, dtype=float).copy()
    output += np.random.default_rng(seed).normal(scale=severity_seconds, size=len(output))
    return np.sort(output)


def robustness_report(values: np.ndarray, evaluate: Callable[[np.ndarray], float], configurations: dict[str, list[float]]) -> dict[str, object]:
    clean = float(evaluate(values.copy()))
    rows = []
    for name, levels in configurations.items():
        for severity in levels:
            score = float(evaluate(corrupt(values, name, severity)))
            rows.append({"corruption": name, "severity": severity, "metric": score, "retention_ratio": score / clean if clean else 0.0, "absolute_degradation": clean - score})
    return {
        "clean_metric": clean, "corruptions": rows,
        "worst_corruption": max(rows, key=lambda row: row["absolute_degradation"])["corruption"] if rows else None,
        "mean_corruption_score": float(np.mean([row["metric"] for row in rows])) if rows else clean,
    }

