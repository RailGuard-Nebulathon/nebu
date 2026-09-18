"""Operation-aware Door features and physics proxies."""

from __future__ import annotations

import numpy as np
import pandas as pd

from railguard.features.base import CombinedFeatureExtractor
from railguard.types import SequenceSample


class DoorFeatureExtractor(CombinedFeatureExtractor):
    def __init__(self, sampling_rate: float = 50.0) -> None:
        super().__init__(sampling_rate)

    def _one(self, sample: SequenceSample) -> dict[str, float]:
        output = super()._one(sample)
        channels = {name: sample.values[:, i].astype(float) for i, name in enumerate(sample.channel_names)}
        current = channels["Motor current(mA)"]
        voltage = channels["Motor Voltage(10mV)"]
        position = channels["Door leaf position"]
        velocity = np.gradient(position)
        acceleration = np.gradient(velocity)
        position_delta = np.abs(velocity)
        power_proxy = current * voltage
        correlation = np.correlate(current - current.mean(), velocity - velocity.mean(), mode="full")
        lag = int(np.argmax(np.abs(correlation)) - (len(current) - 1))
        operation = str(sample.metadata.get("operation", "Unknown"))
        output.update({
            "door__operation_open": float(operation == "Open"), "door__operation_close": float(operation == "Close"),
            "door__total_travel": float(np.abs(position[-1] - position[0])),
            "door__monotonicity_violations": float(np.sum(np.sign(velocity[1:]) != np.sign(np.median(velocity)))),
            "door__average_velocity": float(np.mean(np.abs(velocity))), "door__max_velocity": float(np.max(np.abs(velocity))),
            "door__average_acceleration": float(np.mean(np.abs(acceleration))), "door__max_acceleration": float(np.max(np.abs(acceleration))),
            "door__stall_fraction": float(np.mean(position_delta < max(np.quantile(position_delta, 0.1), 1e-9))),
            "door__power_proxy_mean": float(np.mean(power_proxy)), "door__energy_proxy": float(np.trapz(power_proxy)),
            "door__current_per_position_change": float(np.sum(np.abs(current)) / max(np.sum(position_delta), 1e-9)),
            "door__current_position_velocity_correlation": float(np.corrcoef(current, velocity)[0, 1]) if np.std(velocity) > 0 and np.std(current) > 0 else 0.0,
            "door__current_velocity_max_correlation_lag": float(lag / max(len(current) - 1, 1)),
        })
        return {key: float(np.nan_to_num(value)) for key, value in output.items()}


def door_feature_table(samples: list[SequenceSample], sampling_rate: float = 50.0) -> pd.DataFrame:
    return DoorFeatureExtractor(sampling_rate).fit_transform(samples)
