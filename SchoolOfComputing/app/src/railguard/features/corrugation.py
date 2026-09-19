"""Side-aware vibration/shock features for 10 kHz recordings."""

from __future__ import annotations

import re

import numpy as np
from scipy.stats import kurtosis, skew

from railguard.signal.spectral import spectral_entropy
from railguard.types import SequenceSample

CHANNEL = re.compile(
    r"^(?P<kind>Vibration|Shock) of bearing in position (?P<position>\d+) of car (?P<car>\d+)$"
)


def corrugation_features(
    sample: SequenceSample, sampling_rate: float = 10000.0
) -> dict[str, float]:
    sampling_rate = float(sample.metadata.get("sampling_rate_hz", sampling_rate))
    speed_signal = np.asarray(sample.metadata.get("rotating_speed", []), dtype=float).reshape(-1)
    transitions = (
        float(np.count_nonzero(np.diff(speed_signal) != 0)) if len(speed_signal) > 1 else 0.0
    )
    duration = len(sample.values) / sampling_rate
    # The toothed wheel has 90 teeth and the detector toggles twice per tooth.
    revolutions_per_second = transitions / max(180.0 * duration, 1e-9)
    speed_mps = revolutions_per_second * np.pi * 0.85
    groups: dict[tuple[str, str], list[int]] = {}
    for index, name in enumerate(sample.channel_names):
        match = CHANNEL.match(name)
        if not match:
            continue
        side = "side_i" if int(match.group("position")) % 2 else "side_ii"
        groups.setdefault((side, match.group("kind").lower()), []).append(index)
    if len(groups) != 4:
        raise ValueError(
            f"Expected vibration/shock channels for both sides; found {sorted(groups)}"
        )
    output: dict[str, float] = {
        "speed__transition_count": transitions,
        "speed__revolutions_per_second": revolutions_per_second,
        "speed__metres_per_second": speed_mps,
    }
    for (side, kind), indexes in sorted(groups.items()):
        values = sample.values[:, indexes].astype(float)
        prefix = f"{side}__{kind}"
        channel_rms = np.sqrt(np.mean(np.square(values), axis=0))
        output.update(
            {
                f"{prefix}__rms_mean": float(channel_rms.mean()),
                f"{prefix}__rms_max": float(channel_rms.max()),
                f"{prefix}__std_mean": float(np.std(values, axis=0).mean()),
                f"{prefix}__peak_abs": float(np.max(np.abs(values))),
                f"{prefix}__skew_abs_mean": float(
                    np.mean(np.abs(np.nan_to_num(skew(values, axis=0))))
                ),
                f"{prefix}__kurtosis_mean": float(np.mean(np.nan_to_num(kurtosis(values, axis=0)))),
                f"{prefix}__crest_mean": float(
                    np.mean(np.max(np.abs(values), axis=0) / np.maximum(channel_rms, 1e-9))
                ),
                f"{prefix}__zero_crossing_rate": float(
                    np.mean(np.signbit(values[1:]) != np.signbit(values[:-1]))
                ),
            }
        )
        centered = values - values.mean(axis=0)
        power = np.mean(np.abs(np.fft.rfft(centered, axis=0)) ** 2, axis=1)
        frequencies = np.fft.rfftfreq(len(values), 1 / sampling_rate)
        total = max(float(power.sum()), 1e-9)
        centroid = float(np.sum(frequencies * power) / total)
        output[f"{prefix}__dominant_frequency"] = float(frequencies[np.argmax(power)])
        output[f"{prefix}__spectral_centroid"] = centroid
        output[f"{prefix}__dominant_spatial_frequency"] = float(
            frequencies[np.argmax(power)] / max(speed_mps, 1e-6)
        )
        output[f"{prefix}__spatial_centroid"] = float(centroid / max(speed_mps, 1e-6))
        output[f"{prefix}__spectral_entropy"] = spectral_entropy(power)
        for band_index, (low, high) in enumerate(
            ((0, 250), (250, 1000), (1000, 2500), (2500, 5001))
        ):
            mask = (frequencies >= low) & (frequencies < high)
            output[f"{prefix}__band_ratio_{band_index}"] = float(power[mask].sum() / total)
    for kind in ("vibration", "shock"):
        left = output[f"side_i__{kind}__rms_mean"]
        right = output[f"side_ii__{kind}__rms_mean"]
        output[f"side_energy_ratio__{kind}"] = left / max(right, 1e-9)
        output[f"side_log_energy_ratio__{kind}"] = float(np.log(max(left, 1e-9) / max(right, 1e-9)))
        output[f"side_dominant_frequency_delta__{kind}"] = (
            output[f"side_i__{kind}__dominant_frequency"]
            - output[f"side_ii__{kind}__dominant_frequency"]
        )
        for band_index in range(4):
            output[f"side_band_delta__{kind}_{band_index}"] = (
                output[f"side_i__{kind}__band_ratio_{band_index}"]
                - output[f"side_ii__{kind}__band_ratio_{band_index}"]
            )
    return output
