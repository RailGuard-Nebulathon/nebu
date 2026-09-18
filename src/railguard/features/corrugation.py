"""Side-aware vibration/shock features for 10 kHz recordings."""

from __future__ import annotations

import re

import numpy as np
from scipy.stats import kurtosis, skew

from railguard.signal.spectral import spectral_entropy
from railguard.types import SequenceSample

CHANNEL = re.compile(r"^(?P<kind>Vibration|Shock) of bearing in position (?P<position>\d+) of car (?P<car>\d+)$")


def corrugation_features(sample: SequenceSample, sampling_rate: float = 10000.0) -> dict[str, float]:
    groups: dict[tuple[str, str], list[int]] = {}
    for index, name in enumerate(sample.channel_names):
        match = CHANNEL.match(name)
        if not match:
            continue
        side = "side_i" if int(match.group("position")) % 2 else "side_ii"
        groups.setdefault((side, match.group("kind").lower()), []).append(index)
    if len(groups) != 4:
        raise ValueError(f"Expected vibration/shock channels for both sides; found {sorted(groups)}")
    output: dict[str, float] = {}
    for (side, kind), indexes in sorted(groups.items()):
        values = sample.values[:, indexes].astype(float)
        prefix = f"{side}__{kind}"
        channel_rms = np.sqrt(np.mean(np.square(values), axis=0))
        output.update({
            f"{prefix}__rms_mean": float(channel_rms.mean()), f"{prefix}__rms_max": float(channel_rms.max()),
            f"{prefix}__std_mean": float(np.std(values, axis=0).mean()), f"{prefix}__peak_abs": float(np.max(np.abs(values))),
            f"{prefix}__skew_abs_mean": float(np.mean(np.abs(np.nan_to_num(skew(values, axis=0))))),
            f"{prefix}__kurtosis_mean": float(np.mean(np.nan_to_num(kurtosis(values, axis=0)))),
            f"{prefix}__crest_mean": float(np.mean(np.max(np.abs(values), axis=0) / np.maximum(channel_rms, 1e-9))),
            f"{prefix}__zero_crossing_rate": float(np.mean(np.signbit(values[1:]) != np.signbit(values[:-1]))),
        })
        centered = values - values.mean(axis=0)
        power = np.mean(np.abs(np.fft.rfft(centered, axis=0)) ** 2, axis=1)
        frequencies = np.fft.rfftfreq(len(values), 1 / sampling_rate)
        total = max(float(power.sum()), 1e-9)
        centroid = float(np.sum(frequencies * power) / total)
        output[f"{prefix}__dominant_frequency"] = float(frequencies[np.argmax(power)])
        output[f"{prefix}__spectral_centroid"] = centroid
        output[f"{prefix}__spectral_entropy"] = spectral_entropy(power)
        for band_index, (low, high) in enumerate(((0, 250), (250, 1000), (1000, 2500), (2500, 5001))):
            mask = (frequencies >= low) & (frequencies < high)
            output[f"{prefix}__band_ratio_{band_index}"] = float(power[mask].sum() / total)
    for kind in ("vibration", "shock"):
        left = output[f"side_i__{kind}__rms_mean"]
        right = output[f"side_ii__{kind}__rms_mean"]
        output[f"side_energy_ratio__{kind}"] = left / max(right, 1e-9)
    return output

