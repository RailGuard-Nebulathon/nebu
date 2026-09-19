import numpy as np
import pytest

from railguard.evaluation.robustness import corrupt, jitter_timestamps, robustness_report


@pytest.mark.parametrize("name", ["gaussian_noise", "amplitude_scale", "constant_bias", "random_point_dropout", "contiguous_dropout", "downsample_upsample", "single_channel_dropout"])
def test_corruptions_copy_and_shape(name: str) -> None:
    original = np.ones((50, 3))
    changed = corrupt(original, name, 0.1)
    assert changed.shape == original.shape
    assert np.all(original == 1)


def test_timestamp_jitter_and_report() -> None:
    timestamps = np.arange(10, dtype=float)
    assert np.all(np.diff(jitter_timestamps(timestamps, 0.01)) >= 0)
    report = robustness_report(np.ones((20, 2)), lambda x: float(np.mean(x)), {"amplitude_scale": [0.1]})
    assert "retention_ratio" in report["corruptions"][0]

