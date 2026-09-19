import numpy as np

from railguard.features.base import CombinedFeatureExtractor
from railguard.signal.cleaning import clean_signal
from railguard.signal.stft import stft_magnitude
from railguard.types import SequenceSample


def test_feature_shape_order_and_finiteness() -> None:
    time = np.linspace(0, 1, 128, endpoint=False)
    sample = SequenceSample("a", np.column_stack([np.sin(2 * np.pi * 5 * time), np.cos(2 * np.pi * 9 * time)]), time, ["x", "y"])
    extractor = CombinedFeatureExtractor(sampling_rate=128)
    first = extractor.fit_transform([sample])
    second = extractor.transform([sample])
    assert list(first.columns) == sorted(first.columns)
    assert list(first.columns) == list(second.columns)
    assert np.isfinite(first.to_numpy()).all()
    assert stft_magnitude(sample.values[:, 0], 128)[2].ndim == 2


def test_cleaning_is_explicit() -> None:
    cleaned = clean_signal(np.array([1.0, np.nan, 3.0]), max_missing_fraction=0.5)
    np.testing.assert_allclose(cleaned, [1.0, 2.0, 3.0])

