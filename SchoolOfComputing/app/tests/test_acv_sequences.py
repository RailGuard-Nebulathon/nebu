import numpy as np
import pandas as pd

from railguard.features.acv import ACVSequenceCanonicalizer


def test_acv_sequence_canonicalizer_aligns_vendor_schemas_and_adds_masks() -> None:
    legacy = pd.DataFrame(
        {
            "Indoor Average Temperature": [24.0, 25.0],
            "Outdoor Average Temperature": [30.0, 31.0],
            "ACV Control Temperature (Cooling)": [23.0, 23.0],
            "ACV Running Mode": ["Stop", "Full Cooling"],
            "ACV Setting Mode": ["Centralized Control", "Centralized Control"],
            "Load Halved": ["Normal", "Normal"],
            "ACV Information Valid": ["Valid", "Valid"],
        }
    )
    alternate = pd.DataFrame(
        {
            "Passenger Cabin Temperature Detected Value": [24.0, 25.0],
            "Fresh Air Temperature Detected Value": [30.0, 31.0],
            "Target Temperature Value": [23.0, 23.0],
            "ACV Running Mode": ["Stopped", "Full Cooling"],
            "ACV Control Mode": ["Centralized Control", "Centralized Control"],
            "Load Shedding": ["No Load Shedding", "No Load Shedding"],
        }
    )

    canonicalizer = ACVSequenceCanonicalizer().fit([legacy, alternate])
    old_values, old_sources = canonicalizer.transform(legacy)
    new_values, new_sources = canonicalizer.transform(alternate)

    assert old_values.shape == new_values.shape == (2, 14)
    np.testing.assert_allclose(old_values[:, :6], new_values[:, :6])
    assert old_values[:, 13].tolist() == [1.0, 1.0]
    assert new_values[:, 13].tolist() == [0.0, 0.0]
    assert old_sources["outdoor_temperature"] == "Outdoor Average Temperature"
    assert new_sources["outdoor_temperature"] == "Fresh Air Temperature Detected Value"


def test_acv_sequence_canonicalizer_interpolates_numeric_gaps() -> None:
    frame = pd.DataFrame(
        {
            "Indoor Average Temperature": [20.0, np.nan, 24.0],
            "ACV Running Mode": ["Stop", "Stop", "Stop"],
        }
    )
    values, _ = ACVSequenceCanonicalizer().fit([frame]).transform(frame)
    assert values[:, 0].tolist() == [20.0, 22.0, 24.0]
    assert np.isfinite(values).all()


def test_acv_sequence_canonicalizer_marks_empty_template_car_unavailable() -> None:
    frame = pd.DataFrame({"ACV Running Mode": [np.nan, np.nan]})
    values, _ = ACVSequenceCanonicalizer().fit([frame]).transform(frame)
    assert not values[:, 7:].any()
