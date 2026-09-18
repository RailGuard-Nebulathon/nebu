import numpy as np
import pytest

from railguard.data.timestamp import elapsed_seconds, parse_door_timestamp, parse_door_timestamps


def test_door_timestamp_parser() -> None:
    assert parse_door_timestamp("2023-7-5-0-0-0-20").microsecond == 20_000
    parsed, errors = parse_door_timestamps(["2023-7-5-0-0-0-0", "2023-7-5-0-0-0-20"])
    assert not errors
    np.testing.assert_allclose(elapsed_seconds(parsed), [0.0, 0.02])


def test_door_timestamp_rejects_malformed() -> None:
    with pytest.raises(ValueError, match="Malformed"):
        parse_door_timestamp("2023/7/5")

