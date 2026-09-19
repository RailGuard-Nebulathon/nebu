from dashboard.data_access import demo_data


def test_demo_data_is_complete_and_deterministic() -> None:
    first, second = demo_data(), demo_data()
    assert set(first) == {"door", "acv", "vibration_time", "vibration", "stress_time", "stress", "predictions"}
    assert first["predictions"].equals(second["predictions"])
    assert len(first["predictions"]) == 4

