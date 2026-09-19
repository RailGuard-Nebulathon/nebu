from railguard.config import load_config


def test_config_precedence() -> None:
    config = load_config("configs/door/baseline.yaml", overrides={"training.epochs": 2})
    assert config.task == "door"
    assert config.training.epochs == 2
    assert config.project.seed == 42

