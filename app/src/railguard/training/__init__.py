try:
    from railguard.training.trainer import Trainer, TrainerConfig
except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional deep extra
    if exc.name != "torch":
        raise
    Trainer = TrainerConfig = None  # type: ignore[assignment,misc]

__all__ = ["Trainer", "TrainerConfig"]
