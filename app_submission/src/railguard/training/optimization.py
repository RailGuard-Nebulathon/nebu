"""Optional bounded Optuna entry point; never runs a study automatically."""


def require_optuna():
    try:
        import optuna
    except ImportError as exc:
        raise ImportError("Install railguard-ai[all] to run explicit Optuna studies") from exc
    return optuna

