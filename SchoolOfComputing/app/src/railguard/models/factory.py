"""Task-aware model registry with sklearn fallbacks."""

from __future__ import annotations

from typing import Any, Callable

from railguard.models.classical import ClassicalClassifier, ClassicalRegressor

MODEL_REGISTRY: dict[str, dict[str, Callable[..., Any]]] = {
    "door": {name: (lambda name=name, **kw: ClassicalClassifier(name=name, **kw)) for name in ("logreg", "random_forest", "extra_trees", "hist_gradient_boosting")},
    "acv": {name: (lambda name=name, **kw: ClassicalClassifier(name=name, **kw)) for name in ("logreg", "random_forest", "extra_trees", "hist_gradient_boosting")},
    "corrugation": {name: (lambda name=name, **kw: ClassicalClassifier(name=name, **kw)) for name in ("logreg", "random_forest", "extra_trees", "hist_gradient_boosting")},
    "shm": {name: (lambda name=name, **kw: ClassicalRegressor(name=name, **kw)) for name in ("ridge", "elastic_net", "random_forest", "extra_trees", "hist_gradient_boosting")},
}


def _deep_door(**kwargs: Any) -> Any:
    from railguard.models.door.detector import DoorNet
    return DoorNet(**kwargs)


def _deep_acv(**kwargs: Any) -> Any:
    from railguard.models.acv.relational import ACVRelationalNet
    return ACVRelationalNet(**kwargs)


def _deep_corrugation(**kwargs: Any) -> Any:
    from railguard.models.corrugation.dual_domain import CorrugationNet
    return CorrugationNet(**kwargs)


def _deep_shm(**kwargs: Any) -> Any:
    from railguard.models.shm.damage_regressor import DamageNet
    return DamageNet(**kwargs)


MODEL_REGISTRY["door"]["doornet"] = _deep_door
MODEL_REGISTRY["acv"]["relational_attention"] = _deep_acv
MODEL_REGISTRY["corrugation"]["dual_domain"] = _deep_corrugation
MODEL_REGISTRY["shm"]["damagenet"] = _deep_shm


def create_model(task: str, name: str, **kwargs: Any) -> Any:
    try:
        return MODEL_REGISTRY[task][name](**kwargs)
    except KeyError as exc:
        raise KeyError(f"Unknown model {task}/{name}; choices: {sorted(MODEL_REGISTRY.get(task, {}))}") from exc
