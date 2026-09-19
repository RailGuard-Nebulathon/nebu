"""Configurable demonstration maintenance statuses."""


def maintenance_status(score: float, thresholds: tuple[float, float, float] = (80, 60, 35)) -> str:
    healthy, monitor, inspect = thresholds
    if not healthy > monitor > inspect:
        raise ValueError("Health thresholds must be strictly descending")
    if score >= healthy:
        return "Healthy"
    if score >= monitor:
        return "Monitor"
    if score >= inspect:
        return "Inspect Soon"
    return "High Priority"

