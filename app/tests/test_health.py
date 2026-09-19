from railguard.health.maintenance import maintenance_status
from railguard.health.scoring import health_score


def test_health_score_and_status() -> None:
    healthy = health_score(fault_risk=0.0, damage_risk=0.0)
    concern = health_score(fault_risk=1.0, damage_risk=1.0, anomaly_risk=1.0)
    assert healthy == 100 and concern == 0
    assert maintenance_status(healthy) == "Healthy"
    assert maintenance_status(concern) == "High Priority"
    assert health_score(fault_risk=0, confidence=0, ood=True) == 50

