from __future__ import annotations

import io

import pytest

pytest.importorskip("multipart")
TestClient = pytest.importorskip("fastapi.testclient").TestClient
app = pytest.importorskip("railguard_api.main").app
client = TestClient(app)


def test_health_and_task_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAILGUARD_ALLOW_DEMO", "true")
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    tasks = client.get("/api/tasks").json()
    assert [task["id"] for task in tasks] == ["door", "acv", "corrugation", "shm"]


@pytest.mark.parametrize(
    ("task", "filename", "column"),
    [
        ("door", "Test.csv", "start_time"),
        ("acv", "case.xlsx", "ranked_cars"),
        ("corrugation", "rail.csv", "prediction"),
        ("shm", "stress.txt", "prediction"),
    ],
)
def test_demo_prediction_has_official_schema(task: str, filename: str, column: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAILGUARD_ALLOW_DEMO", "true")
    response = client.post(
        f"/api/predict/{task}?mode=demo",
        files={"file": (filename, io.BytesIO(b"synthetic-input"), "application/octet-stream")},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["mode"] == "demo"
    assert column in body["rows"][0]
    assert "Demonstration" in body["notices"][0]


def test_rejects_wrong_extension(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAILGUARD_ALLOW_DEMO", "true")
    response = client.post(
        "/api/predict/acv?mode=demo",
        files={"file": ("case.csv", b"wrong", "text/csv")},
    )
    assert response.status_code == 415
