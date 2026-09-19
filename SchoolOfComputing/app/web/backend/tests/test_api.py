from __future__ import annotations

import io

import pytest

pytest.importorskip("multipart")
TestClient = pytest.importorskip("fastapi.testclient").TestClient
api_module = pytest.importorskip("railguard_api.main")
app = api_module.app
client = TestClient(app)


def test_cors_origins_are_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "RAILGUARD_CORS_ORIGINS",
        " https://railguard.example.com/, https://operators.example.com ",
    )

    assert api_module._cors_origins() == [
        "https://railguard.example.com",
        "https://operators.example.com",
    ]


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


def test_door_real_format_reports_conservative_confidence_without_changing_official_rows(
) -> None:
    predictions = [
        {
            "prediction": "Normal",
            "confidence": 0.94,
            "metadata": {"start_time": "start-1", "end_time": "end-1"},
        },
        {
            "prediction": "Abnormal resistance",
            "confidence": 0.81,
            "metadata": {"start_time": "start-2", "end_time": "end-2"},
        },
    ]

    rows, summary, visual = api_module._format_real("door", "door.csv", predictions)

    assert rows == [
        {"start_time": "start-1", "end_time": "end-1", "prediction": "Normal"},
        {"start_time": "start-2", "end_time": "end-2", "prediction": "Abnormal resistance"},
    ]
    assert summary == {"cycles": 2, "abnormal_cycles": 1, "confidence": 0.81}
    assert [segment["confidence"] for segment in visual["segments"]] == [0.94, 0.81]
