"""FastAPI wrapper around trusted RailGuard model bundles."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import tempfile
from dataclasses import asdict
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field, ValidationInfo, field_validator

from railguard.constants import CORRUGATION_OFFICIAL, DOOR_OFFICIAL, TASKS
from railguard.inference import RailGuardPredictor
from railguard.submission.schemas import OFFICIAL_COLUMNS, OFFICIAL_FILES
from railguard.submission.validator import validate_prediction_frame

from railguard_api.history import create_analysis, get_analysis, list_analyses, update_metadata
from railguard_api.metadata import MetadataSuggestion, extract_metadata

TaskName = Literal["door", "acv", "corrugation", "shm"]
RunMode = Literal["auto", "real", "demo"]
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
ALLOWED_SUFFIXES: dict[str, set[str]] = {
    "door": {".csv"},
    "acv": {".xlsx"},
    "corrugation": {".csv"},
    "shm": {".csv", ".txt"},
}


class TaskDescriptor(BaseModel):
    id: TaskName
    name: str
    short_name: str
    description: str
    accepted_extensions: list[str]
    output_filename: str
    bundle_available: bool


class PredictionResponse(BaseModel):
    task: TaskName
    task_name: str
    mode: Literal["real", "demo"]
    source_file: str
    input_sha256: str
    model_version: str
    output_filename: str
    rows: list[dict[str, Any]]
    summary: dict[str, Any]
    visual: dict[str, Any]
    csv_text: str
    notices: list[str]


class AnalysisMetadata(BaseModel):
    asset_id: str = Field(min_length=1, max_length=120)
    component_info: str = Field(default="", max_length=250)
    measurement_time: datetime

    @field_validator("asset_id", "component_info")
    @classmethod
    def clean_text(cls, value: str, info: ValidationInfo) -> str:
        cleaned = value.strip()
        if info.field_name == "asset_id" and not cleaned:
            raise ValueError("Asset ID cannot be blank")
        return cleaned


class HistoryCreate(AnalysisMetadata):
    result: PredictionResponse


class HistoryMetadataUpdate(AnalysisMetadata):
    pass


TASK_METADATA: dict[str, dict[str, str]] = {
    "door": {
        "name": "Door diagnostics",
        "short_name": "Door",
        "description": "Detect door cycles and classify abnormal resistance.",
    },
    "acv": {
        "name": "ACV leak localisation",
        "short_name": "ACV",
        "description": "Rank train cars by likelihood of a refrigerant leak.",
    },
    "corrugation": {
        "name": "Rail corrugation",
        "short_name": "Rail",
        "description": "Classify Normal, Side I, or Side II corrugation.",
    },
    "shm": {
        "name": "Structural health monitoring",
        "short_name": "SHM",
        "description": "Estimate cumulative fatigue damage from stress history.",
    },
}


def _bundle_path(task: str) -> Path | None:
    raw = os.getenv(f"RAILGUARD_{task.upper()}_BUNDLE", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    default = Path("outputs/checkpoints/competition") / task
    return default.resolve() if default.is_dir() else None


def _bundle_available(task: str) -> bool:
    path = _bundle_path(task)
    return bool(path and path.is_dir() and (path / "model.joblib").is_file())


@lru_cache(maxsize=8)
def _bundle_version(bundle_path: str) -> str:
    path = Path(bundle_path)
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256((path / "model.joblib").read_bytes()).hexdigest()[:12]
    name = str(metadata.get("model") or metadata.get("kind") or path.name)
    version = str(metadata.get("checkpoint_version", 1))
    return f"{name}:v{version}:{digest}"


def _demo_allowed() -> bool:
    return os.getenv("RAILGUARD_ALLOW_DEMO", "true").lower() in {"1", "true", "yes"}


def _cors_origins() -> list[str]:
    raw = os.getenv(
        "RAILGUARD_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    )
    return [origin.strip().rstrip("/") for origin in raw.split(",") if origin.strip()]


@lru_cache(maxsize=8)
def _load_predictor(bundle_path: str) -> RailGuardPredictor:
    return RailGuardPredictor.from_bundle(bundle_path)


def _tasks() -> list[TaskDescriptor]:
    return [
        TaskDescriptor(
            id=task,  # type: ignore[arg-type]
            name=TASK_METADATA[task]["name"],
            short_name=TASK_METADATA[task]["short_name"],
            description=TASK_METADATA[task]["description"],
            accepted_extensions=sorted(ALLOWED_SUFFIXES[task]),
            output_filename=OFFICIAL_FILES[task],
            bundle_available=_bundle_available(task),
        )
        for task in TASKS
    ]


def _csv_text(task: str, rows: list[dict[str, Any]]) -> str:
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=OFFICIAL_COLUMNS[task], lineterminator="\n")
    writer.writeheader()
    writer.writerows({column: row[column] for column in OFFICIAL_COLUMNS[task]} for row in rows)
    return stream.getvalue()


def _probability(item: dict[str, Any], positive: str | None = None) -> float:
    probabilities = item.get("probabilities") or {}
    if positive:
        for label in (positive, positive.lower(), "1", "True"):
            if label in probabilities:
                return float(probabilities[label])
    return float(item.get("confidence") or max(probabilities.values(), default=0.0))


def _format_real(task: str, filename: str, predictions: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    if task == "door":
        rows = [
            {
                "start_time": item["metadata"]["start_time"],
                "end_time": item["metadata"]["end_time"],
                "prediction": DOOR_OFFICIAL.get(str(item["prediction"]), str(item["prediction"])),
            }
            for item in predictions
        ]
        segments = [
            {**row, "confidence": item.get("confidence")}
            for row, item in zip(rows, predictions, strict=True)
        ]
        confidences = [
            float(item["confidence"])
            for item in predictions
            if item.get("confidence") is not None
        ]
        abnormal = sum(row["prediction"] == "Abnormal resistance" for row in rows)
        return rows, {
            "cycles": len(rows),
            "abnormal_cycles": abnormal,
            "confidence": min(confidences) if confidences else None,
        }, {"segments": segments[:50]}
    if task == "acv":
        ordered = sorted(predictions, key=lambda item: _probability(item, "True"), reverse=True)
        ranking = [str(item["sample_id"]).zfill(2) for item in ordered]
        scores = [round(_probability(item, "True"), 6) for item in ordered]
        rows = [{"file_id": filename, "ranked_cars": "|".join(ranking)}]
        return rows, {"top_car": ranking[0], "cars_ranked": len(ranking)}, {"ranking": [{"car": car, "score": score} for car, score in zip(ranking, scores, strict=True)]}
    if task == "corrugation":
        item = predictions[0]
        label = CORRUGATION_OFFICIAL.get(str(item["prediction"]), str(item["prediction"]))
        probabilities = {CORRUGATION_OFFICIAL.get(key, key): value for key, value in (item.get("probabilities") or {}).items()}
        rows = [{"file_id": filename, "prediction": label}]
        return rows, {"prediction": label, "confidence": item.get("confidence")}, {"probabilities": probabilities}
    item = predictions[0]
    value = float(item["prediction"])
    rows = [{"file_id": filename, "prediction": value}]
    return rows, {"predicted_damage": value, "uncertainty": item.get("uncertainty")}, {"damage": value}


def _demo_result(task: str, filename: str) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    seed = int(hashlib.sha256(f"{task}:{filename}".encode()).hexdigest()[:8], 16)
    if task == "door":
        rows = [
            {"start_time": "2026-01-01-09-00-00-000", "end_time": "2026-01-01-09-00-04-200", "prediction": "Normal"},
            {"start_time": "2026-01-01-09-02-10-000", "end_time": "2026-01-01-09-02-15-100", "prediction": "Abnormal resistance"},
        ]
        return rows, {"cycles": 2, "abnormal_cycles": 1}, {"segments": rows}
    if task == "acv":
        cars = [f"{number:02d}" for number in range(1, 9)]
        offset = seed % len(cars)
        ranking = cars[offset:] + cars[:offset]
        scores = [round(value, 2) for value in (0.42, 0.17, 0.12, 0.09, 0.07, 0.06, 0.04, 0.03)]
        rows = [{"file_id": filename, "ranked_cars": "|".join(ranking)}]
        return rows, {"top_car": ranking[0], "cars_ranked": len(ranking)}, {"ranking": [{"car": car, "score": score} for car, score in zip(ranking, scores, strict=True)]}
    if task == "corrugation":
        labels = ["Normal", "Side I", "Side II"]
        label = labels[seed % 3]
        probabilities = {"Normal": 0.12, "Side I": 0.16, "Side II": 0.72}
        if label != "Side II":
            probabilities[label], probabilities["Side II"] = probabilities["Side II"], probabilities[label]
        rows = [{"file_id": filename, "prediction": label}]
        return rows, {"prediction": label, "confidence": probabilities[label]}, {"probabilities": probabilities}
    value = round(0.1 + (seed % 240) / 1000, 4)
    rows = [{"file_id": filename, "prediction": value}]
    return rows, {"predicted_damage": value, "uncertainty": 0.04}, {"damage": value, "interval": [max(0, value - 0.04), value + 0.04]}


app = FastAPI(
    title="RailGuard API",
    version="1.0.0",
    description="Upload boundary for NebulaX PS3 condition-monitoring inference.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "demo_allowed": _demo_allowed(), "bundles": {task: _bundle_available(task) for task in TASKS}}


@app.get("/api/tasks", response_model=list[TaskDescriptor])
def tasks() -> list[TaskDescriptor]:
    return _tasks()


@app.post("/api/metadata/{task}", response_model=MetadataSuggestion)
async def metadata(
    task: TaskName,
    file: Annotated[UploadFile, File(...)],
    file_modified_ms: Annotated[float | None, Query(ge=0)] = None,
) -> MetadataSuggestion:
    filename = Path(file.filename or "upload").name
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES[task]:
        expected = ", ".join(sorted(ALLOWED_SUFFIXES[task]))
        raise HTTPException(415, f"{TASK_METADATA[task]['short_name']} expects {expected} files")
    payload = await file.read(MAX_UPLOAD_BYTES + 1)
    if not payload:
        raise HTTPException(400, "The uploaded file is empty")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "The upload exceeds the 100 MB limit")
    try:
        return extract_metadata(task, payload, filename, file_modified_ms)
    except (ValueError, KeyError, OSError) as exc:
        raise HTTPException(422, f"Metadata extraction failed: {exc}") from exc


@app.post("/api/predict/{task}", response_model=PredictionResponse)
async def predict(
    task: TaskName,
    file: Annotated[UploadFile, File(...)],
    mode: Annotated[RunMode, Query()] = "auto",
) -> PredictionResponse:
    filename = Path(file.filename or "upload").name
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES[task]:
        expected = ", ".join(sorted(ALLOWED_SUFFIXES[task]))
        raise HTTPException(415, f"{TASK_METADATA[task]['short_name']} expects {expected} files")
    payload = await file.read(MAX_UPLOAD_BYTES + 1)
    if not payload:
        raise HTTPException(400, "The uploaded file is empty")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "The upload exceeds the 100 MB limit")

    has_bundle = _bundle_available(task)
    if mode == "real" and not has_bundle:
        raise HTTPException(409, f"No trusted {task} bundle is configured")
    selected_mode: Literal["real", "demo"] = "real" if mode != "demo" and has_bundle else "demo"
    if selected_mode == "demo" and not _demo_allowed():
        raise HTTPException(409, "Demo fallback is disabled and no trusted bundle is configured")

    notices: list[str] = []
    if selected_mode == "real":
        bundle = _bundle_path(task)
        assert bundle is not None
        try:
            with tempfile.TemporaryDirectory(prefix="railguard_api_") as temporary:
                source = Path(temporary) / filename
                source.write_bytes(payload)
                raw_predictions = [asdict(item) for item in _load_predictor(str(bundle)).predict(task, source)]
            rows, summary, visual = _format_real(task, filename, raw_predictions)
        except Exception as exc:
            raise HTTPException(422, f"Inference failed: {exc}") from exc
        notices.append("Generated with a configured trusted model bundle.")
        model_version = _bundle_version(str(bundle))
    else:
        rows, summary, visual = _demo_result(task, filename)
        notices.extend([
            "Demonstration result only. A trained model was not used for this result.",
            "Do not include this output in a competition submission.",
        ])
        model_version = "demo-v1"

    import pandas as pd

    validate_prediction_frame(task, pd.DataFrame(rows))
    return PredictionResponse(
        task=task,
        task_name=TASK_METADATA[task]["name"],
        mode=selected_mode,
        source_file=filename,
        input_sha256=hashlib.sha256(payload).hexdigest(),
        model_version=model_version,
        output_filename=OFFICIAL_FILES[task],
        rows=rows,
        summary=summary,
        visual=visual,
        csv_text=_csv_text(task, rows),
        notices=notices,
    )


@app.post("/api/history", status_code=status.HTTP_201_CREATED)
def save_history(payload: HistoryCreate) -> dict[str, Any]:
    return create_analysis(
        asset_id=payload.asset_id,
        component_info=payload.component_info,
        measurement_time=payload.measurement_time.isoformat(),
        result=payload.result.model_dump(mode="json"),
    )


@app.get("/api/history")
def history(
    asset: Annotated[str | None, Query(max_length=120)] = None,
    task: Annotated[TaskName | None, Query()] = None,
    date_from: Annotated[datetime | None, Query()] = None,
    date_to: Annotated[datetime | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=1000)] = 500,
) -> list[dict[str, Any]]:
    return list_analyses(
        asset=asset.strip() if asset else None,
        task=task,
        date_from=date_from.isoformat() if date_from else None,
        date_to=date_to.isoformat() if date_to else None,
        limit=limit,
    )


@app.get("/api/history/{analysis_id}")
def history_detail(analysis_id: str) -> dict[str, Any]:
    try:
        return get_analysis(analysis_id)
    except KeyError as exc:
        raise HTTPException(404, "Analysis history record was not found") from exc


@app.patch("/api/history/{analysis_id}/metadata")
def edit_history_metadata(
    analysis_id: str, payload: HistoryMetadataUpdate
) -> dict[str, Any]:
    try:
        return update_metadata(
            analysis_id,
            asset_id=payload.asset_id,
            component_info=payload.component_info,
            measurement_time=payload.measurement_time.isoformat(),
        )
    except KeyError as exc:
        raise HTTPException(404, "Analysis history record was not found") from exc


@app.get("/", response_class=PlainTextResponse)
def root() -> str:
    return "RailGuard API is running. Open /docs for the API contract."
