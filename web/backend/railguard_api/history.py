"""Local SQLite persistence for immutable analysis results."""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


def database_path() -> Path:
    configured = os.getenv("RAILGUARD_HISTORY_DB", "data/railguard.db")
    return Path(configured).expanduser().resolve()


def _connect() -> sqlite3.Connection:
    path = database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS analyses (
            id TEXT PRIMARY KEY,
            asset_id TEXT NOT NULL,
            component_info TEXT NOT NULL DEFAULT '',
            measurement_time TEXT NOT NULL,
            analysis_time TEXT NOT NULL,
            task TEXT NOT NULL CHECK (task IN ('door', 'acv', 'corrugation', 'shm')),
            mode TEXT NOT NULL CHECK (mode IN ('real', 'demo')),
            source_file TEXT NOT NULL,
            input_sha256 TEXT NOT NULL,
            model_version TEXT NOT NULL,
            result_json TEXT NOT NULL
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_analyses_asset_time "
        "ON analyses(asset_id, measurement_time DESC)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_analyses_task_time "
        "ON analyses(task, measurement_time DESC)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_analyses_input_hash ON analyses(input_sha256)"
    )
    return connection


def _record(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "asset_id": row["asset_id"],
        "component_info": row["component_info"],
        "measurement_time": row["measurement_time"],
        "analysis_time": row["analysis_time"],
        "task": row["task"],
        "mode": row["mode"],
        "source_file": row["source_file"],
        "input_sha256": row["input_sha256"],
        "model_version": row["model_version"],
        "result": json.loads(row["result_json"]),
    }


def create_analysis(
    *,
    asset_id: str,
    component_info: str,
    measurement_time: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    analysis_id = str(uuid4())
    analysis_time = datetime.now(timezone.utc).isoformat()
    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO analyses (
                id, asset_id, component_info, measurement_time, analysis_time,
                task, mode, source_file, input_sha256, model_version, result_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                analysis_id,
                asset_id,
                component_info,
                measurement_time,
                analysis_time,
                result["task"],
                result["mode"],
                result["source_file"],
                result["input_sha256"],
                result["model_version"],
                json.dumps(result, separators=(",", ":"), ensure_ascii=False),
            ),
        )
    return get_analysis(analysis_id)


def get_analysis(analysis_id: str) -> dict[str, Any]:
    with _connect() as connection:
        row = connection.execute(
            "SELECT * FROM analyses WHERE id = ?", (analysis_id,)
        ).fetchone()
    if row is None:
        raise KeyError(analysis_id)
    return _record(row)


def list_analyses(
    *,
    asset: str | None = None,
    task: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int = 500,
) -> list[dict[str, Any]]:
    clauses: list[str] = []
    parameters: list[Any] = []
    if asset:
        clauses.append("LOWER(asset_id) LIKE LOWER(?)")
        parameters.append(f"%{asset}%")
    if task:
        clauses.append("task = ?")
        parameters.append(task)
    if date_from:
        clauses.append("measurement_time >= ?")
        parameters.append(date_from)
    if date_to:
        clauses.append("measurement_time <= ?")
        parameters.append(date_to)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    parameters.append(limit)
    with _connect() as connection:
        rows = connection.execute(
            f"SELECT * FROM analyses{where} ORDER BY measurement_time DESC, analysis_time DESC LIMIT ?",
            parameters,
        ).fetchall()
    return [_record(row) for row in rows]


def update_metadata(
    analysis_id: str, *, asset_id: str, component_info: str, measurement_time: str
) -> dict[str, Any]:
    with _connect() as connection:
        cursor = connection.execute(
            """
            UPDATE analyses
            SET asset_id = ?, component_info = ?, measurement_time = ?
            WHERE id = ?
            """,
            (asset_id, component_info, measurement_time, analysis_id),
        )
        if cursor.rowcount == 0:
            raise KeyError(analysis_id)
    return get_analysis(analysis_id)
