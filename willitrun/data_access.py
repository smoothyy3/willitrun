"""Thin SQLite data access layer for the CLI."""

from __future__ import annotations

import sqlite3
import json
from importlib import resources
from pathlib import Path
from typing import Iterable

from .pipeline.schema import BenchmarkRecord, Device, Model


def _db_path() -> Path:
    """Resolve the bundled database path."""
    # packaged data lives inside willitrun/data/
    pkg_dir = Path(resources.files("willitrun")) / "data"
    candidate = pkg_dir / "benchmarks.db"
    if candidate.exists():
        return candidate
    # fallback to repo root during development
    return Path(__file__).resolve().parent.parent / "data" / "benchmarks.db"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row
    return conn


def get_benchmark(model_id: str, device_id: str) -> list[BenchmarkRecord]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT benchmark_id, model_id, device_id, precision, metric, value,
                   framework, source_url, source_name, confidence, collected_at, notes
            FROM benchmarks
            WHERE model_id = ? AND device_id = ?
            """,
            (model_id, device_id),
        ).fetchall()
    return [BenchmarkRecord(**dict(r)) for r in rows]


def get_device(device_id: str) -> Device | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM devices WHERE device_id = ?", (device_id,)
        ).fetchone()
    if not row:
        return None
    data = dict(row)
    for key in ("gpu", "cpu", "memory"):
        if data.get(key):
            data[key] = json.loads(data[key])
    return Device(**data)


def get_model(model_id: str) -> Model | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM models WHERE model_id = ?", (model_id,)
        ).fetchone()
    if not row:
        return None
    data = dict(row)
    for key in ("aliases", "llm_config"):
        if data.get(key):
            data[key] = json.loads(data[key])
    return Model(**data)


def list_devices(filter: str | None = None) -> list[Device]:
    sql = "SELECT * FROM devices"
    params: Iterable[str] = []
    if filter:
        sql += " WHERE device_id LIKE ? OR name LIKE ?"
        params = (f"%{filter}%", f"%{filter}%")
    with _connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    devices = []
    for r in rows:
        data = dict(r)
        for key in ("gpu", "cpu", "memory"):
            if data.get(key):
                data[key] = json.loads(data[key])
        devices.append(Device(**data))
    return devices


def list_models(filter: str | None = None) -> list[Model]:
    sql = "SELECT * FROM models"
    params: Iterable[str] = []
    if filter:
        sql += " WHERE model_id LIKE ? OR name LIKE ?"
        params = (f"%{filter}%", f"%{filter}%")
    with _connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    models = []
    for r in rows:
        data = dict(r)
        for key in ("aliases", "llm_config"):
            if data.get(key):
                data[key] = json.loads(data[key])
        models.append(Model(**data))
    return models


def list_benchmarks() -> list[BenchmarkRecord]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT benchmark_id, model_id, device_id, precision, metric, value,
                   framework, source_url, source_name, confidence, collected_at, notes
            FROM benchmarks
            """
        ).fetchall()
    return [BenchmarkRecord(**dict(r)) for r in rows]


def get_coverage_report() -> tuple[list[dict], list[dict]]:
    """Return (coverage_by_source, gaps_by_device)."""
    with _connect() as conn:
        coverage = conn.execute(
            """
            SELECT source_name AS source,
                   COUNT(*) AS records,
                   COUNT(DISTINCT device_id) AS devices,
                   COUNT(DISTINCT model_id) AS models,
                   MAX(collected_at) AS last_fetched
            FROM benchmarks
            GROUP BY source_name
            ORDER BY source_name
            """
        ).fetchall()

        gaps = conn.execute(
            """
            SELECT d.device_id,
                   CASE WHEN EXISTS(
                        SELECT 1 FROM benchmarks b
                        JOIN models m ON b.model_id = m.model_id
                        WHERE b.device_id = d.device_id AND m.model_type = 'vision'
                   ) THEN 1 ELSE 0 END AS has_vision,
                   CASE WHEN EXISTS(
                        SELECT 1 FROM benchmarks b
                        JOIN models m ON b.model_id = m.model_id
                        WHERE b.device_id = d.device_id AND m.model_type = 'llm'
                   ) THEN 1 ELSE 0 END AS has_llm,
                   CASE WHEN EXISTS(
                        SELECT 1 FROM benchmarks b WHERE b.device_id = d.device_id
                   ) THEN 1 ELSE 0 END AS has_any
            FROM devices d
            ORDER BY d.device_id
            """
        ).fetchall()

    coverage_rows = [dict(r) for r in coverage]
    gaps_rows = [
        {
            "device": r["device_id"],
            "has_vision": bool(r["has_vision"]),
            "has_llm": bool(r["has_llm"]),
            "has_real": bool(r["has_any"]),
        }
        for r in gaps
    ]
    return coverage_rows, gaps_rows
