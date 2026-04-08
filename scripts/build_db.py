#!/usr/bin/env python3
"""Build data/benchmarks.db from normalized JSONL sources."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from willitrun.pipeline.config import load_config
from willitrun.pipeline.schema import BenchmarkRecord, Device, Model

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


def load_yaml(path: Path) -> dict | list:
    import yaml

    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_devices() -> dict[str, Device]:
    raw = load_yaml(DATA_DIR / "devices.yaml")
    devices = {}
    for device_id, spec in raw.items():
        try:
            spec["device_id"] = device_id
            mem = spec.get("memory", {})
            if mem and "total_gb" in mem:
                spec.setdefault("memory_gb", mem.get("total_gb"))
            devices[device_id] = Device(**spec)
        except Exception as exc:
            print(f"[devices] skip {device_id}: {exc}")
    return devices


def load_models() -> dict[str, Model]:
    raw = load_yaml(DATA_DIR / "models.yaml")
    models = {}
    for model_id, spec in raw.items():
        if not isinstance(spec, dict):
            continue
        try:
            spec["model_id"] = model_id
            models[model_id] = Model(**spec)
        except Exception as exc:
            print(f"[models] skip {model_id}: {exc}")
    return models


def list_normalized() -> list[Path]:
    return sorted((DATA_DIR / "normalized").glob("*.jsonl"))


def _resolve_model_id(model_id: str, models: dict[str, Model]) -> str | None:
    if model_id in models:
        return model_id
    for key, m in models.items():
        aliases = m.aliases or []
        if model_id in aliases:
            return key
    return None


def parse_record(entry: dict, devices: dict[str, Device], models: dict[str, Model]) -> BenchmarkRecord | None:
    # Map legacy keys to normalized schema
    def map_precision(p: str) -> str:
        p = p.lower()
        if p in {"4bit", "int4", "q4_k_m"}:  # Q4_K_M is 4-bit; normalise to int4
            return "int4"
        if p in {"8bit", "int8"}:
            return "int8"
        if p in {"q4_0"}:
            return "q4_0"
        if p in {"q8_0"}:
            return "q8_0"
        return p

    def map_metric(m: str, input_size: str | None) -> str:
        m_low = m.lower()
        # Already-normalized values from ingest scripts — pass through directly
        if m_low in {"tok_s_tg", "tok_s_pp", "fps", "latency_ms", "samples_s"}:
            return m_low
        # Legacy display strings from older ingest scripts
        if m_low in {"tok/s", "toks", "tokens/s"}:
            if input_size and input_size.startswith("pp"):
                return "tok_s_pp"
            return "tok_s_tg"
        if m_low == "latency_ms":
            return "latency_ms"
        if m_low == "samples/s":
            return "samples_s"
        return "fps"

    input_size = None
    bid = entry.get("benchmark_id", "")
    if not input_size and bid:
        parts = bid.split("__")
        if len(parts) >= 5:
            candidate = parts[4]
            if not candidate.startswith("bs"):
                input_size = candidate

    precision = map_precision(entry.get("precision", "fp16"))
    metric = map_metric(entry.get("metric", "fps"), input_size)

    source = entry.get("source", {})
    if isinstance(source, dict) and source:
        # Nested source dict — format used by nvidia_jetson, mlperf, ultralytics ingest scripts
        source_name = source.get("name", source.get("type", "unknown"))
        source_url = source.get("url", "")
        confidence_raw = str(source.get("confidence", "community")).lower()
    else:
        # Flat fields — format emitted by BenchmarkRecord.model_dump_json()
        # (used by mac_llm_bench and any future ingest that writes BenchmarkRecord directly)
        source_name = entry.get("source_name") or (str(source) if source else "unknown")
        source_url = entry.get("source_url", "")
        confidence_raw = entry.get("confidence", "community").lower()

    confidence_map = {
        "high": "measured",
        "medium": "community",
        "low": "community",
        "official": "official",
        "community": "community",
        "estimated": "estimated",
        "measured": "measured",
    }
    confidence = confidence_map.get(confidence_raw, "community")

    collected_at = entry.get("provenance", {}).get("collected_at") or entry.get("collected_at")
    if collected_at is None:
        collected_at = datetime.now(timezone.utc).isoformat()

    model_id = entry.get("model_id") or entry.get("model")
    model_id = _resolve_model_id(model_id, models) if model_id else model_id

    try:
        record = BenchmarkRecord(
            benchmark_id=bid,
            model_id=model_id,
            device_id=entry.get("device_id") or entry.get("device"),
            precision=precision,
            metric=metric,
            value=float(entry["value"]),
            framework=entry.get("framework", "unknown"),
            source_url=source_url or entry.get("source_url") or "",
            source_name=source_name or "unknown",
            confidence=confidence,
            collected_at=collected_at,
            notes=entry.get("notes", ""),
        )
    except ValidationError as exc:
        print(f"  [SKIP] {bid}: {exc}")
        return None

    if record.device_id not in devices:
        print(f"  [SKIP] {bid}: device_id '{record.device_id}' not in devices.yaml")
        return None
    if record.model_id not in models:
        print(f"  [SKIP] {bid}: model_id '{record.model_id}' not in models.yaml")
        return None
    return record


def load_normalized(devices: dict[str, Device], models: dict[str, Model]) -> dict[str, list[BenchmarkRecord]]:
    records: dict[str, list[BenchmarkRecord]] = {}
    for path in list_normalized():
        source = path.stem
        records[source] = []
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                rec = parse_record(entry, devices, models)
                if rec:
                    records[source].append(rec)
    return records


def load_curated(devices: dict[str, Device], models: dict[str, Model]) -> list[BenchmarkRecord]:
    curated_raw = load_yaml(DATA_DIR / "benchmarks.curated.yaml")
    curated_records = []
    if isinstance(curated_raw, list):
        for entry in curated_raw:
            rec = parse_record(entry, devices, models)
            if rec:
                curated_records.append(rec)
    return curated_records


def deduplicate(priority: list[str], sources: dict[str, list[BenchmarkRecord]], curated: list[BenchmarkRecord]) -> tuple[list[BenchmarkRecord], int]:
    order = {name: i for i, name in enumerate(priority)}
    pool = []
    for name, recs in sources.items():
        pool.extend([(order.get(name, 999), r) for r in recs])
    # curated always wins
    pool.extend([(-1, r) for r in curated])
    pool.sort(key=lambda x: x[0])

    seen = {}
    collisions = 0
    deduped = []
    for _, rec in pool:
        if rec.benchmark_id in seen:
            collisions += 1
            continue
        seen[rec.benchmark_id] = rec
        deduped.append(rec)
    return deduped, collisions


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        DROP TABLE IF EXISTS benchmarks;
        DROP TABLE IF EXISTS devices;
        DROP TABLE IF EXISTS models;

        CREATE TABLE benchmarks (
            benchmark_id TEXT PRIMARY KEY,
            model_id TEXT NOT NULL,
            device_id TEXT NOT NULL,
            precision TEXT NOT NULL,
            metric TEXT NOT NULL,
            value REAL NOT NULL,
            framework TEXT NOT NULL,
            source_url TEXT NOT NULL,
            source_name TEXT NOT NULL,
            confidence TEXT NOT NULL,
            collected_at TEXT NOT NULL,
            notes TEXT
        );

        CREATE TABLE devices (
            device_id TEXT PRIMARY KEY,
            name TEXT,
            gpu JSON,
            cpu JSON,
            memory JSON,
            memory_gb REAL,
            notes TEXT
        );

        CREATE TABLE models (
            model_id TEXT PRIMARY KEY,
            name TEXT,
            aliases JSON,
            model_type TEXT,
            parameters INTEGER,
            flops INTEGER,
            default_input_size TEXT,
            architecture TEXT,
            weights_size_mb REAL,
            llm_config JSON,
            notes TEXT
        );

        CREATE INDEX idx_model_device ON benchmarks (model_id, device_id);
        CREATE INDEX idx_device ON benchmarks (device_id);
        CREATE INDEX idx_model ON benchmarks (model_id);
        """
    )


def insert_all(conn: sqlite3.Connection, records: list[BenchmarkRecord], devices: dict[str, Device], models: dict[str, Model]) -> None:
    conn.executemany(
        """
        INSERT INTO benchmarks (
            benchmark_id, model_id, device_id, precision, metric, value,
            framework, source_url, source_name, confidence, collected_at, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                r.benchmark_id,
                r.model_id,
                r.device_id,
                r.precision,
                r.metric,
                r.value,
                r.framework,
                r.source_url,
                r.source_name,
                r.confidence,
                r.collected_at.isoformat(),
                r.notes,
            )
            for r in records
        ],
    )

    conn.executemany(
        """
        INSERT INTO devices (device_id, name, gpu, cpu, memory, memory_gb, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                d.device_id,
                d.name,
                json.dumps(d.gpu) if d.gpu is not None else None,
                json.dumps(d.cpu) if d.cpu is not None else None,
                json.dumps(d.memory) if getattr(d, "memory", None) is not None else None,
                d.memory_gb,
                d.notes,
            )
            for d in devices.values()
        ],
    )

    conn.executemany(
        """
        INSERT INTO models (model_id, name, aliases, model_type, parameters, flops,
                            default_input_size, architecture, weights_size_mb, llm_config, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                m.model_id,
                m.name,
                json.dumps(m.aliases) if m.aliases is not None else None,
                m.model_type,
                m.parameters,
                m.flops,
                m.default_input_size,
                m.architecture,
                m.weights_size_mb,
                json.dumps(m.llm_config) if m.llm_config is not None else None,
                m.notes,
            )
            for m in models.values()
        ],
    )


def build_db(validate_only: bool = False) -> None:
    cfg = load_config(ROOT)
    devices = load_devices()
    models = load_models()
    sources = load_normalized(devices, models)
    curated = load_curated(devices, models)

    deduped, collisions = deduplicate(cfg.priority, sources, curated)
    report = defaultdict(int)
    for src, records in sources.items():
        report[src] = len(records)

    print("Build report:")
    total = 0
    for src, count in report.items():
        print(f"  {src:<20} {count:5d} records")
        total += count
    print(f"  {'curated':<20} {len(curated):5d} records")
    print(f"Total records before dedup: {total + len(curated)}")
    print(f"Total records after  dedup: {len(deduped)}")
    print(f"Collisions resolved by priority: {collisions}")

    if validate_only:
        return

    db_path = DATA_DIR / "benchmarks.db"
    conn = sqlite3.connect(db_path)
    try:
        ensure_schema(conn)
        insert_all(conn, deduped, devices, models)
        conn.commit()
        print(f"Wrote {len(deduped)} records to {db_path}")
    finally:
        conn.close()

    meta = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "total_records": len(deduped),
        "per_source": report,
        "collisions": collisions,
    }
    meta_path = DATA_DIR / "benchmarks.meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Wrote metadata to {meta_path}")


def main():
    parser = argparse.ArgumentParser(description="Build SQLite benchmarks database.")
    parser.add_argument("--validate-only", action="store_true", help="Validate only, do not write the DB.")
    args = parser.parse_args()
    build_db(validate_only=args.validate_only)


if __name__ == "__main__":
    main()
