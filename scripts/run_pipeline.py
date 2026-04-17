#!/usr/bin/env python3
"""Unified pipeline entrypoint."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from willitrun.pipeline.config import load_config

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
NORMALIZED_DIR = ROOT / "data" / "normalized"


SCRIPTS_DIR = ROOT / "scripts"

SOURCES = {
    "ultralytics": SCRIPTS_DIR / "ingest_ultralytics.py",
    "ultralytics_gpu": SCRIPTS_DIR / "ingest_ultralytics_gpu.py",
    "nvidia_jetson": SCRIPTS_DIR / "ingest_nvidia_jetson.py",
    "mlperf": SCRIPTS_DIR / "ingest_mlperf.py",
    "llama_cpp": SCRIPTS_DIR / "ingest_llama_cpp.py",
    "xiongjieddai": SCRIPTS_DIR / "ingest_xiongjieddai.py",
    "localscore": SCRIPTS_DIR / "ingest_localscore.py",
    "geerlingguy": SCRIPTS_DIR / "ingest_geerlingguy.py",
}


def _latest_raw_age(source: str) -> float | None:
    raw_dir = DATA_RAW / source
    if not raw_dir.exists():
        return None
    newest = max(raw_dir.glob("*"), default=None)
    if not newest:
        return None
    mtime = newest.stat().st_mtime
    return (datetime.now(timezone.utc).timestamp() - mtime) / 86400


def run_fetch(cfg) -> None:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    for source, script in SOURCES.items():
        age = _latest_raw_age(source)
        if age is not None and age < cfg.ttl_days:
            print(f"[fetch] {source}: cache hit, skipping fetch ({age:.1f} days old)")
            continue
        print(f"[fetch] {source}: running {script.name}")
        subprocess.run([sys.executable, str(script)], check=True)


def run_build() -> None:
    subprocess.run([sys.executable, str(SCRIPTS_DIR / "build_db.py")], check=True)


def run_status() -> None:
    from willitrun.data_access import get_coverage_report

    coverage, gaps = get_coverage_report()
    print("Source coverage:")
    print("Source               | Records | Devices | Models | Last fetched")
    for row in coverage:
        last = row.get("last_fetched") or ""
        print(f"{row['source']:<20}| {row['records']:7d} | {row['devices']:7d} | {row['models']:6d} | {last}")

    print("\nCoverage gaps:")
    print("Device               | Has vision | Has LLM | Has real benchmark")
    for row in gaps:
        def mark(flag: bool) -> str:
            return "✓" if flag else "✗"
        print(f"{row['device']:<20}| {mark(row['has_vision']):^11} | {mark(row['has_llm']):^7} | {mark(row['has_real']):^19}")


def main():
    parser = argparse.ArgumentParser(description="Run willitrun data pipeline.")
    parser.add_argument("command", choices=["fetch", "build", "all", "status"])
    args = parser.parse_args()

    cfg = load_config(ROOT)
    if args.command == "fetch":
        run_fetch(cfg)
    elif args.command == "build":
        run_build()
    elif args.command == "all":
        run_fetch(cfg)
        run_build()
    elif args.command == "status":
        run_status()


if __name__ == "__main__":
    main()
