"""Pipeline configuration helpers."""

from __future__ import annotations

import tomllib
from pathlib import Path

DEFAULT_PRIORITY = [
    "llama_cpp",
    "mlperf",
    "nvidia_jetson",
    "ultralytics",
    "ultralytics_gpu",
    "xiongjieddai",
    "curated",
]


class PipelineConfig:
    def __init__(self, ttl_days: int = 7, priority: list[str] | None = None):
        self.ttl_days = ttl_days
        self.priority = priority or DEFAULT_PRIORITY


def load_config(repo_root: Path | None = None) -> PipelineConfig:
    root = repo_root or Path(__file__).resolve().parent.parent
    pyproject = root / "pyproject.toml"
    if not pyproject.exists():
        return PipelineConfig()

    with open(pyproject, "rb") as f:
        data = tomllib.load(f)
    cfg = data.get("tool", {}).get("willitrun", {})
    ttl = int(cfg.get("raw_ttl_days", 7))
    priority = cfg.get("source_priority") or DEFAULT_PRIORITY
    return PipelineConfig(ttl_days=ttl, priority=list(priority))
