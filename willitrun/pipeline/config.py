"""Pipeline configuration helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

# tomllib is stdlib from Python 3.11; use the tomli backport on 3.9/3.10.
if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "tomli is required on Python < 3.11: pip install tomli"
        ) from exc

DEFAULT_PRIORITY = [
    "llama_cpp",
    "mlperf",
    "nvidia_jetson",
    "ultralytics",
    "ultralytics_gpu",
    "xiongjieddai",
    "localscore",
    "geerlingguy",
    "curated",
]


class PipelineConfig:
    def __init__(self, ttl_days: int = 7, priority: Optional[List[str]] = None):
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
