"""Shared data-loading utilities (SQLite-backed)."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

from . import data_access


def data_dir() -> Path:
    pkg_data = Path(resources.files("willitrun")) / "data"
    if pkg_data.is_dir():
        return pkg_data
    return Path(__file__).resolve().parent.parent / "data"


def load_devices() -> dict:
    """Return devices keyed by device_id."""
    return {d.device_id: d.model_dump() for d in data_access.list_devices()}


def load_models() -> dict:
    return {m.model_id: m.model_dump() for m in data_access.list_models()}


def load_models_raw() -> dict:
    return load_models()
