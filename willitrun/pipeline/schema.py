"""Pipeline schemas and helpers for willitrun.

This module defines the normalized record shape used by the ingestion
pipeline and downstream SQLite build step.  It is intentionally strict to
keep the dataset reproducible and append‑friendly.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl, field_validator


Precision = Literal["fp32", "fp16", "bf16", "int8", "int4", "q4_0", "q8_0"]
Metric = Literal["fps", "tok_s_tg", "tok_s_pp", "latency_ms", "samples_s"]
Confidence = Literal["measured", "official", "community", "estimated"]


class BenchmarkRecord(BaseModel):
    """Normalized benchmark row used across the pipeline."""

    benchmark_id: str
    model_id: str
    device_id: str
    precision: Precision
    metric: Metric
    value: float = Field(gt=0)
    framework: str
    source_url: HttpUrl | str
    source_name: str
    confidence: Confidence
    collected_at: datetime
    notes: str = ""

    @field_validator("benchmark_id")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("benchmark_id must be non-empty")
        return v


class Device(BaseModel):
    device_id: str
    name: str | None = None
    gpu: dict | None = None
    cpu: dict | None = None
    memory: dict | None = None
    memory_gb: float | None = None
    notes: str | None = None


class Model(BaseModel):
    model_id: str
    name: str | None = None
    aliases: list[str] | None = None
    model_type: str | None = None
    parameters: int | None = None
    flops: int | None = None
    default_input_size: str | None = None
    architecture: str | None = None
    weights_size_mb: float | None = None
    llm_config: dict | None = None
    notes: str | None = None


def unique_benchmark_key(source: str, model_id: str, device_id: str, precision: str, metric: str) -> str:
    """Deterministic benchmark_id helper."""
    return f"{source}_{model_id}_{device_id}_{precision}_{metric}"
