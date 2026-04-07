"""Pipeline schemas and helpers for willitrun.

This module defines the normalized record shape used by the ingestion
pipeline and downstream SQLite build step.  It is intentionally strict to
keep the dataset reproducible and append‑friendly.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, HttpUrl, field_validator


Precision = Literal["fp32", "fp16", "bf16", "int8", "int4", "q4_0", "q8_0", "q4_k_m"]
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
    source_url: Union[HttpUrl, str]
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
    name: Optional[str] = None
    gpu: Optional[Dict] = None
    cpu: Optional[Dict] = None
    memory: Optional[Dict] = None
    memory_gb: Optional[float] = None
    notes: Optional[str] = None


class Model(BaseModel):
    model_id: str
    name: Optional[str] = None
    aliases: Optional[List[str]] = None
    model_type: Optional[str] = None
    parameters: Optional[int] = None
    flops: Optional[int] = None
    default_input_size: Optional[str] = None
    architecture: Optional[str] = None
    weights_size_mb: Optional[float] = None
    llm_config: Optional[Dict] = None
    notes: Optional[str] = None


def unique_benchmark_key(source: str, model_id: str, device_id: str, precision: str, metric: str) -> str:
    """Deterministic benchmark_id helper."""
    return f"{source}_{model_id}_{device_id}_{precision}_{metric}"


# Alias used by ingest scripts — keeps their imports stable as the module
# name evolves.  New scripts should call unique_benchmark_key() directly.
make_benchmark_id = unique_benchmark_key
