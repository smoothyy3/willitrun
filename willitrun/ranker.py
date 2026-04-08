"""Inverse query: given a device + category, return ranked model list.

Combines Tier 1 (real benchmark data) and Tier 2 (scaled estimates) so
every model in the category gets a result, with real data ranked above
estimates at the same performance level.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import data_access
from .estimator import (
    _is_pp_benchmark,
    _load_benchmarks,
    _load_devices,
    _load_models_db,
    _resolve_model_key,
    tier2_estimate,
)
from .loader import ModelInfo
from .profiler import profile_model


@dataclass
class RankedModelResult:
    """A single entry in the ranked model list."""

    model_id: str
    model_name: str
    fps: float | None
    fps_range: tuple[float, float] | None
    metric_label: str          # "fps" or "tok/s"
    confidence: str            # "measured" | "estimated"
    source: str | None         # human-readable source label for real benchmarks
    params: str                # human-readable, e.g. "7.2B"
    weights_mb: float | None


def _fmt_params(params: int | None) -> str:
    if params is None:
        return "?"
    if params >= 1_000_000_000:
        return f"{params / 1_000_000_000:.1f}B"
    if params >= 1_000_000:
        return f"{params / 1_000_000:.1f}M"
    return f"{params / 1_000:.0f}K"


def list_categories() -> list[str]:
    """Return distinct model_type values present in the models table."""
    # Always surface the most common categories first, then alphabetical others
    PRIORITY = ["llm", "detection", "classification"]
    with data_access._connect() as conn:
        rows = conn.execute(
            "SELECT DISTINCT model_type FROM models "
            "WHERE model_type IS NOT NULL ORDER BY model_type"
        ).fetchall()
    found = [r["model_type"] for r in rows]
    ordered = [c for c in PRIORITY if c in found]
    ordered += [c for c in sorted(found) if c not in ordered]
    return ordered


def get_best_models_for_device(
    device_id: str,
    category: str,
    limit: int = 10,
) -> list[RankedModelResult]:
    """Return up to `limit` ranked models for a device + category.

    Real benchmark data ranks above estimates in the same performance
    band via a confidence-weighted sort key (estimated fps × 0.7).
    """
    benchmarks = _load_benchmarks()
    devices    = _load_devices()
    models_db  = _load_models_db()

    # All models in this category, keyed by canonical model_id
    category_models = {
        m.model_id: m
        for m in data_access.list_models()
        if m.model_type == category
    }
    if not category_models:
        return []

    is_llm = (category == "llm")
    results: list[RankedModelResult] = []
    benchmarked_ids: set[str] = set()

    # ── Step 1: real benchmarks on this device ──────────────────────────────
    # Deduplicate per model: keep highest value.  Skip pp benchmarks for LLMs —
    # tok/s prompt-processing is 10-30× higher and misleading as a headline.
    best_by_model: dict[str, object] = {}
    for b in benchmarks:
        if b.device != device_id:
            continue
        if is_llm and _is_pp_benchmark(b):
            continue
        canon = _resolve_model_key(b.model, models_db) or b.model
        if canon not in category_models:
            continue
        prev = best_by_model.get(canon)
        if prev is None or b.value > prev.value:  # type: ignore[union-attr]
            best_by_model[canon] = b

    for canon, b in best_by_model.items():
        m = category_models[canon]
        results.append(RankedModelResult(
            model_id=canon,
            model_name=canon,
            fps=b.value,  # type: ignore[union-attr]
            fps_range=None,
            metric_label=b.metric,  # type: ignore[union-attr]
            confidence="measured",
            source=b.source or None,  # type: ignore[union-attr]
            params=_fmt_params(m.parameters),
            weights_mb=m.weights_size_mb,
        ))
        benchmarked_ids.add(canon)

    # ── Step 2: Tier 2 estimates for models with no real data on this device ─
    for model_id, m in category_models.items():
        if model_id in benchmarked_ids:
            continue

        model_info = ModelInfo(
            name=model_id,
            parameters=m.parameters,
            active_params=None,
            is_moe=False,
            flops=m.flops,
            model_type=m.model_type,
            architecture=m.architecture,
            weights_size_mb=m.weights_size_mb,
            llm_config=m.llm_config,
            source="database",
        )
        profile = profile_model(model_info)

        est_val, est_range, _, _, strategy = tier2_estimate(
            profile, device_id, benchmarks, devices, models_db
        )
        if est_val is None or strategy == 0:
            continue  # not enough info for a useful estimate

        metric_label = "tok/s" if is_llm else "fps"
        results.append(RankedModelResult(
            model_id=model_id,
            model_name=model_id,
            fps=est_val,
            fps_range=est_range,
            metric_label=metric_label,
            confidence="estimated",
            source=None,
            params=_fmt_params(m.parameters),
            weights_mb=m.weights_size_mb,
        ))

    # ── Step 3: confidence-weighted sort ────────────────────────────────────
    # A real 45 fps beats an estimated 60 fps; within the same confidence band
    # sort strictly by performance.
    def _sort_key(r: RankedModelResult) -> float:
        if r.fps is None:
            return -1.0
        return r.fps if r.confidence == "measured" else r.fps * 0.7

    results.sort(key=_sort_key, reverse=True)
    return results[:limit]
