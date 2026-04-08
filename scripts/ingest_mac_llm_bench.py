#!/usr/bin/env python3
"""Ingest mac-llm-bench JSON results into willitrun normalized JSONL.

Supports both the gguf/ (Q4_K_M via llama.cpp) and mlx/ (4-bit MLX) sub-
directories produced by the mac-llm-bench tool (v2.0 JSON format).

Usage
-----
  python scripts/ingest_mac_llm_bench.py \
      --results-dir /path/to/m5_10c-10g_32gb \
      --device-id apple-m5-32gb

Output
------
  data/raw/mac_llm_bench/<device_id>/<YYYY-MM-DD>/  — archived raw JSON
  data/normalized/mac_llm_bench.jsonl               — appended normalized records
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from willitrun.pipeline.schema import BenchmarkRecord  # noqa: E402

SOURCE_NAME = "mac_llm_bench"
SOURCE_LABEL = "mac-llm-bench"
SOURCE_URL = "https://github.com/jonas/mac-llm-bench"

# Maps benchmark model IDs → willitrun canonical model IDs.
# Only entries that differ from the benchmark ID need to be listed;
# unrecognised IDs are kept as-is (they'll resolve via alias lookup in the DB).
MODEL_MAP: dict[str, str] = {
    # llama
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B",
    "llama-3.2-1b": "meta-llama/Llama-3.2-1B",
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B",
    # mistral
    "mistral-7b": "mistralai/Mistral-7B-v0.3",
    "mistral-nemo-12b": "mistralai/Mistral-Nemo-12B",
    "mistral-small-3.1-24b": "mistralai/Mistral-Small-3.1-24B",
    "devstral-small-24b": "mistralai/Devstral-Small-24B",
    # gemma 3
    "gemma-3-1b": "google/gemma-3-1b",
    "gemma-3-4b": "google/gemma-3-4b",
    "gemma-3-12b": "google/gemma-3-12b",
    "gemma-3-27b": "google/gemma-3-27b",
    # gemma 4
    "gemma-4-e2b": "google/gemma-4-e2b",
    "gemma-4-e4b": "google/gemma-4-e4b",
    "gemma-4-26b-a4b": "google/gemma-4-26b-a4b",
    "gemma-4-31b": "google/gemma-4-31b",
    # deepseek r1 distill
    "deepseek-r1-distill-7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-r1-distill-14b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "deepseek-r1-distill-32b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    # phi 4
    "phi-4": "microsoft/Phi-4",
    "phi-4-mini": "microsoft/Phi-4-mini",
    "phi-4-mini-reasoning": "microsoft/Phi-4-mini-reasoning",
    "phi-4-reasoning": "microsoft/Phi-4-reasoning",
    # qwen 2.5 coder
    "qwen2.5-coder-7b": "Qwen/Qwen2.5-Coder-7B",
    "qwen2.5-coder-14b": "Qwen/Qwen2.5-Coder-14B",
    "qwen2.5-coder-32b": "Qwen/Qwen2.5-Coder-32B",
    # qwen3
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
    "qwen3-4b": "Qwen/Qwen3-4B",
    "qwen3-8b": "Qwen/Qwen3-8B",
    "qwen3-14b": "Qwen/Qwen3-14B",
    "qwen3-30b-a3b": "Qwen/Qwen3-30B-A3B",
    "qwen3-32b": "Qwen/Qwen3-32B",
    # qwen3.5
    "qwen3.5-4b": "Qwen/Qwen3.5-4B",
    "qwen3.5-9b": "Qwen/Qwen3.5-9B",
    "qwen3.5-27b": "Qwen/Qwen3.5-27B",
    "qwen3.5-35b-a3b": "Qwen/Qwen3.5-35B-A3B",
    # mlx 4-bit variants (model id contains quant suffix)
    "Qwen3-0.6B-4bit": "Qwen/Qwen3-0.6B",
    "Qwen3-8B-4bit": "Qwen/Qwen3-8B",
    "Qwen3.5-4B-4bit": "Qwen/Qwen3.5-4B",
    "gemma-3-1b-it-4bit": "google/gemma-3-1b",
    # qwq
    "qwq-32b": "Qwen/QwQ-32B",
}

# Speed keys → (metric, label)
# pp512 is the most stable prefill measurement; tg128 is standard TG baseline.
SPEED_METRICS: list[tuple[str, str]] = [
    ("pp512", "tok_s_pp"),
    ("tg128", "tok_s_tg"),
]

QUANT_TO_PRECISION: dict[str, str] = {
    "Q4_K_M": "int4",   # Q4_K_M is a 4-bit GGUF format; normalise to int4
    "4bit": "int4",
    "q4_k_m": "int4",
    "fp16": "fp16",
    "bf16": "bf16",
    "fp32": "fp32",
    "int8": "int8",
    "int4": "int4",
}


def _resolve_model_id(raw_id: str) -> str:
    return MODEL_MAP.get(raw_id, raw_id)


def _resolve_precision(quant: str | None) -> str:
    if not quant:
        return "fp16"
    return QUANT_TO_PRECISION.get(quant, "int4")


def _archive_raw(src: Path, device_id: str, today: str) -> None:
    dest_dir = ROOT / "data" / "raw" / SOURCE_NAME / device_id / today
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if not dest.exists():
        shutil.copy2(src, dest)


def _records_from_file(
    path: Path,
    device_id: str,
    collected_at: datetime,
) -> list[BenchmarkRecord]:
    data = json.loads(path.read_text())
    model_info = data.get("model", {})
    speed = data.get("speed", {})
    runtime_name = data.get("runtime", {}).get("name", "llama.cpp")
    raw_model_id = model_info.get("id", path.stem)
    quant = model_info.get("quant")

    model_id = _resolve_model_id(raw_model_id)
    precision = _resolve_precision(quant)
    framework = runtime_name

    records: list[BenchmarkRecord] = []
    for speed_key, metric in SPEED_METRICS:
        if speed_key not in speed:
            continue
        value = float(speed[speed_key])
        # Use the standard __ separator format so build_db can recover input_size
        # from position 4 and _is_tg_benchmark() / _is_pp_benchmark() work correctly.
        # Format: {model_id}__{device_id}__{precision}__{framework}__{input_size}__bs1
        benchmark_id = f"{model_id}__{device_id}__{precision}__{framework}__{speed_key}__bs1"
        records.append(
            BenchmarkRecord(
                benchmark_id=benchmark_id,
                model_id=model_id,
                device_id=device_id,
                precision=precision,  # type: ignore[arg-type]
                metric=metric,  # type: ignore[arg-type]
                value=value,
                framework=framework,
                source_url=SOURCE_URL,
                source_name=SOURCE_LABEL,
                confidence="measured",
                collected_at=collected_at,
                notes=f"raw_file={path.name}",
            )
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest mac-llm-bench results.")
    parser.add_argument(
        "--results-dir",
        required=True,
        type=Path,
        help="Path to the device results directory (contains gguf/ and/or mlx/ subdirs)",
    )
    parser.add_argument(
        "--device-id",
        required=True,
        help="Willitrun device ID (e.g. apple-m5-32gb)",
    )
    args = parser.parse_args()

    results_dir: Path = args.results_dir.expanduser().resolve()
    device_id: str = args.device_id
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    collected_at = datetime.now(timezone.utc)

    # Collect JSON files from gguf/ and mlx/ subdirs (or the dir itself)
    json_files: list[Path] = []
    for subdir in ("gguf", "mlx"):
        candidate = results_dir / subdir
        if candidate.is_dir():
            json_files.extend(sorted(candidate.glob("*.json")))
    if not json_files:
        json_files = sorted(results_dir.glob("*.json"))

    if not json_files:
        print(f"No JSON files found under {results_dir}", file=sys.stderr)
        sys.exit(1)

    # Archive raw files
    for f in json_files:
        _archive_raw(f, device_id, today)
    print(f"[archive] {len(json_files)} files → data/raw/{SOURCE_NAME}/{device_id}/{today}/")

    # Normalise
    all_records: list[BenchmarkRecord] = []
    skipped = 0
    for f in json_files:
        try:
            records = _records_from_file(f, device_id, collected_at)
            all_records.extend(records)
        except Exception as exc:
            print(f"  [skip] {f.name}: {exc}", file=sys.stderr)
            skipped += 1

    # Deduplicate by benchmark_id (last writer wins within this run)
    seen: dict[str, BenchmarkRecord] = {}
    for r in all_records:
        seen[r.benchmark_id] = r
    deduped = list(seen.values())

    # Write JSONL
    out_path = ROOT / "data" / "normalized" / f"{SOURCE_NAME}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing IDs to avoid true duplicates across runs
    existing_ids: set[str] = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    existing_ids.add(json.loads(line)["benchmark_id"])
                except Exception:
                    pass

    new_records = [r for r in deduped if r.benchmark_id not in existing_ids]
    updated_records = [r for r in deduped if r.benchmark_id in existing_ids]

    # Rewrite file: keep old lines that aren't being updated, append new/updated
    update_ids = {r.benchmark_id for r in updated_records}
    if update_ids and out_path.exists():
        kept_lines = [
            line
            for line in out_path.read_text().splitlines()
            if line.strip() and json.loads(line)["benchmark_id"] not in update_ids
        ]
        out_path.write_text("\n".join(kept_lines) + "\n" if kept_lines else "")

    with out_path.open("a") as fh:
        for r in new_records + updated_records:
            fh.write(r.model_dump_json() + "\n")

    total = len(new_records) + len(updated_records)
    print(
        f"[normalize] {total} records written to {out_path.relative_to(ROOT)} "
        f"({len(new_records)} new, {len(updated_records)} updated, {skipped} files skipped)"
    )


if __name__ == "__main__":
    main()
