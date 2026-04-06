#!/usr/bin/env python3
"""Ingest benchmark data from NVIDIA Jetson developer page.

Source: https://developer.nvidia.com/embedded/jetson-benchmarks

Page structure (as of 2026-04):
  Table 0: LLM/VLM results on AGX Thor
    - Sub-header row with "Max Concurrency 1 (tok/s)" and "Max Concurrency 8 (tok/s)"
    - Model names like "Llama 3.1 8B" in cells
  Table 1: MLPerf v4.0 on AGX Orin (GPT-J, SDXL)
  Table 2: MLPerf v3.1 on AGX Orin + Orin NX (ResNet, Retinanet, BERT, etc.)
  Table 3: MLPerf v3.0 on AGX Orin + Orin NX

Model names have task prefixes merged: "Image ClassificationResNet" → "ResNet"

Usage:
    python scripts/ingest_nvidia_jetson.py [--from-cache]

Requires: pip install willitrun[ingestion]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_ROOT / "raw" / "nvidia_jetson"
NORMALIZED_DIR = DATA_ROOT / "normalized"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from willitrun.pipeline.schema import make_benchmark_id

NVIDIA_URL = "https://developer.nvidia.com/embedded/jetson-benchmarks"

# Task prefixes NVIDIA concatenates with model name (no space)
TASK_PREFIXES = [
    "LLM Summarization",
    "LLM",
    "VLM",
    "Image Generation",
    "Image Classification",
    "Object Detection",
    "Speech Recognition",
    "Language Processing",
    "NLP",
    "Medical Imaging",
    "Recommender",
]

MODEL_MAP = {
    "resnet": "resnet50",
    "resnet-50": "resnet50",
    "retinanet": "retinanet",
    "bert": "bert-base",
    "3d-unet": "3d-unet",
    "rnnt": "rnnt",
    "rnn-t": "rnnt",
    "gpt-j 6b": "gpt-j-6b",
    "gpt-j": "gpt-j-6b",
    "stable-diffusion-xl": "sdxl",
    "dlrm": "dlrm",
    "llama 3.1 8b": "meta-llama/Llama-3.1-8B",
    "llama 3.3 70b": "meta-llama/Llama-3.3-70B",
    "qwen 3 30b-a3b": "Qwen/Qwen3-30B-A3B",
    "deepseek r1 7b": "deepseek-ai/DeepSeek-R1-7B",
    "qwen2.5-vl 3b": "Qwen/Qwen2.5-VL-3B",
    "llama 3.2 11b vision": "meta-llama/Llama-3.2-11B-Vision",
}

TASK_MAP = {
    "resnet50": "classification",
    "retinanet": "detection",
    "bert-base": "llm",
    "rnnt": "audio",
    "3d-unet": "segmentation",
    "gpt-j-6b": "llm",
    "sdxl": "image_generation",
    "dlrm": "other",
}


def strip_task_prefix(raw_name: str) -> str:
    """Remove concatenated task prefix from model name."""
    for prefix in TASK_PREFIXES:
        if raw_name.startswith(prefix):
            return raw_name[len(prefix):].strip()
    return raw_name.strip()


def resolve_model(raw: str) -> str | None:
    """Resolve raw model text to canonical ID."""
    cleaned = strip_task_prefix(raw).strip()
    cleaned_lower = cleaned.lower()
    for pattern, model_id in MODEL_MAP.items():
        if pattern == cleaned_lower:
            return model_id
    for pattern, model_id in MODEL_MAP.items():
        if pattern in cleaned_lower:
            return model_id
    return None


def fetch_and_save_raw(url: str) -> str | None:
    try:
        import requests
    except ImportError:
        print("ERROR: requests not installed. Run: pip install willitrun[ingestion]")
        return None

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    print(f"Fetching {url}...")
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        print(f"  ERROR: {e}")
        return None

    raw_path = RAW_DIR / f"jetson_benchmarks_{timestamp}.html"
    raw_path.write_text(resp.text)
    print(f"  Saved raw HTML to {raw_path}")
    return resp.text


def load_cached_raw() -> str | None:
    if not RAW_DIR.exists():
        return None
    files = sorted(RAW_DIR.glob("jetson_benchmarks_*.html"), reverse=True)
    if not files:
        return None
    print(f"Loading cached: {files[0]}")
    return files[0].read_text()


def parse_table_0_llm(table, timestamp: str) -> list[dict]:
    """Parse Table 0: LLM/VLM benchmarks on AGX Thor.

    Structure:
      Header: ['', 'Model', 'NVIDIA Jetson AGX Thor']
      Sub-header: ['', '', 'Max Concurrency 1(tokens/sec)', 'Max Concurrency 8(tokens/sec)']
      Data: ['LLM', 'Llama 3.1 8B', '41.3', '150.8']
      Data: ['Llama 3.3 70B', '4.7', '12.6']  (task cell spans multiple rows)
    """
    records = []
    device_id = "jetson-agx-thor"
    rows = table.find_all("tr")

    for row in rows:
        cells = [td.get_text(strip=True) for td in row.find_all("td")]
        if len(cells) < 3:
            continue

        # Find model name — skip task-only cells and sub-header cells
        model_raw = None
        values = []

        for cell in cells:
            if cell in ("", "LLM", "VLM"):
                continue
            if "concurrency" in cell.lower() or "tokens" in cell.lower():
                continue
            if model_raw is None:
                model_raw = cell
            else:
                try:
                    values.append(float(cell.replace(",", "")))
                except (ValueError, TypeError):
                    pass

        if model_raw is None:
            continue

        model = resolve_model(model_raw)
        if not model:
            continue

        # First value = concurrency 1, second = concurrency 8
        for i, value in enumerate(values):
            if value <= 0:
                continue
            concurrency = 1 if i == 0 else 8

            bid = make_benchmark_id(
                model, device_id, "fp16", "vllm",
                context_length=2048,
            ) + f"__conc{concurrency}"

            records.append({
                "benchmark_id": bid,
                "model": model,
                "device": device_id,
                "task": "llm",
                "precision": "fp16",
                "framework": "vllm",
                "metric": "tok/s",
                "value": round(value, 1),
                "unit": "tokens/sec",
                "batch_size": concurrency,
                "llm_setup": {
                    "context_length": 2048,
                },
                "source": {
                    "name": "NVIDIA Jetson benchmarks",
                    "type": "official_docs",
                    "url": NVIDIA_URL,
                    "confidence": "high",
                },
                "provenance": {
                    "collected_at": timestamp,
                    "adapter": "ingest_nvidia_jetson.py",
                    "parser_version": "0.2.0",
                },
                "notes": f"Concurrency={concurrency}",
            })

    return records


def parse_mlperf_table(table, timestamp: str, seen_ids: set) -> list[dict]:
    """Parse one MLPerf results table (Tables 1-3 on the NVIDIA Jetson page).

    Rules applied:
    - Only Single Stream (SS) records emitted — Offline is server-batch throughput,
      not meaningful for "can one user run this?" decisions.
    - Only AGX Orin — the page only publishes Offline (not SS) for Orin NX,
      so there is no valid single-user data for that device here.
    - GPT-J 6B and SDXL are skipped — their MLPerf metric (samples/sec of completions)
      cannot be reliably converted to tok/s for user display.
    - Older MLPerf versions (Table 3 uses Samples/s for SS, Table 2 uses Latency ms)
      are handled via sub-header detection; seen_ids prevents duplicate benchmark_ids
      from multiple MLPerf rounds in the same page.
    """
    # Models to skip from MLPerf (poor metric fit for user-facing estimation)
    SKIP_MODELS = {"gpt-j-6b", "sdxl"}

    rows = table.find_all("tr")

    # ── Step 1: confirm AGX Orin is covered and detect SS column format ─────
    agx_orin_present = False
    ss_is_latency: bool | None = None  # True = ms latency, False = direct samples/s

    for row in rows:
        all_text = [c.get_text(strip=True) for c in row.find_all(["th", "td"])]
        for text in all_text:
            tl = text.lower()
            if "agx orin" in tl and "maxq" not in tl:
                agx_orin_present = True
            if "single stream" in tl:
                ss_is_latency = "latency" in tl  # True for v3.1, False for v3.0

    if not agx_orin_present or ss_is_latency is None:
        return []  # Table doesn't cover AGX Orin or has no SS column

    # ── Step 2: parse data rows ──────────────────────────────────────────────
    records = []
    for row in rows:
        cells = [td.get_text(strip=True) for td in row.find_all("td")]
        if len(cells) < 2:
            continue

        # Skip sub-header rows (they contain metric/scenario names, not numbers)
        first = cells[0].lower()
        if any(kw in first for kw in ("single stream", "offline", "latency", "multi stream")):
            continue
        if not cells[0]:
            continue

        model = resolve_model(cells[0])
        if not model or model in SKIP_MODELS:
            continue

        task = TASK_MAP.get(model, "other")
        bid = make_benchmark_id(model, "jetson-agx-orin-64gb", "int8", "tensorrt") + "__ss"

        # Skip if we already have this benchmark from a newer MLPerf round
        if bid in seen_ids:
            continue

        # AGX Orin SS is always the first data column (index 1 after model name)
        try:
            ss_raw = float(cells[1].replace(",", ""))
        except (ValueError, IndexError):
            continue

        if ss_raw <= 0:
            continue

        if ss_is_latency:
            fps = round(1000.0 / ss_raw, 2)
            note = f"MLPerf Single Stream, latency={ss_raw}ms"
        else:
            fps = round(ss_raw, 2)
            note = f"MLPerf Single Stream, {ss_raw} samples/s"

        seen_ids.add(bid)
        records.append({
            "benchmark_id": bid,
            "model": model,
            "device": "jetson-agx-orin-64gb",
            "task": task,
            "precision": "int8",
            "framework": "tensorrt",
            "metric": "fps",
            "value": fps,
            "unit": "frames/sec",
            "batch_size": 1,
            "source": {
                "name": "NVIDIA Jetson benchmarks (MLPerf)",
                "type": "mlperf",
                "url": NVIDIA_URL,
                "confidence": "high",
            },
            "provenance": {
                "collected_at": timestamp,
                "adapter": "ingest_nvidia_jetson.py",
                "parser_version": "0.3.0",
            },
            "notes": note,
        })

    return records


def parse_all(html: str) -> list[dict]:
    """Parse all benchmark tables from NVIDIA page."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    records = []
    timestamp = datetime.now(timezone.utc).isoformat()

    tables = soup.find_all("table")
    print(f"  Found {len(tables)} tables")

    if len(tables) == 0:
        return []

    # Table 0: LLM/VLM on AGX Thor
    print("  Parsing Table 0 (LLM/VLM AGX Thor)...")
    llm_records = parse_table_0_llm(tables[0], timestamp)
    print(f"    → {len(llm_records)} records")
    records.extend(llm_records)

    # Tables 1+: MLPerf results (multiple versions on the same page).
    # seen_ids is shared across tables so that when the same benchmark_id
    # appears in both a newer (Table 1/2) and older (Table 3) MLPerf round,
    # the first (newer) version wins and the duplicate is silently dropped.
    mlperf_seen: set[str] = set()
    for i in range(1, len(tables)):
        print(f"  Parsing Table {i} (MLPerf)...")
        mlperf_records = parse_mlperf_table(tables[i], timestamp, mlperf_seen)
        print(f"    → {len(mlperf_records)} records")
        records.extend(mlperf_records)

    return records


def save_normalized(records: list[dict]):
    NORMALIZED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = NORMALIZED_DIR / "nvidia_jetson.jsonl"

    with open(output_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    print(f"\nSaved {len(records)} records to {output_path}")
    devices = set(r["device"] for r in records)
    models = set(r["model"] for r in records)
    print(f"  Devices: {sorted(devices)}")
    print(f"  Models: {sorted(models)}")


def main():
    parser = argparse.ArgumentParser(description="Ingest NVIDIA Jetson benchmarks")
    parser.add_argument("--from-cache", action="store_true",
                        help="Use cached HTML instead of fetching")
    args = parser.parse_args()

    if args.from_cache:
        html = load_cached_raw()
    else:
        html = fetch_and_save_raw(NVIDIA_URL)

    if html is None:
        print("No HTML available.")
        sys.exit(1)

    records = parse_all(html)
    print(f"\nParsed {len(records)} total records")

    if records:
        save_normalized(records)
    else:
        print("No records parsed. Inspect raw HTML.")


if __name__ == "__main__":
    main()
