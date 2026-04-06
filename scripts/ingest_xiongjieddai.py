#!/usr/bin/env python3
"""Ingest benchmark data from XiongjieDai/GPU-Benchmarks-on-LLM-Inference.

Source:
  https://github.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference

The README contains a summary markdown table with columns:
  GPU | 8B Q4_K_M | 8B F16 | 70B Q4_K_M | 70B F16

Values are average text-generation tokens/second at 1024-token context (tg1024).
"OOM" entries are skipped. Multi-GPU rows (* 2, * 4, etc.) are skipped.
Bold markers (**...**) are stripped.

Usage:
    python scripts/ingest_xiongjieddai.py [--from-cache]

Requires: pip install requests
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_ROOT / "raw" / "xiongjieddai"
NORMALIZED_DIR = DATA_ROOT / "normalized"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from willitrun.schemas import make_benchmark_id

README_URL = "https://raw.githubusercontent.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference/main/README.md"
SOURCE_URL = "https://github.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference"

# ── Column header → (model_id, precision) ─────────────────────────────────────
COLUMN_MAP = {
    "8b q4_k_m": ("meta-llama/Llama-3-8B",  "4bit"),
    "8b f16":    ("meta-llama/Llama-3-8B",  "fp16"),
    "70b q4_k_m":("meta-llama/Llama-3-70B", "4bit"),
    "70b f16":   ("meta-llama/Llama-3-70B", "fp16"),
}

# ── GPU name → device ID ──────────────────────────────────────────────────────
# Single-GPU rows only. Keys are lowercased + stripped.
GPU_MAP = {
    "3070 8gb":                   "rtx-3070-8gb",
    "3080 10gb":                  "rtx-3080-10gb",
    "3080 ti 12gb":               "rtx-3080ti-12gb",
    "4070 ti 12gb":               "rtx-4070ti-12gb",
    "4080 16gb":                  "rtx-4080-16gb",
    "rtx 4000 ada 20gb":          "rtx-4000ada-20gb",
    "3090 24gb":                  "rtx-3090-24gb",
    "4090 24gb":                  "rtx-4090-24gb",
    "rtx 5000 ada 32gb":          "rtx-5000ada-32gb",
    "rtx a6000 48gb":             "a6000-48gb",
    "rtx 6000 ada 48gb":          "rtx-6000ada-48gb",
    "a40 48gb":                   "a40-48gb",
    "l40s 48gb":                  "l40s-48gb",
    "a100 pcie 80gb":             "a100-80gb",
    "a100 sxm 80gb":              "a100-sxm-80gb",
    "h100 pcie 80gb":             "h100-80gb",
    "m1 7‑core gpu 8gb":          "apple-m1-8gb",
    "m1 7-core gpu 8gb":          "apple-m1-8gb",
    "m1 max 32‑core gpu 64gb":    "apple-m1-max-64gb",
    "m1 max 32-core gpu 64gb":    "apple-m1-max-64gb",
    "m2 ultra 76-core gpu 192gb": "apple-m2-ultra-192gb",
    "m3 max 40‑core gpu 64gb":    "apple-m3-max-64gb",
    "m3 max 40-core gpu 64gb":    "apple-m3-max-64gb",
}

# Non-breaking hyphen → regular hyphen normalisation
def _normalize_gpu(raw: str) -> str:
    return raw.replace("\u2011", "-").replace("\u2010", "-").lower().strip()


def fetch_and_save_raw() -> str | None:
    try:
        import requests
    except ImportError:
        print("ERROR: requests not installed. Run: pip install requests")
        return None

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    print(f"Fetching {README_URL}...")
    resp = requests.get(README_URL, timeout=30, headers={
        "User-Agent": "willitrun-benchmark-ingest/0.1",
    })
    resp.raise_for_status()

    raw_path = RAW_DIR / f"README_{timestamp}.md"
    raw_path.write_text(resp.text)
    print(f"  Saved raw to {raw_path}")
    return resp.text


def load_cached_raw() -> str | None:
    if not RAW_DIR.exists():
        return None
    files = sorted(RAW_DIR.glob("README_*.md"), reverse=True)
    if not files:
        return None
    print(f"Loading cached: {files[0]}")
    return files[0].read_text()


def _strip_bold(s: str) -> str:
    """Remove markdown bold markers (**...**)."""
    return s.replace("**", "").strip()


def parse_readme(text: str) -> list[dict]:
    """Parse the summary benchmark table from the README markdown.

    Identifies the table by looking for a header row containing 'GPU' as the
    first column and 'Q4_K_M' or 'F16' in subsequent columns.
    Skips multi-GPU rows containing '*'.
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    records = []
    seen_bids: set[str] = set()

    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Detect header row: | GPU | ... Q4_K_M ... |
        if not (line.startswith("|") and "gpu" in line.lower()):
            i += 1
            continue
        if "q4_k_m" not in line.lower() and "f16" not in line.lower():
            i += 1
            continue

        # Parse column headers
        raw_headers = [h.strip() for h in line.split("|")[1:-1]]
        headers_lower = [h.lower() for h in raw_headers]

        # Skip separator row
        if i + 1 < len(lines) and "---" in lines[i + 1]:
            i += 2
        else:
            i += 1
            continue

        # Find which columns map to our model+precision combos
        col_models: dict[int, tuple[str, str]] = {}
        for col_idx, h in enumerate(headers_lower):
            key = _strip_bold(h).strip()
            if key in COLUMN_MAP:
                col_models[col_idx] = COLUMN_MAP[key]

        if not col_models:
            print(f"  WARNING: No recognized model columns in: {raw_headers}")
            continue

        print(f"  Found table with headers: {raw_headers}")
        print(f"  Recognized columns: {col_models}")

        row_count = 0
        # Parse data rows
        while i < len(lines):
            row_line = lines[i].strip()
            if not row_line.startswith("|"):
                break
            i += 1

            cells = [_strip_bold(c.strip()) for c in row_line.split("|")[1:-1]]
            if len(cells) < 2:
                continue

            gpu_raw = cells[0]
            gpu_norm = _normalize_gpu(gpu_raw)

            # Skip multi-GPU rows
            if "*" in gpu_raw:
                continue

            device_id = GPU_MAP.get(gpu_norm)
            if device_id is None:
                print(f"    SKIP unknown GPU: '{gpu_raw}' (normalized: '{gpu_norm}')")
                continue

            for col_idx, (model_id, precision) in col_models.items():
                if col_idx >= len(cells):
                    continue
                val_raw = cells[col_idx]
                if not val_raw or val_raw.upper() == "OOM" or val_raw == "—":
                    continue

                try:
                    value = float(val_raw)
                except ValueError:
                    continue
                if value <= 0:
                    continue

                bid = make_benchmark_id(
                    model_id, device_id, precision, "llama.cpp", "tg1024", 1,
                )

                if bid in seen_bids:
                    continue
                seen_bids.add(bid)

                records.append({
                    "benchmark_id": bid,
                    "model": model_id,
                    "device": device_id,
                    "task": "llm",
                    "precision": precision,
                    "framework": "llama.cpp",
                    "metric": "tok/s",
                    "value": value,
                    "unit": "tokens/sec",
                    "input_size": "tg1024",
                    "batch_size": 1,
                    "llm_setup": {
                        "context_length": 1024,
                        "quantization_format": "GGUF Q4_K_M" if precision == "4bit" else "GGUF F16",
                    },
                    "source": {
                        "name": "XiongjieDai GPU-Benchmarks-on-LLM-Inference",
                        "type": "community_repo",
                        "url": SOURCE_URL,
                        "confidence": "medium",
                    },
                    "provenance": {
                        "collected_at": timestamp,
                        "adapter": "ingest_xiongjieddai.py",
                        "parser_version": "0.1.0",
                    },
                    "notes": (
                        f"tg1024 average; llama.cpp single-GPU; "
                        f"{precision.upper()} quantization"
                        + (" (Q4_K_M)" if precision == "4bit" else "")
                    ),
                })
                row_count += 1
                print(f"    {model_id} {precision} on {device_id}: {value:.2f} tok/s")

        print(f"  → {row_count} records from this table")
        break  # Only parse the first matching table

    return records


def save_normalized(records: list[dict]):
    NORMALIZED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = NORMALIZED_DIR / "xiongjieddai.jsonl"

    with open(output_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    print(f"\nSaved {len(records)} records to {output_path}")
    devices = set(r["device"] for r in records)
    models = set(r["model"] for r in records)
    print(f"  Devices: {sorted(devices)}")
    print(f"  Models: {sorted(models)}")


def main():
    parser = argparse.ArgumentParser(
        description="Ingest XiongjieDai GPU-Benchmarks-on-LLM-Inference"
    )
    parser.add_argument("--from-cache", action="store_true",
                        help="Use cached README instead of fetching")
    args = parser.parse_args()

    if args.from_cache:
        text = load_cached_raw()
        if text is None:
            print("No cache found. Run without --from-cache first.")
            sys.exit(1)
    else:
        text = fetch_and_save_raw()
        if text is None:
            sys.exit(1)

    records = parse_readme(text)
    print(f"\nTotal: {len(records)} benchmark records")
    if records:
        save_normalized(records)
    else:
        print("No records parsed. Check README structure or run without --from-cache.")


if __name__ == "__main__":
    main()
