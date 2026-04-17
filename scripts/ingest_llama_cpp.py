#!/usr/bin/env python3
"""Ingest benchmark data from llama.cpp community scoreboards.

Sources (all benchmark Llama 2 7B Q4_0 unless noted):
  CUDA:   https://github.com/ggml-org/llama.cpp/discussions/15013
  ROCm:   https://github.com/ggml-org/llama.cpp/discussions/15021
  Vulkan: https://github.com/ggml-org/llama.cpp/discussions/10879
  Apple:  https://github.com/ggml-org/llama.cpp/discussions/4167
          (LLaMA 7B F16/Q8_0/Q4_0; different table format)

Page structure for CUDA/ROCm/Vulkan (as of 2026-04):
  - GitHub Discussion with community-contributed benchmark results
  - Two markdown tables rendered as HTML:
    1. Scoreboard (without Flash Attention)
    2. Scoreboard (with Flash Attention)
  - Model: Llama 2 7B, Q4_0 (3.56 GiB, 6.74B params)
  - Test: llama-bench -m llama-2-7b.Q4_0.gguf -ngl 99 -fa 0,1
  - Columns: Chip | Memory | pp512 t/s | tg128 t/s | Commit | Thanks to
  - Values include ± std dev (e.g. "14073.41 ± 115.16")

Apple M-series table format (discussion #4167):
  - Single summary table, no FA split
  - Columns: Chip | BW [GB/s] | GPU Cores | F16 PP | F16 TG | Q8_0 PP | Q8_0 TG | Q4_0 PP | Q4_0 TG
  - Chip names include GPU core count: "M2 Pro (16 GPU)" → strip to "M2 Pro"
  - Missing values represented as "—"

Usage:
    python scripts/ingest_llama_cpp.py [--from-cache]

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
RAW_DIR = DATA_ROOT / "raw" / "llama_cpp_cuda"
NORMALIZED_DIR = DATA_ROOT / "normalized"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from willitrun.pipeline.schema import make_benchmark_id
from gpu_map import resolve_gpu, resolve_apple_chip

# ---- Scoreboard sources (CUDA / ROCm / Vulkan) ----
# All use the same table format: Chip | Memory | pp512 t/s | tg128 t/s
# All benchmark Llama 2 7B Q4_0 with llama-bench
SCOREBOARDS = [
    {
        "url": "https://github.com/ggml-org/llama.cpp/discussions/15013",
        "cache_name": "cuda_scoreboard",
        "backend_note": "CUDA",
    },
    {
        "url": "https://github.com/ggml-org/llama.cpp/discussions/15021",
        "cache_name": "rocm_scoreboard",
        "backend_note": "ROCm",
    },
    {
        "url": "https://github.com/ggml-org/llama.cpp/discussions/10879",
        "cache_name": "vulkan_scoreboard",
        "backend_note": "Vulkan",
    },
]

APPLE_DISCUSSION_URL = "https://github.com/ggml-org/llama.cpp/discussions/4167"

# GPU chip name normalisation and device_id resolution are provided by the
# shared gpu_map module (imported above).  All ingest scripts use the same
# canonical mapping so that adding a new device to gpu_map.py instantly
# benefits every source.

# Apple quant format → our precision label
APPLE_QUANT_PRECISION: dict[str, str] = {
    "f16": "fp16",
    "q8_0": "int8",
    "q4_0": "4bit",
}



def parse_value_with_stddev(raw: str) -> float | None:
    """Parse a value like '14073.41 ± 115.16' → 14073.41."""
    raw = raw.strip().replace(",", "")
    # Match number, optionally followed by ± stddev
    m = re.match(r"([\d.]+)(?:\s*±\s*[\d.]+)?", raw)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def extract_stddev(raw: str) -> float | None:
    """Extract standard deviation from '14073.41 ± 115.16' → 115.16."""
    raw = raw.strip().replace(",", "")
    m = re.search(r"±\s*([\d.]+)", raw)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def fetch_and_save_raw(url: str, cache_name: str) -> str | None:
    """Fetch URL and save raw HTML snapshot."""
    try:
        import requests
    except ImportError:
        print("ERROR: requests not installed. Run: pip install willitrun[ingestion]")
        return None

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    print(f"Fetching {url}...")
    resp = requests.get(url, timeout=30, headers={
        "User-Agent": "willitrun-benchmark-ingest/0.1",
    })
    resp.raise_for_status()

    raw_path = RAW_DIR / f"{cache_name}_{timestamp}.html"
    raw_path.write_text(resp.text)
    print(f"  Saved raw HTML to {raw_path}")
    return resp.text


def load_cached_raw(cache_name: str) -> str | None:
    """Load most recent cached HTML for the given cache_name prefix."""
    if not RAW_DIR.exists():
        return None
    files = sorted(RAW_DIR.glob(f"{cache_name}_*.html"), reverse=True)
    if not files:
        return None
    print(f"Loading cached: {files[0]}")
    return files[0].read_text()


def parse_scoreboard_tables(html: str, backend_note: str = "CUDA") -> list[dict]:
    """Parse the two scoreboard tables from the GitHub discussion page.

    Returns normalized benchmark records. Each GPU row produces up to 2 records:
    one for pp512 (prompt processing) and one for tg128 (text generation).

    backend_note is appended to notes for traceability (e.g. "CUDA", "ROCm", "Vulkan").
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    records = []
    timestamp = datetime.now(timezone.utc).isoformat()
    unmapped_gpus = set()

    # Find all tables in the discussion body
    # GitHub renders the discussion body inside <div class="markdown-body">
    body = soup.find(class_="markdown-body") or soup
    tables = body.find_all("table")

    print(f"  Found {len(tables)} tables on page")

    for table_idx, table in enumerate(tables):
        # Check if this is a scoreboard table by inspecting headers
        headers = [th.get_text(strip=True).lower() for th in table.find_all("th")]
        if not headers:
            # Try first row as header
            first_row = table.find("tr")
            if first_row:
                headers = [td.get_text(strip=True).lower() for td in first_row.find_all(["th", "td"])]

        # Must have "chip" and "pp512" columns
        has_chip = any("chip" in h for h in headers)
        has_pp = any("pp512" in h or "pp 512" in h for h in headers)
        if not has_chip or not has_pp:
            continue

        # Determine if this is the FA or no-FA table
        # Check preceding text for "with FA" / "without FA" / "no FA"
        # Look at the heading/text above this table
        flash_attention = False
        prev_siblings = list(table.find_all_previous(["h1", "h2", "h3", "h4", "p", "strong"]))
        for el in prev_siblings[:5]:
            text = el.get_text(strip=True).lower()
            if "with fa" in text or "flash attention" in text and "without" not in text and "no fa" not in text:
                flash_attention = True
                break
            if "without fa" in text or "no fa" in text:
                flash_attention = False
                break

        fa_label = "fa" if flash_attention else "nofa"
        print(f"  Table {table_idx}: {'with' if flash_attention else 'without'} Flash Attention")

        # Find column indices
        chip_col = None
        mem_col = None
        pp_col = None
        tg_col = None
        commit_col = None

        for i, h in enumerate(headers):
            if "chip" in h:
                chip_col = i
            elif "memory" in h:
                mem_col = i
            elif "pp512" in h or "pp 512" in h:
                pp_col = i
            elif "tg128" in h or "tg 128" in h:
                tg_col = i
            elif "commit" in h:
                commit_col = i

        if chip_col is None or pp_col is None or tg_col is None:
            print(f"    Skipping: couldn't find required columns (chip={chip_col}, pp={pp_col}, tg={tg_col})")
            continue

        # Parse data rows
        rows = table.find_all("tr")
        row_count = 0
        for row in rows[1:]:  # skip header
            cells = row.find_all(["td", "th"])
            if len(cells) <= max(chip_col, pp_col, tg_col):
                continue

            chip_raw = cells[chip_col].get_text(strip=True)
            pp_raw = cells[pp_col].get_text(strip=True)
            tg_raw = cells[tg_col].get_text(strip=True)
            commit_raw = cells[commit_col].get_text(strip=True) if commit_col and commit_col < len(cells) else None

            # Skip empty or header-like rows
            if not chip_raw or chip_raw.lower() == "chip":
                continue

            # Resolve GPU
            device_id = resolve_gpu(chip_raw)
            if not device_id:
                unmapped_gpus.add(chip_raw)
                continue

            # Parse values
            pp_val = parse_value_with_stddev(pp_raw)
            tg_val = parse_value_with_stddev(tg_raw)
            pp_std = extract_stddev(pp_raw)
            tg_std = extract_stddev(tg_raw)

            # Build provenance
            provenance = {
                "collected_at": timestamp,
                "adapter": "ingest_llama_cpp.py",
                "parser_version": "0.1.0",
            }

            source = {
                "name": f"llama.cpp {backend_note} Scoreboard",
                "type": "github_discussion",
                "url": [s["url"] for s in SCOREBOARDS if s["backend_note"] == backend_note][0],
                "confidence": "high",
            }

            # Emit pp512 record (prompt processing tok/s)
            if pp_val and pp_val > 0:
                bid = make_benchmark_id(
                    "llama_cpp", "llama-2-7b", device_id, "int4", "tok_s_pp",
                ) + f"__{fa_label}"

                notes_parts = [
                    f"Llama 2 7B Q4_0, prompt processing 512 tokens",
                    f"{'with' if flash_attention else 'without'} Flash Attention",
                    f"backend={backend_note}",
                ]
                if pp_std:
                    notes_parts.append(f"stddev=±{pp_std}")
                if commit_raw:
                    notes_parts.append(f"commit={commit_raw}")
                notes_parts.append(f"chip={chip_raw}")

                records.append({
                    "benchmark_id": bid,
                    "model_id": "llama-2-7b",
                    "device_id": device_id,
                    "precision": "int4",
                    "framework": "llama.cpp",
                    "metric": "tok_s_pp",
                    "value": round(pp_val, 2),
                    "source_name": f"llama.cpp {backend_note} Scoreboard",
                    "source_url": source["url"],
                    "confidence": "measured",
                    "collected_at": provenance["collected_at"],
                    "notes": "; ".join(notes_parts),
                })
                row_count += 1

            # Emit tg128 record (text generation tok/s)
            if tg_val and tg_val > 0:
                bid = make_benchmark_id(
                    "llama_cpp", "llama-2-7b", device_id, "int4", "tok_s_tg",
                ) + f"__{fa_label}"

                notes_parts = [
                    f"Llama 2 7B Q4_0, text generation 128 tokens",
                    f"{'with' if flash_attention else 'without'} Flash Attention",
                    f"backend={backend_note}",
                ]
                if tg_std:
                    notes_parts.append(f"stddev=±{tg_std}")
                if commit_raw:
                    notes_parts.append(f"commit={commit_raw}")
                notes_parts.append(f"chip={chip_raw}")

                records.append({
                    "benchmark_id": bid,
                    "model_id": "llama-2-7b",
                    "device_id": device_id,
                    "precision": "int4",
                    "framework": "llama.cpp",
                    "metric": "tok_s_tg",
                    "value": round(tg_val, 2),
                    "source_name": f"llama.cpp {backend_note} Scoreboard",
                    "source_url": source["url"],
                    "confidence": "measured",
                    "collected_at": provenance["collected_at"],
                    "notes": "; ".join(notes_parts),
                })
                row_count += 1

        print(f"    → {row_count} records")

    if unmapped_gpus:
        print(f"\n  WARNING: {len(unmapped_gpus)} unmapped GPU(s) — add to gpu_map.py + devices.yaml:")
        for gpu in sorted(unmapped_gpus):
            print(f"    - {gpu!r}")

    return records


def parse_apple_scoreboard(html: str) -> list[dict]:
    """Parse the Apple M-series llama.cpp scoreboard (discussion #4167).

    Table format:
      Chip | BW [GB/s] | GPU Cores | F16 PP | F16 TG | Q8_0 PP | Q8_0 TG | Q4_0 PP | Q4_0 TG

    Chip names include GPU core count in parens: "M2 Pro (16 GPU)" → strip to "M2 Pro".
    Missing values are "—". Values may include "± stddev" notation.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    records = []
    timestamp = datetime.now(timezone.utc).isoformat()
    unmapped_chips: set[str] = set()

    body = soup.find(class_="markdown-body") or soup
    tables = body.find_all("table")
    print(f"  Found {len(tables)} tables in Apple discussion")

    source = {
        "name": "llama.cpp Apple Silicon Scoreboard",
        "type": "github_discussion",
        "url": APPLE_DISCUSSION_URL,
        "confidence": "high",
    }
    provenance = {
        "collected_at": timestamp,
        "adapter": "ingest_llama_cpp.py",
        "parser_version": "0.2.0",
    }

    for table in tables:
        headers = [th.get_text(strip=True).lower() for th in table.find_all("th")]
        if not headers:
            first_row = table.find("tr")
            if first_row:
                headers = [td.get_text(strip=True).lower() for td in first_row.find_all(["th", "td"])]

        # Must have at least one quant TG column — chip column may have empty header
        has_tg = any("tg" in h for h in headers)
        has_quant = any(q in h for h in headers for q in ("f16", "q8_0", "q4_0"))
        if not has_tg or not has_quant:
            continue

        print(f"  Apple table headers: {headers}")

        # Detect column indices for each quant×metric combination
        # Chip column: first column (header may be empty in GitHub rendering)
        chip_col = 0
        col_map: dict[tuple[str, str], int] = {}

        for i, h in enumerate(headers):
            for quant in ("f16", "q8_0", "q4_0"):
                if quant in h:
                    if " pp" in h or "pp[" in h:
                        col_map[(quant, "pp")] = i
                    elif " tg" in h or "tg[" in h:
                        col_map[(quant, "tg")] = i

        if not col_map:
            print(f"  Skipping: couldn't identify quant columns")
            continue

        # best_vals[device_id][(quant, pp_or_tg)] = best value seen so far
        # Multiple rows per chip (e.g. M1 7-core vs 8-core) → keep highest TG value
        best_vals: dict[str, dict[tuple[str, str], float]] = {}

        rows = table.find_all("tr")[1:]
        for row in rows:
            cells = row.find_all(["td", "th"])
            if not cells or len(cells) <= chip_col:
                continue

            chip_raw = cells[chip_col].get_text(strip=True)
            if not chip_raw:
                continue

            # Strip leading emoji/symbols, trailing footnote digits
            chip_clean = re.sub(r"[^\w\s]", " ", chip_raw).strip().lower()
            chip_clean = re.sub(r"\s+", " ", chip_clean).strip()

            # Resolve via shared gpu_map module (handles normalisation + lookup)
            device_id = resolve_apple_chip(chip_raw)

            if device_id is None:
                unmapped_chips.add(chip_raw)
                continue

            if device_id not in best_vals:
                best_vals[device_id] = {}

            for (quant, pp_or_tg), col_idx in col_map.items():
                if col_idx >= len(cells):
                    continue
                raw_val = cells[col_idx].get_text(strip=True)
                if raw_val in ("—", "-", "", "N/A"):
                    continue
                val = parse_value_with_stddev(raw_val)
                if val is None or val <= 0:
                    continue
                key = (quant, pp_or_tg)
                if key not in best_vals[device_id] or val > best_vals[device_id][key]:
                    best_vals[device_id][key] = val

        row_count = 0
        for device_id, quant_vals in best_vals.items():
            for (quant, pp_or_tg), val in quant_vals.items():
                precision = APPLE_QUANT_PRECISION.get(quant, quant)
                metric_val = "tok_s_pp" if pp_or_tg == "pp" else "tok_s_tg"

                bid = make_benchmark_id(
                    "llama_cpp", "llama-2-7b", device_id, precision, metric_val,
                ) + "__metal"

                records.append({
                    "benchmark_id": bid,
                    "model_id": "llama-2-7b",
                    "device_id": device_id,
                    "precision": precision,
                    "framework": "llama.cpp",
                    "metric": metric_val,
                    "value": round(val, 2),
                    "source_name": source["name"],
                    "source_url": source["url"],
                    "confidence": "measured",
                    "collected_at": provenance["collected_at"],
                    "notes": (
                        f"LLaMA 7B {quant.upper()}, "
                        f"{'prompt processing' if pp_or_tg == 'pp' else 'text generation'}; "
                        f"backend=Metal; device={device_id}"
                    ),
                })
                row_count += 1

        print(f"  → {row_count} records from Apple table")

    if unmapped_chips:
        print(f"\n  WARNING: {len(unmapped_chips)} unmapped Apple chip(s) — add to gpu_map.py + devices.yaml:")
        for chip in sorted(unmapped_chips):
            print(f"    - {chip!r}")

    return records


def save_normalized(records: list[dict]):
    """Save normalized records as JSONL."""
    NORMALIZED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = NORMALIZED_DIR / "llama_cpp_cuda.jsonl"

    with open(output_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    print(f"\nSaved {len(records)} records to {output_path}")

    # Summary
    devices = set(r["device"] for r in records)
    models = set(r["model"] for r in records)
    print(f"  Devices: {sorted(devices)}")
    print(f"  Models: {sorted(models)}")
    print(f"  Records per type:")
    pp = sum(1 for r in records if "pp512" in r["benchmark_id"])
    tg = sum(1 for r in records if "tg128" in r["benchmark_id"])
    fa = sum(1 for r in records if "__fa" in r["benchmark_id"] and "__nofa" not in r["benchmark_id"])
    nofa = sum(1 for r in records if "__nofa" in r["benchmark_id"])
    print(f"    pp512: {pp}, tg128: {tg}")
    print(f"    with FA: {fa}, without FA: {nofa}")


def main():
    parser = argparse.ArgumentParser(
        description="Ingest llama.cpp scoreboards (CUDA, ROCm, Vulkan, Apple Metal)"
    )
    parser.add_argument("--from-cache", action="store_true",
                        help="Use cached HTML instead of fetching")
    args = parser.parse_args()

    all_records: list[dict] = []
    # Track benchmark_ids across sources — first seen wins (CUDA > ROCm > Vulkan)
    seen_ids: set[str] = set()

    # --- CUDA / ROCm / Vulkan scoreboards (same table format) ---
    for scoreboard in SCOREBOARDS:
        print(f"\n=== {scoreboard['backend_note']} scoreboard ===")
        if args.from_cache:
            html = load_cached_raw(scoreboard["cache_name"])
            if html is None:
                print(f"  No cache found. Skipping {scoreboard['backend_note']}.")
                continue
        else:
            html = fetch_and_save_raw(scoreboard["url"], scoreboard["cache_name"])
            if html is None:
                continue

        records = parse_scoreboard_tables(html, backend_note=scoreboard["backend_note"])
        # Dedup: skip if same device+model was already seen from a higher-priority source
        new_records = []
        for r in records:
            if r["benchmark_id"] not in seen_ids:
                seen_ids.add(r["benchmark_id"])
                new_records.append(r)
        print(f"  {len(records)} parsed, {len(new_records)} new (after dedup)")
        all_records.extend(new_records)

    # --- Apple M-series scoreboard (different table format) ---
    print(f"\n=== Apple Metal scoreboard ===")
    if args.from_cache:
        html = load_cached_raw("apple_scoreboard")
        if html is None:
            print("  No Apple cache found. Skipping.")
        else:
            apple_records = parse_apple_scoreboard(html)
            for r in apple_records:
                if r["benchmark_id"] not in seen_ids:
                    seen_ids.add(r["benchmark_id"])
                    all_records.append(r)
            print(f"  {len(apple_records)} Apple records parsed")
    else:
        html = fetch_and_save_raw(APPLE_DISCUSSION_URL, "apple_scoreboard")
        if html:
            apple_records = parse_apple_scoreboard(html)
            for r in apple_records:
                if r["benchmark_id"] not in seen_ids:
                    seen_ids.add(r["benchmark_id"])
                    all_records.append(r)
            print(f"  {len(apple_records)} Apple records parsed")

    print(f"\nTotal: {len(all_records)} benchmark records")
    if all_records:
        save_normalized(all_records)
    else:
        print("No records parsed. The page structure may have changed.")
        print("Inspect the raw HTML and update the parser.")


if __name__ == "__main__":
    main()
