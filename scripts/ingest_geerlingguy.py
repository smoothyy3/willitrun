#!/usr/bin/env python3
"""Ingest benchmark data from geerlingguy/ai-benchmarks.

Source:  https://github.com/geerlingguy/ai-benchmarks
License: MIT

Fetches the README.md, parses each model's markdown table, extracts the
GPU name from the System column, and emits one JSONL record per
(device, model) pair.

Table format:
  | System                              | CPU/GPU | Eval Rate     | Power (Peak) |
  | [PC Name (GPU Model)](issue_url)   | GPU     | 92.49 Tokens/s| 464.2W       |

Models covered (as of 2026-04):
  DeepSeek R1 14b  → deepseek-ai/DeepSeek-R1-Distill-Qwen-14B  (Ollama Q4_K_M)
  DeepSeek R1 671b → deepseek-ai/DeepSeek-R1-685B               (Ollama Q4_K_M)
  Llama 3.2:3b     → meta-llama/Llama-3.2-3B                    (Ollama Q4_K_M)
  Llama 3.1:70b    → meta-llama/Llama-3-70B                     (Ollama Q4_K_M)
  gpt-oss 20b      → skipped (not in models.yaml)
  Llama 3.1:405b   → skipped (not in models.yaml)

Precision note: geerlingguy uses Ollama which defaults to Q4_K_M for all
models listed above. This is reflected in the precision field.

Usage:
    python scripts/ingest_geerlingguy.py [--from-cache]

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
RAW_DIR   = DATA_ROOT / "raw" / "geerlingguy"
NORM_FILE = DATA_ROOT / "normalized" / "geerlingguy.jsonl"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from willitrun.pipeline.schema import make_benchmark_id
from gpu_map import resolve_apple_chip, resolve_gpu

README_URL  = "https://raw.githubusercontent.com/geerlingguy/ai-benchmarks/master/README.md"
SOURCE_URL  = "https://github.com/geerlingguy/ai-benchmarks"
SOURCE_NAME = "geerlingguy"

# ── Heading text → (model_id, precision) ──────────────────────────────────────
# Keys are lowercased headings as they appear in the README (stripped of #).
# Precision is q4_k_m (Ollama default) unless specified otherwise.
# Set model_id to None to skip a model entirely.
MODEL_MAP: dict[str, tuple[str | None, str]] = {
    "deepseek r1 14b":  ("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", "q4_k_m"),
    "deepseek r1 671b": ("deepseek-ai/DeepSeek-R1-685B",              "q4_k_m"),
    "llama 3.2:3b":     ("meta-llama/Llama-3.2-3B",                   "q4_k_m"),
    "llama 3.1:70b":    ("meta-llama/Llama-3-70B",                    "q4_k_m"),
    # Skip models not yet in the catalog
    "gpt-oss 20b":      (None, ""),
    "llama 3.1:405b":   (None, ""),
}


def fetch_readme(from_cache: bool) -> str:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cached = RAW_DIR / "README.md"

    if from_cache and cached.exists():
        print(f"[geerlingguy] Using cached README: {cached}")
        return cached.read_text()

    import requests
    print(f"[geerlingguy] Fetching {README_URL}")
    r = requests.get(README_URL, timeout=30)
    r.raise_for_status()
    cached.write_text(r.text)
    print(f"[geerlingguy] Cached README ({len(r.text):,} chars)")
    return r.text


def _strip_html(text: str) -> str:
    """Remove HTML tags and their inline content (e.g. <sup>1</sup> footnote markers)."""
    # Remove footnote-style tags with their numeric content first (e.g. <sup>1</sup> → "")
    text = re.sub(r"<sup>\d+</sup>", "", text)
    # Remove any remaining tags (keep content for other tags)
    return re.sub(r"<[^>]+>", "", text).strip()


def _extract_link_text(cell: str) -> str:
    """Return display text of a markdown link, or the raw cell text."""
    m = re.search(r"\[([^\]]+)\]", cell)
    text = m.group(1) if m else cell.strip()
    return _strip_html(text)


def _is_apple(name: str) -> bool:
    n = name.lower()
    return any(f"m{i}" in n for i in range(1, 6)) or "mac" in n or "apple" in n


def _resolve_system(system_cell: str) -> str | None:
    """Extract GPU/chip from a geerlingguy System cell and return device_id.

    System cells look like one of:
      [PC Name (Nvidia RTX 4090)](url)        → GPU in parens
      [Mac Studio (M3 Ultra 512GB)](url)      → Apple chip+memory in parens
      [M1 Ultra (64 GPU Core) 128GB](url)     → Apple chip in main text
      [Pi 5 - 16GB](url)                      → no GPU in parens (CPU/SBC)
    """
    link_text = _extract_link_text(system_cell)

    # Attempt 1: extract text in the last set of parens
    paren_matches = re.findall(r"\(([^)]+)\)", link_text)
    for paren in reversed(paren_matches):
        paren = paren.strip()
        # Check for Apple chip in parens (e.g. "M3 Ultra 512GB")
        if _is_apple(paren):
            device_id = resolve_apple_chip(paren)
            if device_id:
                return device_id
        # Try GPU resolution before applying the digit-prefix heuristic —
        # short aliases like "395+" start with a digit but are valid GPU keys.
        device_id = resolve_gpu(paren)
        if device_id:
            return device_id
        # Skip non-GPU parens that start with a digit: "64 GPU Core", "8 core",
        # RAM sizes ("128GB"), etc.  We already tried resolve_gpu above, so a
        # None result here is a reliable signal to skip.
        if re.match(r"^\d+", paren):
            continue

    # Attempt 2: Apple chip in the main link text (e.g. "M1 Ultra (64 GPU Core)")
    # Strip parens from link text and try the remainder
    bare = re.sub(r"\([^)]*\)", "", link_text).strip()
    if _is_apple(bare):
        return resolve_apple_chip(bare)

    # Attempt 3: GPU name somewhere in the full bare text
    return resolve_gpu(bare)


def _parse_eval_rate(cell: str) -> float | None:
    """Parse "92.49 Tokens/s" → 92.49."""
    m = re.search(r"([\d.]+)\s*[Tt]okens?/s", cell)
    return float(m.group(1)) if m else None


def parse_tables(readme: str) -> list[dict]:
    """Parse all model tables from the README and return raw row dicts."""
    rows = []
    current_model: str | None = None

    for line in readme.splitlines():
        # Detect model headings: "## DeepSeek R1 14b" or "### DeepSeek R1 14b"
        h = re.match(r"^#{2,4}\s+(.+)", line)
        if h:
            heading = h.group(1).strip().lower()
            # Only track headings that are in our MODEL_MAP
            if heading in MODEL_MAP:
                current_model = heading
            else:
                # Non-model heading resets context
                current_model = None
            continue

        if current_model is None:
            continue

        # Parse table data rows (skip header and separator)
        if not line.startswith("|") or re.match(r"^\|[\s\-:]+\|", line):
            continue

        cells = [c.strip() for c in line.split("|")[1:-1]]
        if len(cells) < 3:
            continue

        system_cell, mode_cell, eval_cell = cells[0], cells[1], cells[2]
        rows.append({
            "model_heading": current_model,
            "system_cell":   system_cell,
            "mode":          mode_cell.strip().upper(),
            "eval_rate_raw": eval_cell.strip(),
        })

    return rows


def emit_records(raw_rows: list[dict]) -> list[dict]:
    records = []
    skipped_model = skipped_device = skipped_cpu = skipped_parse = 0
    now = datetime.now(timezone.utc).isoformat()

    for row in raw_rows:
        model_id, precision = MODEL_MAP.get(row["model_heading"], (None, ""))
        if model_id is None:
            skipped_model += 1
            continue

        # Skip CPU-only entries
        if row["mode"] == "CPU":
            skipped_cpu += 1
            continue

        eval_tps = _parse_eval_rate(row["eval_rate_raw"])
        if eval_tps is None:
            skipped_parse += 1
            continue

        device_id = _resolve_system(row["system_cell"])
        if device_id is None:
            skipped_device += 1
            print(f"  [skip] unknown device: {_extract_link_text(row['system_cell'])!r}")
            continue

        bid = make_benchmark_id(SOURCE_NAME, model_id, device_id, precision, "tok_s_tg")
        records.append({
            "benchmark_id": bid,
            "model_id":     model_id,
            "device_id":    device_id,
            "precision":    precision,
            "metric":       "tok_s_tg",
            "value":        round(eval_tps, 2),
            "framework":    "ollama",
            "source_url":   SOURCE_URL,
            "source_name":  SOURCE_NAME,
            "confidence":   "community",
            "collected_at": now,
            "notes":        f"geerlingguy/ai-benchmarks · {_extract_link_text(row['system_cell'])}",
        })

    print(f"[geerlingguy] Skipped: {skipped_model} unknown models, "
          f"{skipped_device} unknown devices, {skipped_cpu} CPU-only, "
          f"{skipped_parse} unparseable eval rates")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest geerlingguy/ai-benchmarks.")
    parser.add_argument("--from-cache", action="store_true",
                        help="Use cached README without re-fetching.")
    args = parser.parse_args()

    readme  = fetch_readme(args.from_cache)
    rows    = parse_tables(readme)
    records = emit_records(rows)

    NORM_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(NORM_FILE, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"[geerlingguy] Wrote {len(records)} records → {NORM_FILE}")


if __name__ == "__main__":
    main()
