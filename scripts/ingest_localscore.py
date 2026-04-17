#!/usr/bin/env python3
"""Ingest benchmark data from LocalScore (github.com/cjpais/LocalScore).

Source:  https://github.com/cjpais/LocalScore
License: Apache 2.0

Two fetch modes (controlled by --mode):

  db (default):   Downloads the dev SQLite DB from the GitHub repo.
                  Fast, offline-capable, but only contains a small fixture
                  dataset (~11 GPU entries). Good for CI / development.

  web (full):     Paginates https://www.localscore.ai/latest?offset=N,
                  extracts the __NEXT_DATA__ JSON from each page, and
                  collects all ~3,300 individual benchmark runs. Aggregates
                  to best avg_gen_tps per (accelerator, model) pair.
                  Respects the TTL cache — skips fetch if cached pages exist.

Models benchmarked (hardcoded in the LocalScore benchmark suite, Q4_K_M):
  • Meta Llama 3.2 1B Instruct
  • Meta Llama 3.1 8B Instruct
  • Qwen2.5 14B Instruct

Usage:
    python scripts/ingest_localscore.py              # fast dev DB mode
    python scripts/ingest_localscore.py --mode web   # full production scrape
    python scripts/ingest_localscore.py --from-cache # use cached data

Requires: pip install willitrun[ingestion]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
RAW_DIR   = DATA_ROOT / "raw" / "localscore"
NORM_FILE = DATA_ROOT / "normalized" / "localscore.jsonl"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from willitrun.pipeline.schema import make_benchmark_id
from gpu_map import resolve_apple_chip, resolve_gpu

# ── GitHub dev DB (try in order until one is a valid SQLite file) ─────────────
DB_URLS = [
    "https://raw.githubusercontent.com/cjpais/LocalScore/main/db.sqlite",
    "https://raw.githubusercontent.com/cjpais/LocalScore/main/local.db",
    "https://raw.githubusercontent.com/cjpais/LocalScore/main/localscore.db",
]

# ── Production web scrape ─────────────────────────────────────────────────────
WEB_BASE_URL = "https://www.localscore.ai/latest"
PAGE_SIZE    = 10   # results per page (hardcoded in LocalScore's UI)
REQUEST_DELAY = 0.5  # seconds between requests (be polite)

SOURCE_URL  = "https://github.com/cjpais/LocalScore"
SOURCE_NAME = "localscore"

# ── Model name → willitrun model_id ──────────────────────────────────────────
# LocalScore model names vary slightly between the DB and the web API.
MODEL_MAP: dict[str, str] = {
    # DB names (from accelerator_model_performance_scores)
    "llama 3.2 1b instruct":       "meta-llama/Llama-3.2-1B",
    "meta llama 3.2 1b instruct":  "meta-llama/Llama-3.2-1B",
    "llama 3.1 8b instruct":       "meta-llama/Llama-3.1-8B",
    "meta llama 3.1 8b instruct":  "meta-llama/Llama-3.1-8B",
    "qwen2.5 14b instruct":        "Qwen/Qwen2.5-14B",
    # Web API names (from __NEXT_DATA__ model.name field)
    "meta-llama-3.2-1b-instruct":  "meta-llama/Llama-3.2-1B",
    "meta-llama-3.1-8b-instruct":  "meta-llama/Llama-3.1-8B",
    "qwen2.5-14b-instruct":        "Qwen/Qwen2.5-14B",
}


def _resolve_model(name: str) -> str | None:
    return MODEL_MAP.get(name.lower().strip().replace(" ", "-").rstrip("-"))  \
        or MODEL_MAP.get(name.lower().strip())


def _is_apple_chip(name: str) -> bool:
    n = name.lower().strip()
    # Require "apple" explicitly, or the chip designation as a word boundary
    # ("m1 " / "m2 " / … at the start or after a space) to avoid false positives
    # on GPU names like "NVIDIA Tesla M40" which contain "m4".
    if "apple" in n:
        return True
    for i in range(1, 6):
        pattern = f"m{i} "
        if n.startswith(pattern) or f" {pattern}" in n:
            return True
    return False


def _resolve_device(accel_name: str, accel_type: str) -> str | None:
    if accel_type.upper() != "GPU":
        return None
    if _is_apple_chip(accel_name):
        return resolve_apple_chip(accel_name)
    return resolve_gpu(accel_name)


# ── DB mode ───────────────────────────────────────────────────────────────────

def download_db(from_cache: bool) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cached = RAW_DIR / "localscore.db"

    if from_cache and cached.exists():
        print(f"[localscore] Using cached DB: {cached}")
        return cached

    import requests

    last_err = None
    for url in DB_URLS:
        print(f"[localscore] Trying {url} ...")
        try:
            r = requests.get(url, timeout=60, stream=True)
            r.raise_for_status()
            first_chunk = b""
            with open(cached, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    if not first_chunk:
                        first_chunk = chunk
                    f.write(chunk)
            if not first_chunk.startswith(b"SQLite format 3"):
                cached.unlink(missing_ok=True)
                print(f"[localscore]   Not a SQLite file — skipping")
                continue
            size_mb = cached.stat().st_size / 1_048_576
            print(f"[localscore] Downloaded DB ({size_mb:.1f} MB) → {cached}")
            return cached
        except Exception as exc:
            last_err = exc
            print(f"[localscore]   Failed: {exc}")

    raise RuntimeError(f"Could not download LocalScore DB. Last error: {last_err}")


def rows_from_db(db_path: Path) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        try:
            rows = conn.execute("""
                SELECT accelerator_name, accelerator_type, accelerator_memory_gb,
                       model_name, model_variant_quant, avg_gen_tps, avg_prompt_tps
                FROM accelerator_model_performance_scores
                WHERE accelerator_type = 'GPU'
            """).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.OperationalError:
            pass
        # Fallback join
        rows = conn.execute("""
            SELECT a.name AS accelerator_name, a.type AS accelerator_type,
                   a.memory_gb AS accelerator_memory_gb,
                   m.name AS model_name, mv.quantization AS model_variant_quant,
                   MAX(br.avg_gen_tps) AS avg_gen_tps,
                   AVG(br.avg_prompt_tps) AS avg_prompt_tps
            FROM benchmark_runs br
            JOIN accelerators a ON a.id = br.accelerator_id
            JOIN model_variants mv ON mv.id = br.model_variant_id
            JOIN models m ON m.id = mv.model_id
            WHERE a.type = 'GPU'
            GROUP BY a.id, mv.id
        """).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ── Web scrape mode ───────────────────────────────────────────────────────────

def _cache_path(offset: int) -> Path:
    return RAW_DIR / f"page_{offset:06d}.json"


def fetch_page(offset: int, session, from_cache: bool) -> list[dict] | None:
    """Fetch one paginated page; return list of run dicts or None if exhausted."""
    cached = _cache_path(offset)
    if from_cache and cached.exists():
        return json.loads(cached.read_text())

    url = f"{WEB_BASE_URL}?offset={offset}"
    try:
        r = session.get(url, timeout=30)
        r.raise_for_status()
    except Exception as exc:
        print(f"  [warn] page {offset}: {exc}")
        return None

    # Extract JSON from Next.js __NEXT_DATA__ script tag
    import re
    m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', r.text, re.DOTALL)
    if not m:
        return None

    page_data = json.loads(m.group(1))
    results = page_data.get("props", {}).get("pageProps", {}).get("results", [])
    cached.write_text(json.dumps(results))
    return results


def rows_from_web(from_cache: bool) -> list[dict]:
    """Scrape all pages and return a de-duplicated best-result-per-GPU list."""
    import requests

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "willitrun-data-ingestion/1.0 (github.com/smoothyy3/willitrun)"

    # Aggregate: best avg_gen_tps per (accelerator_name, model_name, quant) tuple
    best: dict[tuple, dict] = {}
    offset = 0
    total_runs = 0

    print(f"[localscore] Scraping {WEB_BASE_URL} (10 results/page, ~334 pages expected)")
    while True:
        results = fetch_page(offset, session, from_cache)
        if results is None or len(results) == 0:
            break

        for run in results:
            accel_type = run.get("accelerator_type", "GPU")
            if accel_type != "GPU":
                continue
            model_info = run.get("model", {})
            key = (
                run.get("accelerator", ""),
                model_info.get("name", ""),
                model_info.get("quant", ""),
            )
            gen_tps = run.get("avg_gen_tps", 0) or 0
            existing = best.get(key)
            if existing is None or gen_tps > existing.get("avg_gen_tps", 0):
                best[key] = {
                    "accelerator_name":    run.get("accelerator", ""),
                    "accelerator_type":    accel_type,
                    "accelerator_memory_gb": run.get("accelerator_memory_gb"),
                    "model_name":          model_info.get("name", ""),
                    "model_variant_quant": model_info.get("quant", "Q4_K_M"),
                    "avg_gen_tps":         gen_tps,
                    "avg_prompt_tps":      run.get("avg_prompt_tps", 0) or 0,
                }

        total_runs += len(results)
        print(f"  offset={offset:5d}  runs so far: {total_runs}  unique GPUs: {len(best)}", end="\r")

        if len(results) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
        time.sleep(REQUEST_DELAY)

    print()  # newline after \r progress
    print(f"[localscore] Collected {total_runs} runs → {len(best)} unique (accel, model) pairs")
    return list(best.values())


# ── Record emission (shared by both modes) ────────────────────────────────────

def emit_records(rows: list[dict]) -> list[dict]:
    records = []
    skipped_model = skipped_device = 0
    now = datetime.now(timezone.utc).isoformat()

    for row in rows:
        model_id = _resolve_model(row["model_name"])
        if model_id is None:
            skipped_model += 1
            continue

        device_id = _resolve_device(row["accelerator_name"], row["accelerator_type"])
        if device_id is None:
            skipped_device += 1
            print(f"  [skip] unknown device: {row['accelerator_name']!r}")
            continue

        quant = row.get("model_variant_quant", "Q4_K_M") or "Q4_K_M"
        precision = "q4_k_m" if "q4_k" in quant.lower() else "int4"

        base = dict(
            model_id=model_id,
            device_id=device_id,
            precision=precision,
            framework="llamafile",
            source_url=SOURCE_URL,
            source_name=SOURCE_NAME,
            confidence="community",
            collected_at=now,
            notes=f"LocalScore · {row['accelerator_name']} · {quant}",
        )

        gen_tps = row.get("avg_gen_tps")
        if gen_tps and gen_tps > 0:
            bid = make_benchmark_id(SOURCE_NAME, model_id, device_id, precision, "tok_s_tg")
            records.append({**base, "benchmark_id": bid,
                             "metric": "tok_s_tg", "value": round(float(gen_tps), 2)})

        pp_tps = row.get("avg_prompt_tps")
        if pp_tps and pp_tps > 0:
            bid = make_benchmark_id(SOURCE_NAME, model_id, device_id, precision, "tok_s_pp")
            records.append({**base, "benchmark_id": bid,
                             "metric": "tok_s_pp", "value": round(float(pp_tps), 2)})

    print(f"[localscore] Skipped {skipped_model} unknown models, "
          f"{skipped_device} unknown devices")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest LocalScore benchmark data.")
    parser.add_argument("--mode", choices=["db", "web"], default="db",
                        help="'db' = download dev SQLite (fast, ~11 GPUs). "
                             "'web' = scrape production site (~300+ GPUs, slower).")
    parser.add_argument("--from-cache", action="store_true",
                        help="Use cached data without re-fetching.")
    args = parser.parse_args()

    if args.mode == "web":
        rows = rows_from_web(args.from_cache)
    else:
        db_path = download_db(args.from_cache)
        rows = rows_from_db(db_path)
        print(f"[localscore] {len(rows)} GPU rows from DB")

    records = emit_records(rows)

    NORM_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(NORM_FILE, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"[localscore] Wrote {len(records)} records → {NORM_FILE}")


if __name__ == "__main__":
    main()
