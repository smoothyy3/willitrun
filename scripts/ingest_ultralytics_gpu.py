#!/usr/bin/env python3
"""Ingest benchmark data from Ultralytics model documentation (GPU tables).

Sources:
  YOLOv8:  https://docs.ultralytics.com/models/yolov8/   — NVIDIA A100, TensorRT FP16
  YOLO11:  https://docs.ultralytics.com/models/yolo11/   — NVIDIA T4,  TensorRT10 FP16

Page structure (as of 2026-04):
  - Detection table with columns including model, mAP, CPU ONNX ms, GPU TensorRT ms
  - YOLOv8 GPU column: "Speed A100 TensorRT (ms)"  → device: a100-80gb
  - YOLO11 GPU column: "Speed T4 TensorRT10 (ms)"  → device: t4-16gb
  - YOLO11 values include ± stddev (e.g. "1.5 ± 0.0") — we take the mean
  - All benchmarks at 640×640 input, batch size 1

Usage:
    python scripts/ingest_ultralytics_gpu.py [--from-cache]

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
RAW_DIR = DATA_ROOT / "raw" / "ultralytics_gpu"
NORMALIZED_DIR = DATA_ROOT / "normalized"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from willitrun.pipeline.schema import make_benchmark_id

# ---- Pages to scrape ----
PAGES = [
    {
        "url": "https://docs.ultralytics.com/models/yolov8/",
        "cache_name": "yolov8",
        # Substring that must appear in the GPU column header
        "gpu_col_pattern": "A100",
        "device_id": "a100-80gb",
        "framework": "tensorrt",
        "precision": "fp16",
        # Maps table row model names → our canonical IDs
        "model_map": {
            "yolov8n": "yolov8n",
            "yolov8s": "yolov8s",
            "yolov8m": "yolov8m",
            "yolov8l": "yolov8l",
            "yolov8x": "yolov8x",
        },
    },
    {
        "url": "https://docs.ultralytics.com/models/yolo11/",
        "cache_name": "yolo11",
        "gpu_col_pattern": "T4",
        "device_id": "t4-16gb",
        "framework": "tensorrt",
        "precision": "fp16",
        "model_map": {
            "yolo11n": "yolo11n",
            "yolo11s": "yolo11s",
            "yolo11m": "yolo11m",
            "yolo11l": "yolo11l",
            "yolo11x": "yolo11x",
        },
    },
]


def fetch_and_save_raw(url: str, cache_name: str) -> str | None:
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
    if not RAW_DIR.exists():
        return None
    files = sorted(RAW_DIR.glob(f"{cache_name}_*.html"), reverse=True)
    if not files:
        return None
    print(f"Loading cached: {files[0]}")
    return files[0].read_text()


def parse_ms_value(raw: str) -> float | None:
    """Parse '1.5 ± 0.0' or '0.99' → float ms. Returns None on failure."""
    raw = raw.strip()
    # Strip ± stddev
    m = re.match(r"([\d.]+)(?:\s*±\s*[\d.]+)?", raw)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def parse_detection_table(html: str, page: dict) -> list[dict]:
    """Find the detection benchmark table and extract GPU latency rows.

    Ultralytics docs pages have multiple tables (detection, seg, pose, cls).
    We identify the right table by looking for the GPU column pattern in headers.
    Model names in the first column are matched case-insensitively.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    records = []
    timestamp = datetime.now(timezone.utc).isoformat()

    tables = soup.find_all("table")
    print(f"  Found {len(tables)} tables on page")

    gpu_pattern = page["gpu_col_pattern"].lower()
    device_id = page["device_id"]
    model_map = page["model_map"]
    found_table = False
    seen_bids: set[str] = set()  # dedup within this page — first detection table wins

    for table in tables:
        # Get header cells
        header_row = table.find("tr")
        if not header_row:
            continue
        headers = [th.get_text(strip=True) for th in header_row.find_all(["th", "td"])]
        headers_lower = [h.lower() for h in headers]

        # Only accept the COCO detection table — must have "mapval" (mAPval50-95)
        # This excludes segmentation (mAPbox/mAPmask), pose (mAPpose), cls (acctop), OBB (mAPtest)
        if not any("mapval" in h for h in headers_lower):
            continue

        # Find GPU column index — must contain gpu_pattern (e.g. "A100" or "T4")
        gpu_col = None
        for i, h in enumerate(headers_lower):
            if gpu_pattern in h and "tensorrt" in h:
                gpu_col = i
                break
        if gpu_col is None:
            continue

        # Find model column (first column)
        model_col = 0
        found_table = True
        print(f"  Found GPU table: headers={headers}")
        print(f"  GPU column: [{gpu_col}] = '{headers[gpu_col]}'")

        rows = table.find_all("tr")[1:]  # skip header
        row_count = 0
        for row in rows:
            cells = row.find_all(["td", "th"])
            if len(cells) <= gpu_col:
                continue

            model_raw = cells[model_col].get_text(strip=True).lower()
            latency_raw = cells[gpu_col].get_text(strip=True)

            # Match model name
            model_id = None
            for key, val in model_map.items():
                if key in model_raw:
                    model_id = val
                    break
            if model_id is None:
                continue

            ms = parse_ms_value(latency_raw)
            if ms is None or ms <= 0:
                continue

            fps = round(1000.0 / ms, 1)

            bid = make_benchmark_id(
                model_id, device_id, page["precision"], page["framework"],
                "640x640", 1,
            )

            if bid in seen_bids:
                continue
            seen_bids.add(bid)

            records.append({
                "benchmark_id": bid,
                "model": model_id,
                "device": device_id,
                "task": "detection",
                "precision": page["precision"],
                "framework": page["framework"],
                "metric": "fps",
                "value": fps,
                "unit": "frames/sec",
                "input_size": "640x640",
                "batch_size": 1,
                "source": {
                    "name": f"Ultralytics {model_id.upper()} docs",
                    "type": "official_docs",
                    "url": page["url"],
                    "confidence": "high",
                },
                "provenance": {
                    "collected_at": timestamp,
                    "adapter": "ingest_ultralytics_gpu.py",
                    "parser_version": "0.1.0",
                },
                "notes": (
                    f"Converted from {ms:.2f} ms/im TensorRT latency; "
                    f"batch_size=1, input=640×640"
                ),
            })
            row_count += 1
            print(f"    {model_id} on {device_id}: {ms} ms → {fps} fps")

        print(f"  → {row_count} records from this table")

    if not found_table:
        print(f"  WARNING: No table found matching gpu_col_pattern='{gpu_pattern}' + 'tensorrt'")

    return records


def save_normalized(records: list[dict]):
    NORMALIZED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = NORMALIZED_DIR / "ultralytics_gpu.jsonl"

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
        description="Ingest Ultralytics GPU benchmark tables (YOLOv8/YOLO11)"
    )
    parser.add_argument("--from-cache", action="store_true",
                        help="Use cached HTML instead of fetching")
    args = parser.parse_args()

    all_records = []

    for page in PAGES:
        print(f"\n--- {page['cache_name']} ({page['device_id']}) ---")
        if args.from_cache:
            html = load_cached_raw(page["cache_name"])
            if html is None:
                print(f"  No cache found for {page['cache_name']}. Skipping.")
                continue
        else:
            html = fetch_and_save_raw(page["url"], page["cache_name"])
            if html is None:
                continue

        records = parse_detection_table(html, page)
        all_records.extend(records)

    print(f"\nTotal: {len(all_records)} benchmark records")
    if all_records:
        save_normalized(all_records)
    else:
        print("No records parsed. Check page structure or run without --from-cache.")


if __name__ == "__main__":
    main()
