#!/usr/bin/env python3
"""Ingest benchmark data from Ultralytics Jetson guide.

Parses the YOLO benchmark tables from:
  https://docs.ultralytics.com/guides/nvidia-jetson/

Page structure (as of 2026-04):
  - 4 device sections (AGX Thor, AGX Orin 64GB, Orin Nano Super, Orin NX 16GB)
  - Each has a tabbed set with 5 model tabs (YOLO26n, s, m, l, x)
  - Each tab contains a table: Format | Status | Size (MB) | mAP | Inference time (ms/im)
  - We convert ms/im → FPS and emit one record per (model, device, format) triple

Usage:
    python scripts/ingest_ultralytics.py [--from-cache]

Requires: pip install willitrun[ingestion]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_ROOT / "raw" / "ultralytics"
NORMALIZED_DIR = DATA_ROOT / "normalized"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from willitrun.pipeline.schema import make_benchmark_id

JETSON_URL = "https://docs.ultralytics.com/guides/nvidia-jetson/"

# Map heading text → our device ID
DEVICE_MAP = {
    "NVIDIA Jetson AGX Thor": "jetson-agx-thor",
    "Jetson AGX Thor": "jetson-agx-thor",
    "AGX Thor": "jetson-agx-thor",
    "NVIDIA Jetson AGX Orin Developer Kit (64GB)": "jetson-agx-orin-64gb",
    "NVIDIA Jetson AGX Orin": "jetson-agx-orin-64gb",
    "Jetson AGX Orin 64GB": "jetson-agx-orin-64gb",
    "AGX Orin": "jetson-agx-orin-64gb",
    "NVIDIA Jetson Orin Nano Super Developer Kit": "jetson-orin-nano-8gb",
    "Jetson Orin Nano Super": "jetson-orin-nano-8gb",
    "Orin Nano Super": "jetson-orin-nano-8gb",
    "NVIDIA Jetson Orin NX 16GB": "jetson-orin-nx-16gb",
    "Jetson Orin NX 16GB": "jetson-orin-nx-16gb",
    "Orin NX 16GB": "jetson-orin-nx-16gb",
}

# Map tab label → canonical model name
MODEL_MAP = {
    "YOLO26n": "yolo26n",
    "YOLO26s": "yolo26s",
    "YOLO26m": "yolo26m",
    "YOLO26l": "yolo26l",
    "YOLO26x": "yolo26x",
    "YOLO11n": "yolo11n",
    "YOLO11s": "yolo11s",
    "YOLO11m": "yolo11m",
    "YOLO11l": "yolo11l",
    "YOLO11x": "yolo11x",
    "YOLOv8n": "yolov8n",
    "YOLOv8s": "yolov8s",
    "YOLOv8m": "yolov8m",
    "YOLOv8l": "yolov8l",
    "YOLOv8x": "yolov8x",
}

# Map export format → our framework enum value
FRAMEWORK_MAP = {
    "PyTorch": "pytorch",
    "TorchScript": "torchscript",
    "ONNX": "onnxruntime",
    "OpenVINO": "openvino",
    "TensorRT": "tensorrt",
    "CoreML": "coreml",
    "TF SavedModel": "pytorch",  # skip — not a real inference framework
    "TF GraphDef": "pytorch",
    "TF Lite": "tflite",
    "PaddlePaddle": "pytorch",
    "NCNN": "ncnn",
    "MNN": "unknown",
    "IMX": "unknown",
}

# Frameworks we actually want to keep (skip niche/non-standard ones)
KEEP_FRAMEWORKS = {"pytorch", "torchscript", "onnxruntime", "tensorrt", "tflite", "ncnn"}


def resolve_device(heading_text: str) -> str | None:
    """Resolve a heading/section title to a device ID."""
    for pattern, device_id in DEVICE_MAP.items():
        if pattern.lower() in heading_text.lower():
            return device_id
    return None


def fetch_and_save_raw(url: str) -> str | None:
    """Fetch URL and save raw HTML snapshot."""
    try:
        import requests
    except ImportError:
        print("ERROR: requests not installed. Run: pip install willitrun[ingestion]")
        return None

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    print(f"Fetching {url}...")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    raw_path = RAW_DIR / f"jetson_page_{timestamp}.html"
    raw_path.write_text(resp.text)
    print(f"  Saved raw HTML to {raw_path}")
    return resp.text


def load_cached_raw() -> str | None:
    """Load most recent cached HTML."""
    if not RAW_DIR.exists():
        return None
    files = sorted(RAW_DIR.glob("jetson_page_*.html"), reverse=True)
    if not files:
        return None
    print(f"Loading cached: {files[0]}")
    return files[0].read_text()


def parse_tabbed_benchmarks(html: str) -> list[dict]:
    """Parse benchmark data from Ultralytics tabbed section structure.

    Structure:
      <h3> or <h2> with device name
        ...
        <div class="tabbed-set">
          <div class="tabbed-labels">
            <label>YOLO26n</label> <label>YOLO26s</label> ...
          </div>
          <div class="tabbed-content">
            <table>Format | Status | Size | mAP | Inference time (ms/im)</table>
          </div>
          <div class="tabbed-content">
            <table>...</table>
          </div>
        </div>
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    records = []
    timestamp = datetime.now(timezone.utc).isoformat()

    tabbed_sets = soup.find_all(class_="tabbed-set")
    print(f"  Found {len(tabbed_sets)} tabbed sections")

    for ts in tabbed_sets:
        # Get model tab labels
        labels_div = ts.find(class_="tabbed-labels")
        if not labels_div:
            continue
        labels = [l.get_text(strip=True) for l in labels_div.find_all("label")]
        if not labels or not any(k in labels[0] for k in ("YOLO", "yolo")):
            continue

        # Resolve device from nearest heading above this tabbed-set
        device_id = None
        for heading in ts.find_all_previous(["h2", "h3", "h4"]):
            text = heading.get_text(strip=True)
            device_id = resolve_device(text)
            if device_id:
                break

        if not device_id:
            print(f"  WARNING: Could not resolve device for tabbed-set with labels {labels}")
            continue

        # Get tables (one per tab, in order matching labels)
        tables = ts.find_all("table")
        if len(tables) != len(labels):
            print(f"  WARNING: {len(labels)} labels but {len(tables)} tables for {device_id}")
            # Try to match anyway — take min
            pass

        for tab_idx, (label, table) in enumerate(zip(labels, tables)):
            model = MODEL_MAP.get(label)
            if model is None:
                print(f"  WARNING: Unknown model tab label: {label}")
                continue

            # Parse table rows
            rows = table.find_all("tr")
            for row in rows[1:]:  # skip header
                cells = [td.get_text(strip=True) for td in row.find_all("td")]
                if len(cells) < 5:
                    continue

                format_name = cells[0]
                status = cells[1]
                # size_mb = cells[2]
                # map_val = cells[3]
                latency_str = cells[4]

                # Skip failed exports
                if "❌" in status or status.lower() == "error":
                    continue

                # Parse latency → FPS
                try:
                    latency_ms = float(latency_str.replace(",", ""))
                except (ValueError, TypeError):
                    continue
                if latency_ms <= 0:
                    continue
                fps = 1000.0 / latency_ms

                # Map framework
                framework = FRAMEWORK_MAP.get(format_name, "unknown")
                if framework not in KEEP_FRAMEWORKS:
                    continue

                # Determine precision from framework
                # Ultralytics TensorRT defaults to FP16 on Jetson
                precision = "fp16"
                if framework == "pytorch":
                    precision = "fp16"  # Ultralytics benchmarks use half()

                bid = make_benchmark_id(
                    model, device_id, precision, framework, "640x640", 1
                )

                records.append({
                    "benchmark_id": bid,
                    "model": model,
                    "device": device_id,
                    "task": "detection",
                    "precision": precision,
                    "framework": framework,
                    "metric": "fps",
                    "value": round(fps, 1),
                    "unit": "frames/sec",
                    "input_size": "640x640",
                    "batch_size": 1,
                    "source": {
                        "name": "Ultralytics Jetson benchmarks",
                        "type": "official_docs",
                        "url": JETSON_URL,
                        "confidence": "high",
                    },
                    "provenance": {
                        "collected_at": timestamp,
                        "adapter": "ingest_ultralytics.py",
                        "parser_version": "0.2.0",
                    },
                    "notes": f"Converted from {latency_ms:.2f} ms/im latency",
                })

    return records


def save_normalized(records: list[dict]):
    """Save normalized records as JSONL."""
    NORMALIZED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = NORMALIZED_DIR / "ultralytics.jsonl"

    with open(output_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    print(f"\nSaved {len(records)} records to {output_path}")

    # Summary
    devices = set(r["device"] for r in records)
    models = set(r["model"] for r in records)
    print(f"  Devices: {sorted(devices)}")
    print(f"  Models: {sorted(models)}")


def main():
    parser = argparse.ArgumentParser(description="Ingest Ultralytics Jetson benchmarks")
    parser.add_argument("--from-cache", action="store_true",
                        help="Use cached HTML instead of fetching")
    args = parser.parse_args()

    if args.from_cache:
        html = load_cached_raw()
        if html is None:
            print("No cached HTML found. Run without --from-cache first.")
            sys.exit(1)
    else:
        html = fetch_and_save_raw(JETSON_URL)
        if html is None:
            sys.exit(1)

    records = parse_tabbed_benchmarks(html)
    print(f"\nParsed {len(records)} benchmark records")

    if records:
        save_normalized(records)
    else:
        print("No records parsed. The page structure may have changed.")
        print("Inspect the raw HTML and update the parser.")


if __name__ == "__main__":
    main()
