#!/usr/bin/env python3
"""Ingest MLCommons MLPerf Inference benchmark results.

Source: MLCommons inference_results_v5.1 (and later versions)
  https://raw.githubusercontent.com/mlcommons/inference_results_v{VERSION}/main/summary_results.json

Results are published on GitHub under Apache 2.0. Numbers may be cited with
attribution; the MLPerf name is a trademark of MLCommons Association.

Design decisions:
  - Only single-GPU submissions (a# == 1) are used. Multi-GPU server results
    cannot be reliably divided to produce per-GPU estimates.
  - Vision models: SingleStream scenario only (fps = 1000 / latency_ms).
    This represents true one-request-at-a-time throughput.
  - Audio models (whisper): Offline scenario accepted since audio models
    process fixed-length segments without autoregressive token dependency.
  - LLM tok/s: EXCLUDED. MLPerf Offline/Server tok/s is aggregate batch
    throughput, not single-user text generation speed. It would be 10-50×
    higher than what a user running llama.cpp would observe and would be
    directly misleading alongside our llama.cpp tg128 benchmarks.

Usage:
    python scripts/ingest_mlperf.py               # fetch latest from GitHub
    python scripts/ingest_mlperf.py --from-cache  # use cached JSON
    python scripts/ingest_mlperf.py --version 5.0 # specify version

Requires: pip install willitrun[ingestion]  (requests)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_ROOT / "raw" / "mlperf"
NORMALIZED_DIR = DATA_ROOT / "normalized"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from willitrun.pipeline.schema import make_benchmark_id

DEFAULT_VERSION = "5.1"
GITHUB_URL_TEMPLATE = (
    "https://raw.githubusercontent.com/mlcommons/inference_results_v{version}"
    "/main/summary_results.json"
)

# ── Model name mapping ────────────────────────────────────────────────────────
# MLPerf model name → canonical willitrun model ID
MODEL_MAP: dict[str, str | None] = {
    "resnet":                 "resnet50",
    "resnet-50":              "resnet50",
    "retinanet":              "retinanet",
    "3d-unet-99":             "3d-unet",
    "3d-unet-99.9":           "3d-unet",
    "whisper":                "whisper-large-v3",
    "stable-diffusion-xl":    "sdxl",
    # LLMs: excluded from output (batch metrics), mapped to None to mark as skip
    "llama2-70b-99":          None,
    "llama2-70b-99.9":        None,
    "llama3.1-8b":            None,
    "llama3.1-8b-edge":       None,
    "llama3.1-405b":          None,
    "mixtral-8x7b":           None,
    "deepseek-r1":            None,
    "gpt-oss-120b":           None,
    # Niche / not in our models.yaml
    "dlrm-v2-99":             None,
    "dlrm-v2-99.9":           None,
    "rgat":                   None,
    "pointpainting":          None,
    "wan-2.2-t2v-a14b":       None,
}

# ── Device (accelerator) name mapping ────────────────────────────────────────
# MLPerf "Accelerator" string → canonical willitrun device ID
# Only devices that exist in data/devices.yaml are included.
# Multi-GPU system strings are intentionally absent — they cannot map to
# single-device entries.
DEVICE_MAP: dict[str, str] = {
    # Consumer desktop
    "NVIDIA GeForce RTX 4090":          "rtx-4090-24gb",
    "NVIDIA GeForce RTX 3090":          "rtx-3090-24gb",
    "NVIDIA GeForce RTX 3080":          "rtx-3080-10gb",
    # Datacenter
    "NVIDIA H100-SXM-80GB":             "h100-80gb",
    "NVIDIA H100-PCIe-80GB":            "h100-80gb",
    "NVIDIA A100-SXM-80GB":             "a100-80gb",
    "NVIDIA A100-PCIe-80GB":            "a100-80gb",
    "NVIDIA L40S":                      "l40s-48gb",
    # Edge
    "NVIDIA Jetson AGX Thor 128G":      "jetson-agx-thor",
    "NVIDIA Jetson AGX Orin":           "jetson-agx-orin-64gb",
}

# ── Task mapping ──────────────────────────────────────────────────────────────
TASK_MAP: dict[str, str] = {
    "resnet50":           "classification",
    "retinanet":          "detection",
    "3d-unet":            "segmentation",
    "whisper-large-v3":   "audio",
    "sdxl":               "image_generation",
}

# ── Precision normalisation ───────────────────────────────────────────────────
PRECISION_MAP: dict[str, str] = {
    "fp32":   "fp32",
    "fp16":   "fp16",
    "fp8":    "fp8",
    "bf16":   "bf16",
    "int8":   "int8",
    "INT8":   "int8",
    "UINT4":  "4bit",
    "fp4":    "4bit",
}


def normalize_precision(raw: str) -> str:
    """Simplify MLPerf's sometimes-verbose precision strings."""
    raw = raw.strip()
    if raw in PRECISION_MAP:
        return PRECISION_MAP[raw]
    # Composite strings like "clip1:fp32,unet:fp8,vae:fp32" → dominant type
    for key in ("fp8", "int8", "fp16", "fp4", "UINT4", "bf16", "fp32"):
        if key.lower() in raw.lower():
            return PRECISION_MAP.get(key, key.lower())
    return raw.lower()


def resolve_device(accelerator: str) -> str | None:
    """Map MLPerf accelerator string to a canonical device ID."""
    if not accelerator:
        return None
    # Exact match first
    if accelerator in DEVICE_MAP:
        return DEVICE_MAP[accelerator]
    # Substring match (handles minor suffix variations)
    for pattern, device_id in DEVICE_MAP.items():
        if pattern in accelerator:
            return device_id
    return None


def fetch_and_save_raw(version: str) -> list[dict] | None:
    try:
        import requests
    except ImportError:
        print("ERROR: requests not installed. Run: pip install willitrun[ingestion]")
        return None

    url = GITHUB_URL_TEMPLATE.format(version=version)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    print(f"Fetching MLPerf v{version} results from GitHub...")
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
    except Exception as e:
        print(f"  ERROR: {e}")
        return None

    data = resp.json()
    cache_path = RAW_DIR / f"summary_results_v{version}_{timestamp}.json"
    cache_path.write_text(json.dumps(data, indent=2))
    print(f"  Saved {len(data)} records → {cache_path}")
    return data


def load_cached_raw(version: str) -> list[dict] | None:
    if not RAW_DIR.exists():
        return None
    files = sorted(RAW_DIR.glob(f"summary_results_v{version}_*.json"), reverse=True)
    if not files:
        print(f"No cached file for v{version} in {RAW_DIR}")
        return None
    print(f"Loading cached: {files[0]}")
    with open(files[0]) as f:
        return json.load(f)


def parse(data: list[dict], version: str) -> list[dict]:
    """Convert raw MLPerf summary records into willitrun normalized records.

    Selection criteria (see module docstring for rationale):
      1. a# == 1           — single GPU only
      2. Scenario match    — SingleStream for vision, Offline for audio
      3. Model in MODEL_MAP and not mapped to None
      4. Device in DEVICE_MAP
      5. errors == 0 and compliance == 1 (or "closed")
    """
    records: list[dict] = []
    seen_ids: set[str] = set()
    skipped: dict[str, int] = {}
    collected_at = datetime.now(timezone.utc).isoformat()
    source_url = GITHUB_URL_TEMPLATE.format(version=version)

    for entry in data:
        model_raw = entry.get("Model", entry.get("UsedModel", ""))
        accelerator = entry.get("Accelerator", "")
        scenario = entry.get("Scenario", "")
        a_num = entry.get("a#", 0)
        perf = entry.get("Performance_Result")
        perf_units = entry.get("Performance_Units", "")
        prec_raw = entry.get("weight_data_types", "fp32")
        compliance = entry.get("compliance", 0)
        errors = entry.get("errors", 1)

        # ── Filter ────────────────────────────────────────────────────────────
        # Single GPU only
        if a_num != 1:
            skipped["multi_gpu"] = skipped.get("multi_gpu", 0) + 1
            continue

        # Skip failed / non-compliant submissions
        if errors != 0:
            skipped["errors"] = skipped.get("errors", 0) + 1
            continue
        if compliance not in (1, "closed"):
            skipped["non_compliant"] = skipped.get("non_compliant", 0) + 1
            continue

        # Model must be mapped and not excluded (None)
        model_lower = model_raw.lower().strip()
        if model_lower not in MODEL_MAP:
            skipped["unknown_model"] = skipped.get("unknown_model", 0) + 1
            continue
        canonical_model = MODEL_MAP[model_lower]
        if canonical_model is None:
            skipped["excluded_model"] = skipped.get("excluded_model", 0) + 1
            continue

        # Device must be known
        device_id = resolve_device(accelerator)
        if device_id is None:
            skipped["unknown_device"] = skipped.get("unknown_device", 0) + 1
            continue

        # Value must be present and positive
        if perf is None or perf <= 0:
            skipped["no_value"] = skipped.get("no_value", 0) + 1
            continue

        # ── Scenario → metric conversion ──────────────────────────────────────
        task = TASK_MAP.get(canonical_model, "other")
        is_audio = canonical_model == "whisper-large-v3"

        if scenario == "SingleStream":
            # Value is 90th-percentile latency in ms → convert to fps
            if "latency" not in perf_units.lower() and "ms" not in perf_units.lower():
                # Some SingleStream entries report samples/s directly
                fps = perf
                metric_note = f"MLPerf SingleStream {perf:.1f} samples/s"
            else:
                if perf <= 0:
                    continue
                fps = round(1000.0 / perf, 2)
                metric_note = f"MLPerf SingleStream, latency={perf:.3f}ms"
            metric = "fps"

        elif scenario == "Offline" and is_audio:
            # Audio Offline: throughput in samples/s is meaningful
            # (each audio chunk is independent — no autoregressive bottleneck)
            fps = round(perf, 3)
            metric = "fps"
            metric_note = f"MLPerf Offline, {perf:.1f} samples/s (audio segments)"

        else:
            # All other scenarios (Offline vision, Server, Interactive, MultiStream)
            # are skipped for now
            skipped["scenario"] = skipped.get("scenario", 0) + 1
            continue

        # ── Build benchmark record ────────────────────────────────────────────
        precision = normalize_precision(str(prec_raw))
        framework = "tensorrt"  # all GPU submissions use TRT; CPU would be "onnxruntime"
        if "openvino" in str(entry.get("Software", "")).lower():
            framework = "openvino"
        elif "onnx" in str(entry.get("Software", "")).lower():
            framework = "onnxruntime"

        bid = make_benchmark_id("mlperf", canonical_model, device_id, precision, "fps") + "__mlperf_ss"

        # Keep the best (highest fps) if duplicate benchmark_id
        if bid in seen_ids:
            # Find existing and compare
            existing = next(r for r in records if r["benchmark_id"] == bid)
            if fps > existing["value"]:
                records.remove(existing)
                seen_ids.discard(bid)
            else:
                skipped["duplicate_lower"] = skipped.get("duplicate_lower", 0) + 1
                continue

        seen_ids.add(bid)
        records.append({
            "benchmark_id": bid,
            "model": canonical_model,
            "device": device_id,
            "task": task,
            "precision": precision,
            "framework": framework,
            "metric": metric,
            "value": fps,
            "unit": "frames/sec",
            "batch_size": 1,
            "source": {
                "name": f"MLCommons MLPerf Inference v{version}",
                "type": "mlperf",
                "url": source_url,
                "confidence": "high",
            },
            "provenance": {
                "collected_at": collected_at,
                "adapter": "ingest_mlperf.py",
                "parser_version": "0.1.0",
            },
            "notes": metric_note,
        })

    print(f"\n  Parsed {len(records)} records. Skipped: {skipped}")
    return records


def save_normalized(records: list[dict]) -> None:
    NORMALIZED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = NORMALIZED_DIR / "mlperf.jsonl"

    with open(output_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    print(f"\nSaved {len(records)} records → {output_path}")
    for r in records:
        print(f"  {r['benchmark_id']}: {r['value']} {r['metric']}  ({r['notes'][:60]})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest MLCommons MLPerf Inference results")
    parser.add_argument("--from-cache", action="store_true", help="Use cached JSON")
    parser.add_argument("--version", default=DEFAULT_VERSION,
                        help=f"MLPerf version (default: {DEFAULT_VERSION})")
    args = parser.parse_args()

    if args.from_cache:
        data = load_cached_raw(args.version)
    else:
        data = fetch_and_save_raw(args.version)

    if data is None:
        print("No data available.")
        sys.exit(1)

    print(f"\nTotal raw records: {len(data)}")
    records = parse(data, args.version)

    if records:
        save_normalized(records)
    else:
        print("No usable records parsed.")


if __name__ == "__main__":
    main()
