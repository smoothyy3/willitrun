# willitrun

[![PyPI version](https://img.shields.io/pypi/v/willitrun)](https://pypi.org/project/willitrun/)
[![Python versions](https://img.shields.io/pypi/pyversions/willitrun)](https://pypi.org/project/willitrun/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/smoothyy3/willitrun/ci.yml?branch=main)](https://github.com/smoothyy3/willitrun/actions)

> CLI to tell you if an ML model will fit and run on your device -- and find the best models for your hardware.

![willitrun inverse mode demo](assets/demo_inverse.gif)

![willitrun model check demo](assets/demo.gif)

- **Real data first:** 556 measured benchmarks across NVIDIA Jetson, Apple Silicon, desktop GPUs, and more.
- **Inverse query:** given a device, rank the best-performing models for a task category.
- **Edge to cloud:** Jetson, Apple Silicon, desktop GPUs, SBCs.
- **Vision + LLMs:** exact benchmark lookup with FLOPs/memory-based estimation as fallback.
- **MoE-aware:** distinguishes total vs active parameters for memory and speed estimation.
- **CLI-first:** fast, scriptable, offline-capable once the DB is installed.

## Quick Start

```bash
pipx install willitrun   # recommended: isolated install
willitrun                # interactive mode
```

Or with pip:
```bash
pip install willitrun
willitrun
```

### Check if a model runs on your device

```bash
willitrun                              # interactive picker
willitrun <model> --device <device>   # direct
willitrun <model> --device <device> --json
```

### Find the best models for your device

```bash
willitrun --device apple-m5-32gb --task llm
willitrun --device jetson-agx-orin-64gb --task detection
```

Or select **"Find best models for my device"** in interactive mode.

Results are sorted by real benchmark speed first, with Tier 2 estimates filling in uncovered models.

Install extras for local file profiling:
```bash
pip install willitrun[profiling,onnx]   # PyTorch / ONNX parsing
```

### Supported model inputs
- Model name from DB (`yolov8n`, `resnet50`, ...)
- HuggingFace model ID
- Local PyTorch / ONNX file

## How it works

### Estimation
- **Tier 1 (lookup):** exact model + device match from the SQLite benchmark DB.
- **Tier 2 (estimate):** FLOPs/TFLOPS scaling with memory fit check and 20% overhead. MoE models use active parameters for speed and total parameters for memory.

Results from real measurements always rank above estimates. Tier 2 results are clearly labelled.

### Data pipeline
- **Raw (immutable cache):** ingest scripts store HTTP responses in `data/raw/{source}/` with configurable TTL.
- **Normalized:** scripts parse raw data into JSONL at `data/normalized/{source}.jsonl` using the `BenchmarkRecord` schema (`willitrun/pipeline/schema.py`).
- **Serving:** `make build_db` validates, deduplicates by source priority, and writes `data/benchmarks.db`. The packaged copy in `willitrun/data/` is refreshed before each release.

## Commands

```bash
willitrun                                      # interactive mode
willitrun <model> --device <device>            # model check
willitrun --device <device> --task <category>  # best models for device
willitrun --list-devices
willitrun --list-models
willitrun <model> --device <device> --json
```

Pipeline (data contributors):
```bash
make fetch       # run all ingests (respects cache TTL)
make build_db    # normalized -> SQLite + metadata
make status      # coverage summary + gaps
make wheel       # rebuild DB, sync into package, build wheel
```

## Coverage

The inverse query mode (best models for device) is backed by real benchmarks for:

| Device | Real benchmarks |
|---|---|
| `apple-m5-32gb` | 37 LLM benchmarks (5 models) |
| `jetson-agx-orin-64gb` | 11 models across 5 categories |

Most other devices have 1-3 real benchmarks and fall back to Tier 2 estimation for uncovered models.

## Contributing data

1. Add benchmarks: append JSON lines to `data/normalized/<source>.jsonl` following `BenchmarkRecord`.
2. Add devices/models: edit `data/devices.yaml` / `data/models.yaml`.
3. Rebuild: `make build_db`.
4. Run tests: `pytest`.
5. Open a PR with the updated normalized file(s), DB, and metadata.

## Development

```bash
pip install -e ".[dev]"
make build_db
pytest
```

## Limitations
- Tier 2 estimates rely on available reference benchmarks; uncommon model/device combos may show wider ranges.
- Inverse query coverage is limited outside of M5 and Jetson AGX Orin today -- more data PRs welcome.
- Ingest scripts need network access; cached raw files avoid re-fetching within TTL.

## Roadmap
- More benchmark data (community PRs welcome!)
- Web frontend for shareability
- Training memory estimation
- Multi-GPU / model parallelism awareness
- Auto-detect local hardware

## License
MIT
