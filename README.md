# willitrun

[![PyPI version](https://img.shields.io/pypi/v/willitrun)](https://pypi.org/project/willitrun/)
[![Python versions](https://img.shields.io/pypi/pyversions/willitrun)](https://pypi.org/project/willitrun/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/smoothyy3/willitrun/ci.yml?branch=main)](https://github.com/smoothyy3/willitrun/actions)

> CLI to tell you if an ML model will fit and run on your device, using real benchmarks + lightweight estimation.

![willitrun demo](assets/demo.gif)

- **Real data first:** SQLite DB of measured benchmarks (NVIDIA, Ultralytics, MLPerf, community).
- **Edge to cloud:** Jetson, Apple Silicon, desktop GPUs, SBCs.
- **Vision + LLMs:** Looks up exact matches, falls back to FLOPs/TFLOPS scaling with memory fit checks.
- **CLI-first:** Fast, scriptable, offline-capable once the DB is installed.

## Quick Start (users)

The interactive UI is the primary entrypoint:
```bash
# recommended: isolated CLI install
pipx install willitrun
willitrun

# or with pip if you prefer
pip install willitrun
willitrun
```

Pick a model (name, HF ID, or local file), then pick a device. You’ll get a verdict and context instantly.

Power users (non-interactive):
```bash
willitrun <model> --device <device> [--json]
```

Install extras as needed:
```bash
# PyTorch/ONNX profiling (local .pt / .onnx parsing)
pip install willitrun[profiling,onnx]
```

## How it works

### Data pipeline (contributors)
- **Raw (immutable cache):** ingest scripts fetch and store HTTP responses in `data/raw/{source}/{timestamp}.ext` with TTL (configurable via `tool.willitrun.raw_ttl_days`).
- **Normalized:** the same scripts parse raw into JSONL at `data/normalized/{source}.jsonl` using the single `BenchmarkRecord` schema (`willitrun/pipeline/schema.py`).
- **Serving:** `make build_db` (or `python scripts/run_pipeline.py build`) validates, deduplicates by source priority, and writes `data/benchmarks.db` + `data/benchmarks.meta.json`. The packaged copy in `willitrun/data/` is refreshed by `make wheel`.

### Estimation
- **Tier 1 (lookup):** exact model+device benchmarks from SQLite.
- **Tier 2 (estimate):** FLOPs/TFLOPS scaling plus memory fit with 20% overhead; emits ranges and a confidence marker.

### Supported inputs
- Model name from DB (`yolov8n`, `resnet50`, …)
- Local PyTorch / ONNX file
- HuggingFace model ID

## Commands

User CLI:
```bash
willitrun --list-devices
willitrun --list-models
willitrun <model> --device <device> [--json]
```

Pipeline (for data contributors/maintainers):
```bash
make fetch       # run all ingests (respects cache TTL)
make build_db    # normalized -> SQLite + metadata
make status      # coverage summary + gaps
make wheel       # rebuild DB, sync into package, build wheel
```

## Contributing data

1) Add benchmarks: append JSON lines to `data/normalized/<source>.jsonl` following `BenchmarkRecord`.  
2) Add devices/models: edit `data/devices.yaml` / `data/models.yaml`.  
3) Rebuild DB: `make build_db` (regenerates DB + meta).  
4) Run tests: `pytest`.  
5) Open a PR with the updated normalized file(s), DB, and metadata.

## Development
```bash
pip install -e ".[dev]"
make build_db
pytest
```

## Release flow
```bash
make build_db          # refresh DB + metadata
make wheel             # sync into package and build
```
CI (GitHub Actions) runs `make build_db` + `pytest` on push/PR to guard against stale DBs and failing tests.

## Limitations
- Ingest scripts need network access; cached raw files avoid re-fetching within TTL.
- Tier 2 estimates rely on available reference benchmarks; uncommon models/devices may show wider ranges.

## Roadmap
- More benchmark data (community PRs welcome!)
- Web frontend for shareability
- Training memory estimation
- Multi-GPU / model parallelism awareness
- Auto-detect local hardware

## License
MIT
