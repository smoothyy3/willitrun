# Contributing to willitrun

Thanks for your interest in contributing! The most impactful way to help is by **adding real benchmark data**.

## Adding Benchmark Data

1. Run your model on your device and measure FPS or tok/s.
2. Append a JSON line to `data/normalized/<source>.jsonl` using the `BenchmarkRecord` schema (see `willitrun/pipeline/schema.py`). Use existing files as examples.
3. If you’re adding hand-verified data, you can also place it in `data/benchmarks.curated.yaml` (highest priority).
4. Run `make build_db` to regenerate `data/benchmarks.db` and `data/benchmarks.meta.json`.
5. Submit a PR with the updated JSONL + regenerated DB/metadata.

## Adding a New Device

Edit `data/devices.yaml`. Required fields:

```yaml
device-id:
  name: "Human-readable name"
  type: edge | desktop_gpu | soc | sbc | accelerator
  gpu:
    name: "GPU name"
    tflops_fp16: 0.0
  memory:
    total_gb: 8
    type: "LPDDR5"
    bandwidth_gbps: 100
    unified: false
  tdp_watts: 15
  supported_precisions: [fp32, fp16, int8]
```

## Development Setup

```bash
git clone https://github.com/your-username/willitrun.git
cd willitrun
pip install -e ".[dev]"
make build_db
pytest
```

## Guidelines

- Keep benchmark data honest — only submit real measurements, not theoretical values.
- Include the source of your benchmark (URL or description).
- One PR per device or benchmark batch is fine.
- Run `pytest` and `make build_db` before submitting.
