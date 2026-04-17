# Contributing to willitrun

The most impactful way to help is **adding real benchmark data**. Every new
record directly improves the tool for the next user who queries that
model–device pair.

---

## Adding Benchmark Data

### 1. Measure or source a benchmark

Run the model on the target device and record the result. Accepted metrics:

| Metric key | Meaning | Typical range |
|---|---|---|
| `fps` | Frames per second (vision models) | 1–10 000 |
| `tok_s_tg` | Tokens/sec, text generation (`tg128`) | 1–1 000 |
| `tok_s_pp` | Tokens/sec, prompt processing (`pp512`) | 10–10 000 |
| `samples_s` | Samples/sec (audio, classification) | 1–10 000 |

Only submit **real measurements**. Not theoretical peak or marketing numbers.

### 2. Write a JSONL record

Append one JSON line to `data/normalized/<source_name>.jsonl`. If no file
exists for your source yet, create one (e.g. `data/normalized/community.jsonl`).

**Required fields:**

```json
{
  "benchmark_id": "llama-3-8b__rtx-3060-12gb__4bit__llama.cpp__tg128__bs1",
  "model_id":     "llama-3-8b",
  "device_id":    "rtx-3060-12gb",
  "precision":    "int4",
  "metric":       "tok_s_tg",
  "value":        22.4,
  "framework":    "llama.cpp",
  "source_url":   "https://github.com/ggml-org/llama.cpp/discussions/15013#commentXXXX",
  "source_name":  "llama_cpp_cuda",
  "confidence":   "community",
  "collected_at": "2026-04-14T12:00:00+00:00",
  "notes":        "Q4_K_M quant, 2048 context, RTX 3060 12 GB VRAM"
}
```

**Field reference:**

| Field | Type | Allowed values | Notes |
|---|---|---|---|
| `benchmark_id` | string | (see below) | Must be unique |
| `model_id` | string | key from `data/models.yaml` | Must match exactly |
| `device_id` | string | key from `data/devices.yaml` | Must match exactly |
| `precision` | string | `fp32` `fp16` `bf16` `int8` `int4` `q4_0` `q8_0` `q4_k_m` | |
| `metric` | string | `fps` `tok_s_tg` `tok_s_pp` `samples_s` `latency_ms` | |
| `value` | number | > 0 | |
| `framework` | string | `pytorch` `tensorrt` `llama.cpp` `onnxruntime` `coreml` etc. | |
| `source_url` | string | URL or `""` | Link to the original benchmark |
| `source_name` | string | short slug | e.g. `"llama_cpp_cuda"`, `"community"` |
| `confidence` | string | `measured` `official` `community` `estimated` | Use `measured` only if you ran it yourself |
| `collected_at` | string | ISO 8601 datetime | |
| `notes` | string | freeform | Hardware setup details, quant format, etc. |

**Building the `benchmark_id`:**

```
{model_id}__{device_id}__{precision}__{framework}__{input_size}__bs{batch}
```

Example: `llama-3-8b__rtx-3060-12gb__int4__llama.cpp__tg128__bs1`

Use `tg128` or `pp512` as the input_size token for LLM benchmarks. For vision
models use the input resolution: `640x640`, `1280x1280`, etc.

### 3. Validate locally

```bash
python scripts/build_db.py --validate-only
```

This checks your record against all known `model_id` and `device_id` values
and reports any mismatches without writing to the database.

### 4. Rebuild the database and run tests

```bash
make build_db
pytest
```

### 5. Submit a PR

Include in your PR:
- The updated `data/normalized/<source>.jsonl`
- The regenerated `data/benchmarks.db` and `data/benchmarks.meta.json`
- A one-line description of the benchmark source

---

## Adding a New Model

Edit `data/models.yaml`. Required fields:

```yaml
llama-3-8b:
  aliases: ["llama3-8b", "meta-llama/Meta-Llama-3-8B"]
  model_type: llm          # llm | detection | segmentation | classification | audio | image_generation
  architecture: llama
  parameters: 8030000000
  flops: null              # null for LLMs (memory-bandwidth bound; FLOPs unused)
  weights_size_mb: 4800    # fp16 weight size
  llm_config:
    hidden_size: 4096
    num_layers: 32
    vocab_size: 128256
```

For vision models, `flops` is used for Tier 2 scaling estimates — include it
if you have it:

```yaml
yolov8m:
  aliases: ["yolov8m", "yolov8m.pt"]
  model_type: detection
  architecture: yolo
  default_input_size: "640x640"
  parameters: 25900000
  flops: 78900000000
  weights_size_mb: 49.7
```

---

## Adding a New Device

Edit `data/devices.yaml`. Required fields:

```yaml
rtx-3060-12gb:
  name: "NVIDIA GeForce RTX 3060 12 GB"
  vendor: nvidia
  type: desktop_gpu          # edge | desktop_gpu | soc | sbc | accelerator | server_gpu
  gpu:
    name: "GA106"
    tflops_fp32: 12.7
    tflops_fp16: 12.7        # 0.0 if no GPU; null if unknown
    cuda_cores: 3584
  memory:
    total_gb: 12
    type: "GDDR6"
    bandwidth_gbps: 360
    unified: false
  tdp_watts: 170
  supported_precisions: [fp32, fp16, int8, int4]
```

---

## Development Setup

```bash
git clone https://github.com/smoothyy3/willitrun.git
cd willitrun
pip install -e ".[dev]"
make build_db
pytest
```

---

## Guidelines

- Only submit real measurements, not theoretical values.
- Include the source URL so results can be verified.
- Run `pytest` and `make build_db` before submitting.
- One PR per device or benchmark batch is fine.
- If you're adding many records from the same source, consider writing an
  ingest script in `scripts/` following the pattern of existing ones.
