# Data Sources

All benchmark data shipped with willitrun comes from publicly available sources.
This file documents each source, its license, and how data is used.

---

## 1. llama.cpp Community Scoreboards

| | |
|---|---|
| **Source** | GitHub Discussions: [CUDA #15013](https://github.com/ggml-org/llama.cpp/discussions/15013), [ROCm #15021](https://github.com/ggml-org/llama.cpp/discussions/15021), [Vulkan #10879](https://github.com/ggml-org/llama.cpp/discussions/10879), [Apple #4167](https://github.com/ggml-org/llama.cpp/discussions/4167) |
| **License** | MIT (llama.cpp repository). Community-contributed benchmark numbers; no explicit license on individual entries. |
| **Data** | Llama 2 7B Q4_0 text-generation and prompt-processing tok/s across CUDA, ROCm, Vulkan, and Apple Metal GPUs. |
| **Ingest script** | `scripts/ingest_llama_cpp.py` |
| **JSONL file** | `data/normalized/llama_cpp_cuda.jsonl` |

## 2. MLCommons MLPerf Inference

| | |
|---|---|
| **Source** | [inference_results_v5.1](https://raw.githubusercontent.com/mlcommons/inference_results_v5.1/main/summary_results.json) (and prior versions) |
| **License** | Apache 2.0. The MLPerf name is a trademark of MLCommons Association; results may be cited with attribution. |
| **Data** | Single-GPU SingleStream inference results for vision models (ResNet-50, RetinaNet, BERT) on server-class GPUs. LLM tok/s results are excluded — MLPerf measures aggregate batch throughput, not single-user generation speed. |
| **Ingest script** | `scripts/ingest_mlperf.py` |
| **JSONL file** | `data/normalized/mlperf.jsonl` |

## 3. NVIDIA Jetson Benchmarks

| | |
|---|---|
| **Source** | [developer.nvidia.com/embedded/jetson-benchmarks](https://developer.nvidia.com/embedded/jetson-benchmarks) |
| **License** | Public benchmark page published by NVIDIA. No explicit open license; cited for informational purposes. |
| **Data** | LLM/VLM tok/s and MLPerf results for Jetson AGX Thor, AGX Orin, Orin NX, and Orin Nano devices. |
| **Ingest script** | `scripts/ingest_nvidia_jetson.py` |
| **JSONL file** | `data/normalized/nvidia_jetson.jsonl` |

## 4. Ultralytics Jetson Guide

| | |
|---|---|
| **Source** | [docs.ultralytics.com/guides/nvidia-jetson/](https://docs.ultralytics.com/guides/nvidia-jetson/) |
| **License** | AGPL-3.0 (Ultralytics repository). Benchmark numbers are factual measurements; citation is standard practice. |
| **Data** | YOLO26 detection benchmarks (FPS, mAP) across Jetson AGX Thor, AGX Orin, Orin Nano Super, and Orin NX 16 GB, covering multiple export formats (TensorRT, ONNX, PyTorch). |
| **Ingest script** | `scripts/ingest_ultralytics.py` |
| **JSONL file** | `data/normalized/ultralytics.jsonl` |

## 5. Ultralytics GPU Model Docs

| | |
|---|---|
| **Source** | [docs.ultralytics.com/models/yolov8/](https://docs.ultralytics.com/models/yolov8/) and [docs.ultralytics.com/models/yolo11/](https://docs.ultralytics.com/models/yolo11/) |
| **License** | AGPL-3.0 (Ultralytics repository). |
| **Data** | YOLOv8 benchmarks on NVIDIA A100 (TensorRT FP16) and YOLO11 benchmarks on NVIDIA T4 (TensorRT10 FP16), at 640×640 input, batch size 1. |
| **Ingest script** | `scripts/ingest_ultralytics_gpu.py` |
| **JSONL file** | `data/normalized/ultralytics_gpu.jsonl` |

## 6. XiongjieDai GPU-Benchmarks-on-LLM-Inference

| | |
|---|---|
| **Source** | [github.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference](https://github.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference) |
| **License** | MIT |
| **Data** | Community-measured LLM text-generation tok/s (tg1024) for Llama 3 8B and 70B in Q4_K_M and F16 across a wide range of consumer and server GPUs. |
| **Ingest script** | `scripts/ingest_xiongjieddai.py` |
| **JSONL file** | `data/normalized/xiongjieddai.jsonl` |

## 7. Hand-Curated Records

| | |
|---|---|
| **Source** | Maintainer-verified measurements and official model card numbers |
| **License** | MIT (same as willitrun) |
| **Data** | High-confidence records that override scraped data when conflicts arise. |
| **File** | `data/benchmarks.curated.yaml` |

---

## mac-llm-bench

| | |
|---|---|
| **Source** | [mac-llm-bench](https://github.com/mac-llm-bench/mac-llm-bench) community tool |
| **License** | MIT |
| **Data** | LLM inference benchmarks on Apple Silicon, collected via the mac-llm-bench CLI. Records are contributed by device owners and ingested via `scripts/ingest_mac_llm_bench.py`. |
| **JSONL file** | `data/normalized/mac_llm_bench.jsonl` |
| **Note** | This source requires running the ingest script manually with a local results directory — it is not fetched automatically. |

## 8. LocalScore

| | |
|---|---|
| **Source** | [github.com/cjpais/LocalScore](https://github.com/cjpais/LocalScore) |
| **License** | Apache 2.0 |
| **Data** | Community-submitted GPU inference benchmarks for three fixed models: Meta Llama 3.2 1B, Meta Llama 3.1 8B, and Qwen2.5 14B (all Q4_K_M via llamafile). The leaderboard aggregates results from hundreds of GPU submissions into one row per (GPU, model). Two metrics per entry: text-generation tok/s and prompt-processing tok/s. |
| **Ingest script** | `scripts/ingest_localscore.py` |
| **JSONL file** | `data/normalized/localscore.jsonl` |

## 9. geerlingguy/ai-benchmarks

| | |
|---|---|
| **Source** | [github.com/geerlingguy/ai-benchmarks](https://github.com/geerlingguy/ai-benchmarks) |
| **License** | MIT |
| **Data** | Community benchmark table in README.md covering DeepSeek R1 (14B, 671B), Llama 3.2 3B, Llama 3.1 70B, and gpt-oss 20B across 50+ systems including AMD RDNA3, Intel Arc, Apple Silicon, and NVIDIA RTX 30/40/50 series. Notable for AMD and non-NVIDIA coverage. All results use Ollama (Q4_K_M default). Metric is token generation eval rate (tok/s). |
| **Ingest script** | `scripts/ingest_geerlingguy.py` |
| **JSONL file** | `data/normalized/geerlingguy.jsonl` |

---

## Adding a New Source

See [CONTRIBUTING.md](CONTRIBUTING.md) for instructions on adding benchmark
records, new models, and new devices.

If you are adding a scraper for a new data source, document it here before
submitting the PR.
