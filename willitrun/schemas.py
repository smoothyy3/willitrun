"""Pydantic schemas for benchmark, device, and model validation.

These define the canonical data contracts. Every record — whether hand-curated,
scraped, or community-submitted — must pass validation against these schemas.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# --- Canonical Enums ---

class Precision(str, Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    FOUR_BIT = "4bit"
    FP8 = "fp8"


class Metric(str, Enum):
    FPS = "fps"
    TOK_S = "tok/s"
    LATENCY_MS = "latency_ms"


class Framework(str, Enum):
    PYTORCH = "pytorch"
    TENSORRT = "tensorrt"
    ONNXRUNTIME = "onnxruntime"
    COREML = "coreml"
    NCNN = "ncnn"
    TFLITE = "tflite"
    OPENVINO = "openvino"
    LLAMA_CPP = "llama.cpp"
    VLLM = "vllm"
    MLC_LLM = "mlc-llm"
    CTRANSLATE2 = "ctranslate2"
    TORCHSCRIPT = "torchscript"
    UNKNOWN = "unknown"


class DeviceType(str, Enum):
    EDGE = "edge"
    DESKTOP_GPU = "desktop_gpu"
    SOC = "soc"
    SBC = "sbc"
    ACCELERATOR = "accelerator"
    SERVER_GPU = "server_gpu"


class ModelTask(str, Enum):
    CLASSIFICATION = "classification"
    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    POSE = "pose"
    LLM = "llm"
    IMAGE_GENERATION = "image_generation"
    AUDIO = "audio"
    OTHER = "other"


class SourceType(str, Enum):
    OFFICIAL_DOCS = "official_docs"
    OFFICIAL_REPO = "official_repo"
    MLPERF = "mlperf"
    COMMUNITY_BLOG = "community_blog"
    COMMUNITY_REPO = "community_repo"
    GITHUB_DISCUSSION = "github_discussion"
    FORUM = "forum"
    SELF_MEASURED = "self_measured"


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# --- Alias normalization maps ---

PRECISION_ALIASES: dict[str, str] = {
    "fp32": "fp32", "float32": "fp32", "f32": "fp32",
    "fp16": "fp16", "float16": "fp16", "f16": "fp16", "half": "fp16",
    "bf16": "bf16", "bfloat16": "bf16",
    "int8": "int8", "i8": "int8",
    "4bit": "4bit", "4-bit": "4bit", "int4": "4bit", "w4": "4bit",
    "fp8": "fp8", "float8": "fp8",
}

FRAMEWORK_ALIASES: dict[str, str] = {
    "pytorch": "pytorch", "torch": "pytorch", "pt": "pytorch",
    "tensorrt": "tensorrt", "trt": "tensorrt",
    "onnxruntime": "onnxruntime", "ort": "onnxruntime", "onnx-runtime": "onnxruntime",
    "coreml": "coreml", "core-ml": "coreml",
    "ncnn": "ncnn",
    "tflite": "tflite", "tf-lite": "tflite", "tensorflow-lite": "tflite",
    "openvino": "openvino",
    "llama.cpp": "llama.cpp", "llamacpp": "llama.cpp", "llama_cpp": "llama.cpp",
    "vllm": "vllm",
    "mlc-llm": "mlc-llm", "mlc_llm": "mlc-llm",
    "ctranslate2": "ctranslate2", "ct2": "ctranslate2",
    "torchscript": "torchscript",
    "unknown": "unknown",
}


def normalize_precision(raw: str) -> str:
    return PRECISION_ALIASES.get(raw.lower().strip(), raw.lower().strip())


def normalize_framework(raw: str) -> str:
    return FRAMEWORK_ALIASES.get(raw.lower().strip(), raw.lower().strip())


# --- Benchmark Schema ---

class BenchmarkSource(BaseModel):
    """Where this benchmark came from."""
    name: str
    type: SourceType
    url: Optional[str] = None
    confidence: Confidence = Confidence.MEDIUM


class BenchmarkProvenance(BaseModel):
    """How and when this benchmark was collected."""
    collected_at: Optional[datetime] = None
    adapter: Optional[str] = None  # e.g. "ingest_ultralytics.py"
    parser_version: Optional[str] = None
    raw_ref: Optional[str] = None  # path to raw source snapshot


class LLMTestSetup(BaseModel):
    """LLM-specific test parameters — required for tok/s metrics."""
    context_length: Optional[int] = None
    prompt_tokens: Optional[int] = None
    generation_tokens: Optional[int] = None
    quantization_format: Optional[str] = None  # e.g. "GGUF Q4_K_M"


class BenchmarkRecord(BaseModel):
    """A single validated benchmark data point.

    The benchmark_id is the canonical dedup key, built from:
    (model, device, precision, framework, metric, input_size, batch_size)
    """
    benchmark_id: str
    model: str
    device: str
    task: ModelTask
    precision: Precision
    framework: Framework
    metric: Metric
    value: float = Field(gt=0)
    unit: Optional[str] = None  # e.g. "frames/sec", "tokens/sec"
    input_size: Optional[str] = None  # e.g. "640x640", "1x3x224x224"
    batch_size: int = Field(default=1, ge=1)

    # LLM-specific
    llm_setup: Optional[LLMTestSetup] = None

    # Runtime details
    engine_version: Optional[str] = None
    opset: Optional[int] = None
    execution_provider: Optional[str] = None
    power_mode: Optional[str] = None

    # Source & provenance
    source: BenchmarkSource
    provenance: Optional[BenchmarkProvenance] = None

    notes: Optional[str] = None

    @field_validator("precision", mode="before")
    @classmethod
    def normalize_precision_field(cls, v):
        if isinstance(v, str):
            return normalize_precision(v)
        return v

    @field_validator("framework", mode="before")
    @classmethod
    def normalize_framework_field(cls, v):
        if isinstance(v, str):
            return normalize_framework(v)
        return v

    @model_validator(mode="after")
    def validate_llm_setup(self):
        """tok/s metrics should have LLM setup info."""
        if self.metric == Metric.TOK_S and self.llm_setup is None:
            # Don't reject, but this is a gap — callers can check
            pass
        return self

    @model_validator(mode="after")
    def validate_unit(self):
        """Auto-fill unit from metric if not provided."""
        if self.unit is None:
            self.unit = {
                Metric.FPS: "frames/sec",
                Metric.TOK_S: "tokens/sec",
                Metric.LATENCY_MS: "ms",
            }.get(self.metric)
        return self


def make_benchmark_id(
    model: str,
    device: str,
    precision: str,
    framework: str,
    input_size: str | None = None,
    batch_size: int = 1,
    context_length: int | None = None,
) -> str:
    """Generate a canonical benchmark_id for deduplication."""
    parts = [model, device, precision, framework]
    if input_size:
        parts.append(input_size.replace(" ", ""))
    parts.append(f"bs{batch_size}")
    if context_length:
        parts.append(f"ctx{context_length}")
    return "__".join(parts)


# --- Device Schema ---

class GPUSpec(BaseModel):
    name: str
    cuda_cores: Optional[int] = None
    tensor_cores: Optional[int] = None
    cores: Optional[int] = None  # Apple GPU cores
    tflops_fp32: Optional[float] = None  # FP32 shader throughput
    tflops_fp16: Optional[float] = None  # FP16 throughput (None = unknown, 0 = no GPU)
    tops_int8: Optional[float] = None


class MemorySpec(BaseModel):
    total_gb: Optional[float] = None  # None = unknown, fill from datasheet
    type: Optional[str] = None  # e.g. "GDDR6X", "LPDDR5"
    bandwidth_gbps: Optional[float] = None  # GB/s
    unified: bool = False


class DeviceRecord(BaseModel):
    """A validated device specification."""
    device_id: str  # filled at load time from dict key
    name: str
    vendor: Optional[str] = None
    device_family: Optional[str] = None
    type: DeviceType
    gpu: GPUSpec
    memory: MemorySpec
    tdp_watts: Optional[float] = None
    supported_precisions: list[Precision]
    supported_runtimes: Optional[list[Framework]] = None
    notes: Optional[str] = None


# --- Validation Utilities ---

class ValidationResult(BaseModel):
    valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def validate_benchmark_against_device(
    benchmark: BenchmarkRecord,
    devices: dict[str, DeviceRecord],
) -> ValidationResult:
    """Cross-validate a benchmark record against the device database."""
    errors = []
    warnings = []

    # Device must exist
    if benchmark.device not in devices:
        errors.append(f"Unknown device: {benchmark.device}")
        return ValidationResult(valid=False, errors=errors)

    device = devices[benchmark.device]

    # Precision must be supported by device
    if benchmark.precision not in device.supported_precisions:
        errors.append(
            f"Device {benchmark.device} does not support {benchmark.precision}. "
            f"Supported: {[p.value for p in device.supported_precisions]}"
        )

    # CPU-only devices should not have GPU-only frameworks
    # Only reject if tflops_fp16 is explicitly 0 (no GPU), not if None (unknown)
    gpu_frameworks = {Framework.TENSORRT, Framework.COREML}
    if device.gpu.tflops_fp16 == 0 and benchmark.framework in gpu_frameworks:
        errors.append(
            f"Framework {benchmark.framework} requires GPU, "
            f"but {benchmark.device} has no GPU compute."
        )

    # Sanity: value should not be suspiciously large
    if benchmark.metric == Metric.FPS and benchmark.value > 10000:
        warnings.append(f"Suspiciously high FPS: {benchmark.value}")
    # tok/s threshold: prompt processing (pp) can hit 10k+ on fast GPUs,
    # text generation (tg) is typically <500. Use a generous threshold.
    if benchmark.metric == Metric.TOK_S and benchmark.value > 20000:
        warnings.append(f"Suspiciously high tok/s: {benchmark.value}")

    # LLM metric without setup info
    if benchmark.metric == Metric.TOK_S and benchmark.llm_setup is None:
        warnings.append("tok/s metric without LLM test setup (context_length, quantization_format)")

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )
