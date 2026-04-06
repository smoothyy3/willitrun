"""Tests for Pydantic schemas and validation."""

import pytest
from pydantic import ValidationError

from willitrun.schemas import (
    BenchmarkRecord,
    BenchmarkSource,
    Confidence,
    DeviceRecord,
    Framework,
    LLMTestSetup,
    Metric,
    Precision,
    SourceType,
    ValidationResult,
    make_benchmark_id,
    normalize_framework,
    normalize_precision,
    validate_benchmark_against_device,
)


# --- Precision normalization ---

def test_normalize_precision_standard():
    assert normalize_precision("fp16") == "fp16"
    assert normalize_precision("FP16") == "fp16"
    assert normalize_precision("float16") == "fp16"
    assert normalize_precision("half") == "fp16"


def test_normalize_precision_4bit():
    assert normalize_precision("4bit") == "4bit"
    assert normalize_precision("4-bit") == "4bit"
    assert normalize_precision("int4") == "4bit"


# --- Framework normalization ---

def test_normalize_framework():
    assert normalize_framework("tensorrt") == "tensorrt"
    assert normalize_framework("trt") == "tensorrt"
    assert normalize_framework("llama.cpp") == "llama.cpp"
    assert normalize_framework("llamacpp") == "llama.cpp"
    assert normalize_framework("ort") == "onnxruntime"


# --- Benchmark ID ---

def test_make_benchmark_id():
    bid = make_benchmark_id("yolov8n", "jetson-orin-nano-8gb", "fp16", "tensorrt", "640x640", 1)
    assert bid == "yolov8n__jetson-orin-nano-8gb__fp16__tensorrt__640x640__bs1"


def test_make_benchmark_id_with_context():
    bid = make_benchmark_id("llama-3", "rtx-3060", "4bit", "llama.cpp", context_length=2048)
    assert "ctx2048" in bid


# --- BenchmarkRecord validation ---

def test_valid_benchmark_record():
    record = BenchmarkRecord(
        benchmark_id="test__dev__fp16__tensorrt__640x640__bs1",
        model="yolov8n",
        device="jetson-orin-nano-8gb",
        task="detection",
        precision="fp16",
        framework="tensorrt",
        metric="fps",
        value=97,
        input_size="640x640",
        source=BenchmarkSource(name="test", type="official_docs", confidence="high"),
    )
    assert record.precision == Precision.FP16
    assert record.framework == Framework.TENSORRT
    assert record.unit == "frames/sec"  # auto-filled


def test_benchmark_record_precision_normalization():
    record = BenchmarkRecord(
        benchmark_id="test",
        model="m",
        device="d",
        task="detection",
        precision="half",  # should normalize to fp16
        framework="trt",   # should normalize to tensorrt
        metric="fps",
        value=10,
        source=BenchmarkSource(name="test", type="official_docs"),
    )
    assert record.precision == Precision.FP16
    assert record.framework == Framework.TENSORRT


def test_benchmark_record_rejects_negative_value():
    with pytest.raises(ValidationError):
        BenchmarkRecord(
            benchmark_id="test",
            model="m",
            device="d",
            task="detection",
            precision="fp16",
            framework="tensorrt",
            metric="fps",
            value=-5,
            source=BenchmarkSource(name="test", type="official_docs"),
        )


def test_benchmark_record_rejects_zero_value():
    with pytest.raises(ValidationError):
        BenchmarkRecord(
            benchmark_id="test",
            model="m",
            device="d",
            task="detection",
            precision="fp16",
            framework="tensorrt",
            metric="fps",
            value=0,
            source=BenchmarkSource(name="test", type="official_docs"),
        )


# --- Cross-validation ---

def _make_device(device_id="test-device", tflops=10.0, precisions=None):
    if precisions is None:
        precisions = [Precision.FP32, Precision.FP16, Precision.INT8]
    return DeviceRecord(
        device_id=device_id,
        name="Test Device",
        type="desktop_gpu",
        gpu={"name": "Test", "tflops_fp16": tflops},
        memory={"total_gb": 8, "type": "GDDR6", "bandwidth_gbps": 100},
        supported_precisions=precisions,
    )


def _make_benchmark(device="test-device", precision="fp16", framework="tensorrt",
                    metric="fps", value=100):
    return BenchmarkRecord(
        benchmark_id="test",
        model="test-model",
        device=device,
        task="detection",
        precision=precision,
        framework=framework,
        metric=metric,
        value=value,
        source=BenchmarkSource(name="test", type="official_docs"),
    )


def test_cross_validate_unknown_device():
    devices = {"known": _make_device("known")}
    benchmark = _make_benchmark(device="unknown")
    result = validate_benchmark_against_device(benchmark, devices)
    assert not result.valid
    assert any("Unknown device" in e for e in result.errors)


def test_cross_validate_unsupported_precision():
    devices = {"dev": _make_device("dev", precisions=[Precision.FP32, Precision.FP16])}
    benchmark = _make_benchmark(device="dev", precision="int8")
    result = validate_benchmark_against_device(benchmark, devices)
    assert not result.valid
    assert any("does not support" in e for e in result.errors)


def test_cross_validate_gpu_framework_on_cpu_device():
    devices = {"cpu-dev": DeviceRecord(
        device_id="cpu-dev",
        name="CPU Only",
        type="sbc",
        gpu={"name": "None", "tflops_fp16": 0.0},
        memory={"total_gb": 8, "type": "DDR4", "bandwidth_gbps": 20},
        supported_precisions=[Precision.FP32],
    )}
    benchmark = _make_benchmark(device="cpu-dev", precision="fp32", framework="tensorrt")
    result = validate_benchmark_against_device(benchmark, devices)
    assert not result.valid
    assert any("requires GPU" in e for e in result.errors)


def test_cross_validate_warns_on_high_fps():
    devices = {"test-device": _make_device()}
    benchmark = _make_benchmark(value=15000)
    result = validate_benchmark_against_device(benchmark, devices)
    assert result.valid  # warning, not error
    assert any("Suspiciously high" in w for w in result.warnings)


def test_cross_validate_warns_missing_llm_setup():
    devices = {"test-device": _make_device()}
    benchmark = _make_benchmark(metric="tok/s", value=25)
    result = validate_benchmark_against_device(benchmark, devices)
    assert result.valid
    assert any("LLM test setup" in w for w in result.warnings)
