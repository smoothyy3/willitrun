"""Tests for the estimation engine."""

import pytest

from willitrun.estimator import estimate
from willitrun.loader import resolve_model
from willitrun.profiler import profile_model


def test_tier1_exact_match():
    """Known model + known device = Tier 1 real benchmark."""
    info = resolve_model("yolo26n")
    profile = profile_model(info)
    est = estimate(profile, "jetson-orin-nano-8gb")

    assert est.tier == 1
    assert est.benchmark is not None
    assert est.benchmark.value > 0
    assert est.metric == "fps"
    assert est.model_name == "yolo26n"
    assert est.device_name == "NVIDIA Jetson Orin Nano 8GB"


def test_tier1_llm():
    """LLM benchmark lookup from real NVIDIA data."""
    info = resolve_model("meta-llama/Llama-3.1-8B")
    profile = profile_model(info)
    est = estimate(profile, "jetson-agx-thor")

    assert est.tier == 1
    assert est.benchmark is not None
    assert est.metric == "tok/s"
    assert est.benchmark.value > 0


def test_tier1_llama_cpp():
    """llama.cpp benchmark lookup from CUDA scoreboard data."""
    info = resolve_model("llama-2-7b")
    profile = profile_model(info)
    est = estimate(profile, "rtx-4090-24gb")

    assert est.tier == 1
    assert est.benchmark is not None
    assert est.metric == "tok/s"
    assert est.benchmark.value > 0


def test_tier2_estimation():
    """Unknown combo should fall back to Tier 2 estimation.

    Note: Tier 2 requires FLOPs in models.yaml and TFLOPS in devices.yaml.
    With skeleton data (no numeric specs), Tier 2 returns None gracefully.
    """
    info = resolve_model("yolov8n")
    profile = profile_model(info)
    est = estimate(profile, "jetson-orin-nano-8gb")

    assert est.tier == 2
    # With skeleton data, Tier 2 can't estimate — that's correct behavior
    if profile.flops is None:
        assert est.estimated_fps is None
        assert "Insufficient model info" in est.scaling_notes
    else:
        assert est.estimated_fps is not None
        assert est.estimated_fps > 0


def test_memory_fit_check():
    """Memory fit requires both model params and device memory.

    With skeleton data, fits_* are all None (unknown).
    """
    info = resolve_model("meta-llama/Llama-3-8B")
    profile = profile_model(info)
    est = estimate(profile, "jetson-orin-nano-8gb")

    if profile.parameters is not None and est.device_memory_gb:
        assert est.fits_fp16 is False
        assert est.fits_4bit is True
    else:
        # With skeleton data, can't determine fit
        assert est.fits_fp16 is None
        assert est.fits_4bit is None


def test_unknown_device():
    """Unknown device should return graceful error."""
    info = resolve_model("yolo26n")
    profile = profile_model(info)
    est = estimate(profile, "nonexistent-device")

    assert "Unknown device" in est.scaling_notes


def test_fits_small_model():
    """Small model on large device should fit at all precisions.

    With skeleton data (no params/memory), fits are None.
    """
    info = resolve_model("yolo26n")
    profile = profile_model(info)
    est = estimate(profile, "jetson-agx-thor")

    if profile.parameters is not None and est.device_memory_gb:
        assert est.fits_fp32 is True
        assert est.fits_fp16 is True
    else:
        # Tier 1 still works even without memory info
        assert est.tier == 1
        assert est.benchmark is not None
