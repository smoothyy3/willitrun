import pytest

from willitrun.estimator import (
    BenchmarkResult,
    tier2_estimate,
)
from willitrun.profiler import ModelProfile


def make_profile(**kwargs):
    defaults = dict(
        name="dummy",
        parameters=None,
        flops=None,
        model_type="vision",
        architecture=None,
        default_input_size=None,
        llm_config=None,
        memory_fp32=None,
        memory_fp16=None,
        memory_fp8=None,
        memory_int8=None,
        memory_4bit=None,
        source="database",
    )
    defaults.update(kwargs)
    return ModelProfile(**defaults)


def test_tier2_requires_model_info():
    profile = make_profile()
    est, rng, ref, note, strat = tier2_estimate(profile, "dev", [], {}, {})
    assert est is None and rng is None and ref is None
    assert "Insufficient model info" in note
    assert strat == 0


def test_tier2_zero_tflops_device():
    profile = make_profile(flops=1_000_000_000, parameters=1_000_000, model_type="vision")
    devices = {"dev": {"memory": {"total_gb": 8}, "gpu": {}, "supported_precisions": []}}
    est, rng, ref, note, strat = tier2_estimate(profile, "dev", [], devices, {})
    assert est is None and rng is None and ref is None
    assert "TFLOPS" in note
    assert strat == 0


def test_tier2_clamps_extreme_scaling():
    # Reference 1 GFLOP, target 1 TFLOP on same device → raw scale 0.001 → clamp to 0.1
    profile = make_profile(flops=1_000_000_000_000, parameters=10_000_000, model_type="vision")
    devices = {"dev": {"memory": {"total_gb": 8}, "gpu": {"tflops_fp16": 10}, "supported_precisions": []}}
    models_db = {"ref": {"flops": 1_000_000_000, "model_type": "vision"}}
    bench = BenchmarkResult(
        model="ref",
        device="dev",
        precision="fp16",
        framework="tensorrt",
        metric="fps",
        value=100.0,
        input_size=None,
        batch_size=1,
        source="",
        notes="",
        benchmark_id="ref__dev",
    )
    est, rng, ref, note, strat = tier2_estimate(profile, "dev", [bench], devices, models_db)
    assert strat == 1
    assert est == pytest.approx(10.0, rel=1e-6)  # 100 * clamp(0.001 -> 0.1)
    assert rng[0] < est < rng[1]
    assert "FLOPs ratio" in note
