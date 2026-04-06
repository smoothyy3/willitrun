"""Tests for model profiling."""

from willitrun.loader import ModelInfo
from willitrun.profiler import ModelProfile, profile_model


def test_profile_basic():
    info = ModelInfo(
        name="test-model",
        parameters=1_000_000,
        flops=2_000_000_000,
        model_type="classification",
    )
    profile = profile_model(info)

    assert profile.parameters == 1_000_000
    assert profile.flops == 2_000_000_000
    assert profile.memory_fp32 == 4_000_000
    assert profile.memory_fp16 == 2_000_000
    assert profile.memory_int8 == 1_000_000
    assert profile.memory_4bit == 500_000


def test_profile_parameters_human():
    assert ModelProfile(name="a", parameters=3_200_000, flops=None, model_type=None,
                        architecture=None, default_input_size=None, llm_config=None).parameters_human == "3.2M"
    assert ModelProfile(name="a", parameters=8_000_000_000, flops=None, model_type=None,
                        architecture=None, default_input_size=None, llm_config=None).parameters_human == "8.0B"
    assert ModelProfile(name="a", parameters=1_500, flops=None, model_type=None,
                        architecture=None, default_input_size=None, llm_config=None).parameters_human == "1.5K"
    assert ModelProfile(name="a", parameters=None, flops=None, model_type=None,
                        architecture=None, default_input_size=None, llm_config=None).parameters_human == "Unknown"


def test_profile_flops_human():
    assert ModelProfile(name="a", parameters=None, flops=8_700_000_000, model_type=None,
                        architecture=None, default_input_size=None, llm_config=None).flops_human == "8.7 GFLOPs"
    assert ModelProfile(name="a", parameters=None, flops=2_734_000_000_000, model_type=None,
                        architecture=None, default_input_size=None, llm_config=None).flops_human == "2.7 TFLOPs"


def test_profile_memory_human():
    info = ModelInfo(name="test", parameters=3_200_000)
    profile = profile_model(info)
    assert profile.memory_human("fp16") == "6.1 MB"
    assert profile.memory_human("fp32") == "12.2 MB"


def test_profile_no_params():
    info = ModelInfo(name="test", parameters=None)
    profile = profile_model(info)
    assert profile.memory_fp32 is None
    assert profile.memory_human("fp16") == "Unknown"
