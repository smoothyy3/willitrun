"""Tests for model loading and resolution."""

from willitrun.loader import ModelInfo, resolve_from_database, resolve_model


def test_resolve_known_model_by_name():
    info = resolve_from_database("yolov8n")
    assert info is not None
    assert info.name == "yolov8n"
    assert info.model_type == "detection"
    assert info.source == "database"


def test_resolve_known_model_by_alias():
    info = resolve_from_database("yolov8n.pt")
    assert info is not None
    assert info.name == "yolov8n"


def test_resolve_known_model_case_insensitive():
    info = resolve_from_database("ResNet50")
    assert info is not None
    assert info.name == "resnet50"


def test_resolve_unknown_model():
    info = resolve_from_database("nonexistent-model-xyz")
    assert info is None


def test_resolve_llm_model():
    info = resolve_from_database("meta-llama/Llama-3-8B")
    assert info is not None
    assert info.model_type == "llm"
    assert info.architecture == "llama"


def test_resolve_model_fallback():
    """Unknown model returns minimal ModelInfo."""
    info = resolve_model("totally-unknown-model")
    assert info is not None
    assert info.name == "totally-unknown-model"
    assert info.source == "unknown"
    assert info.parameters is None
