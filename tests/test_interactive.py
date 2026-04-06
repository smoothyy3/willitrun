"""Tests for the interactive UI helpers and end-to-end slug resolution.

Parametrized over every model in models.yaml so that:
  - Any future model entry that breaks slug formatting is caught immediately.
  - The slug-extraction / database-resolution path that was previously broken
    for HuggingFace-namespaced models (e.g. deepseek-ai/DeepSeek-R1-Distill-*)
    is continuously verified.
"""

from __future__ import annotations

import pytest

from willitrun._data import load_models
from willitrun.interactive import (
    _device_choices,
    _model_choices,
    _slug_from_choice,
)
from willitrun.loader import resolve_from_database


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def models_db():
    db = load_models()
    assert db, "models.yaml is empty or missing"
    return db


@pytest.fixture(scope="module")
def model_display_map(models_db):
    """Map each display string → original slug for parametrized tests."""
    choices = _model_choices(models_db)
    return {choice: slug for choice, slug in zip(choices, sorted(models_db))}


# ---------------------------------------------------------------------------
# Slug formatting + extraction
# ---------------------------------------------------------------------------

def test_model_choices_count(models_db):
    """_model_choices() must return one entry per model."""
    choices = _model_choices(models_db)
    assert len(choices) == len(models_db)


def test_model_choices_no_empty_string(models_db):
    """No display string should be blank."""
    for choice in _model_choices(models_db):
        assert choice.strip(), f"Empty display string in model choices"


@pytest.mark.parametrize("slug", sorted(load_models().keys()))
def test_slug_roundtrip(slug, models_db):
    """Format a model as a padded display string, then extract slug back.

    This is the exact code path followed when the user selects an item from
    the dropdown.  Any mismatch here means the resolver would receive a padded
    string (e.g. 'llama-3.2-1b   1.2B   llm') instead of the clean slug.
    """
    choices = _model_choices(models_db)
    # Find the display string that corresponds to this slug
    matching = [c for c in choices if c.split()[0] == slug]
    assert matching, f"No display string found for slug '{slug}'"
    display_str = matching[0]

    # Simulate the extraction logic from interactive.py
    # (path check not needed since model slugs are never file paths)
    extracted = _slug_from_choice(display_str) if " " in display_str else display_str
    assert extracted == slug, (
        f"Slug roundtrip failed: '{display_str}' → '{extracted}' (expected '{slug}')"
    )


@pytest.mark.parametrize("slug", sorted(load_models().keys()))
def test_database_resolution(slug):
    """Every slug in models.yaml must resolve via resolve_from_database()."""
    info = resolve_from_database(slug)
    assert info is not None, (
        f"resolve_from_database('{slug}') returned None — "
        f"slug is in models.yaml but cannot be resolved"
    )
    assert info.source == "database", (
        f"Expected source='database' for '{slug}', got '{info.source}'"
    )
    assert info.name == slug, (
        f"Resolved name '{info.name}' does not match requested slug '{slug}'"
    )


@pytest.mark.parametrize("slug", sorted(load_models().keys()))
def test_database_resolution_after_display_formatting(slug, models_db):
    """Simulate selecting a model from the interactive dropdown end-to-end.

    Verifies the full path: format → user selects padded string → extract slug
    → resolve from database.  This is the regression test for the bug where
    HF-namespaced slugs like 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B' would
    not resolve because the padded string bypassed _slug_from_choice().
    """
    choices = _model_choices(models_db)
    matching = [c for c in choices if c.split()[0] == slug]
    assert matching, f"No display string found for slug '{slug}'"
    display_str = matching[0]

    # Replicate interactive.py extraction logic
    from pathlib import Path as _Path
    raw = display_str.strip()
    if raw.endswith((".pt", ".onnx", ".bin")) or _Path(raw).exists():
        model_str = raw
    elif " " in raw:
        model_str = _slug_from_choice(raw)
    else:
        model_str = raw

    info = resolve_from_database(model_str)
    assert info is not None, (
        f"Database resolution failed after display formatting for '{slug}': "
        f"display='{display_str}' → extracted='{model_str}'"
    )
    assert info.source == "database"


# ---------------------------------------------------------------------------
# Device choices
# ---------------------------------------------------------------------------

from willitrun._data import load_devices  # noqa: E402


def test_device_choices_count():
    devices = load_devices()
    choices = _device_choices(devices)
    assert len(choices) == len(devices)


@pytest.mark.parametrize("device_id", sorted(load_devices().keys()))
def test_device_slug_roundtrip(device_id):
    """Device IDs must survive the same format→extract roundtrip."""
    devices = load_devices()
    choices = _device_choices(devices)
    matching = [c for c in choices if c.split()[0] == device_id]
    assert matching, f"No display string for device '{device_id}'"
    extracted = _slug_from_choice(matching[0]) if " " in matching[0] else matching[0]
    assert extracted == device_id
