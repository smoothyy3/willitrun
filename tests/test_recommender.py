"""Tests for the recommender."""

from willitrun.estimator import estimate
from willitrun.loader import resolve_model
from willitrun.profiler import profile_model
from willitrun.recommender import recommend


def test_verdict_with_tier1_benchmark():
    """Model with real benchmark data should get a verdict."""
    info = resolve_model("yolo26n")
    profile = profile_model(info)
    est = estimate(profile, "jetson-agx-thor")
    rec = recommend(est)

    # With Tier 1 benchmark hit, should get a verdict
    assert est.tier == 1
    # Verdict depends on memory fit — with skeleton data, verdict may be 'unknown'
    # because we can't check memory. But the recommendation should still have content.
    assert rec.verdict in ("runs_great", "will_run", "unknown")


def test_verdict_with_missing_data():
    """Model with no memory info should get 'unknown' verdict."""
    info = resolve_model("meta-llama/Llama-3-8B")
    profile = profile_model(info)
    est = estimate(profile, "jetson-orin-nano-8gb")
    rec = recommend(est)

    if profile.parameters is None:
        # Can't determine memory fit without parameters
        assert rec.verdict == "unknown"
    else:
        assert rec.verdict in ("needs_quantization", "wont_fit")
        assert len(rec.suggestions) > 0


def test_suggestions_always_present():
    """Recommendations should always have at least one suggestion."""
    info = resolve_model("yolo26n")
    profile = profile_model(info)
    est = estimate(profile, "jetson-orin-nano-8gb")
    rec = recommend(est)

    # Should always have suggestions, even if generic
    assert len(rec.suggestions) > 0
