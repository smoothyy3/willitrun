"""Recommender — generates verdicts and actionable suggestions."""

from __future__ import annotations

from dataclasses import dataclass

from .estimator import Estimate


@dataclass
class Recommendation:
    """Final recommendation for the user."""

    verdict: str  # "runs_great", "runs", "needs_quantization", "tight_fit", "wont_fit", "unknown"
    verdict_emoji: str
    verdict_text: str
    suggestions: list[str]
    best_precision: str | None = None


def recommend(est: Estimate) -> Recommendation:
    """Generate a recommendation from an estimation result."""
    suggestions = []

    # Determine best precision that fits
    best_precision = None
    for prec in ["fp16", "int8", "4bit"]:
        fit = getattr(est, f"fits_{prec}", None)
        if fit is True:
            best_precision = prec
            break

    # If none of the reduced precisions fit, check fp32
    if best_precision is None and est.fits_fp32 is True:
        best_precision = "fp32"

    # Special case: no fit info available
    if all(
        getattr(est, f"fits_{p}", None) is None
        for p in ["fp32", "fp16", "int8", "4bit"]
    ):
        return Recommendation(
            verdict="unknown",
            verdict_emoji="?",
            verdict_text="Insufficient data to determine fit",
            suggestions=["Could not determine model memory requirements."],
            best_precision=None,
        )

    # Determine verdict — thresholds differ between model types
    has_perf = est.estimated_fps is not None
    fps = est.estimated_fps or 0
    is_llm = est.model_type == "llm"

    # LLM thresholds (tok/s, text generation):
    #   ≥ 15 tok/s = great (interactive, faster than most people read)
    #   ≥ 5  tok/s = usable (slow but workable)
    # Vision thresholds (fps):
    #   ≥ 30 fps = great (real-time)
    #   ≥ 10 fps = usable (near real-time)
    great_threshold = 15 if is_llm else 30
    ok_threshold = 5 if is_llm else 10

    if est.fits_fp16 is True:
        if has_perf and fps >= great_threshold:
            return Recommendation(
                verdict="runs_great",
                verdict_emoji="RUNS_GREAT",
                verdict_text="Runs great",
                suggestions=_build_suggestions(est, best_precision),
                best_precision=best_precision,
            )
        elif has_perf and fps >= ok_threshold:
            slow_note = "slower than real-time" if not is_llm else "usable but slower than ideal"
            return Recommendation(
                verdict="runs",
                verdict_emoji="RUNS",
                verdict_text=f"Runs (but may be {slow_note})",
                suggestions=_build_suggestions(est, best_precision),
                best_precision=best_precision,
            )
        else:
            return Recommendation(
                verdict="runs",
                verdict_emoji="RUNS",
                verdict_text="Fits in memory",
                suggestions=_build_suggestions(est, best_precision),
                best_precision=best_precision,
            )

    elif est.fits_fp16 is False and (est.fits_int8 is True or est.fits_4bit is True):
        quant = "INT8" if est.fits_int8 else "4-bit"
        return Recommendation(
            verdict="needs_quantization",
            verdict_emoji="NEEDS_QUANT",
            verdict_text="Needs quantization",
            suggestions=[
                f"FP16 model ({est.memory_by_precision.get('fp16', '?')}) exceeds device memory ({est.device_memory_gb} GB).",
                f"Use {quant} quantization to fit ({est.memory_by_precision.get('int8' if est.fits_int8 else '4bit', '?')}).",
            ]
            + _build_suggestions(est, best_precision),
            best_precision="int8" if est.fits_int8 else "4bit",
        )

    elif est.fits_fp32 is True and est.fits_fp16 is not True:
        return Recommendation(
            verdict="runs",
            verdict_emoji="RUNS",
            verdict_text="Runs at FP32",
            suggestions=_build_suggestions(est, "fp32"),
            best_precision="fp32",
        )

    elif all(
        getattr(est, f"fits_{p}", None) is False
        for p in ["fp32", "fp16", "int8", "4bit"]
    ):
        return Recommendation(
            verdict="wont_fit",
            verdict_emoji="WONT_FIT",
            verdict_text="Won't fit",
            suggestions=[
                f"Model is too large for this device even at 4-bit ({est.memory_by_precision.get('4bit', '?')}) "
                f"with only {est.device_memory_gb} GB available.",
                "Consider a smaller model variant or a device with more memory.",
            ],
            best_precision=None,
        )

    else:
        # Tight fit or uncertain
        return Recommendation(
            verdict="tight_fit",
            verdict_emoji="TIGHT",
            verdict_text="Tight fit",
            suggestions=[
                "Model may fit but with limited headroom for runtime buffers.",
            ]
            + _build_suggestions(est, best_precision),
            best_precision=best_precision,
        )


def _build_suggestions(est: Estimate, best_precision: str | None) -> list[str]:
    """Build contextual suggestions."""
    suggestions = []

    if est.model_type == "llm" and best_precision == "4bit":
        suggestions.append(
            "For LLMs, 4-bit GGUF (Q4_K_M) via llama.cpp gives the best size/quality tradeoff."
        )
    elif est.model_type in ("detection", "segmentation"):
        if "int8" in est.supported_precisions:
            suggestions.append(
                "For edge deployment, TensorRT with INT8 calibration gives the best FPS."
            )
        elif "fp16" in est.supported_precisions:
            suggestions.append(
                "Use TensorRT FP16 for best inference speed on NVIDIA devices."
            )

    if est.device_id and "jetson" in est.device_id:
        suggestions.append("Use JetPack SDK + TensorRT for optimized Jetson inference.")

    if est.device_id and "apple" in est.device_id:
        suggestions.append("Use CoreML or MPS backend for best Apple Silicon performance.")

    if est.device_id and "rpi" in est.device_id:
        suggestions.append(
            "Use NCNN or TF Lite for best CPU-only performance on Raspberry Pi."
        )

    return suggestions
