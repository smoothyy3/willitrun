"""Core estimation engine — Tier 1 lookup + Tier 2 FLOPs-based scaling."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from . import data_access
from ._data import data_dir
from .profiler import ModelProfile


@dataclass
class BenchmarkResult:
    """A single benchmark data point."""

    model: str
    device: str
    precision: str
    framework: str
    metric: str  # "fps" or "tok/s"
    value: float
    input_size: str | None = None  # e.g. "640x640", "tg128", "pp512"
    batch_size: int = 1
    source: str = ""
    notes: str = ""
    benchmark_id: str = ""  # canonical ID for dedup and type detection


@dataclass
class Estimate:
    """The final estimation result."""

    model_name: str
    device_name: str
    device_id: str

    # From profile
    parameters: int | None = None
    parameters_human: str = "Unknown"
    flops: int | None = None
    flops_human: str = "Unknown"
    model_type: str | None = None

    # Memory
    memory_by_precision: dict = field(default_factory=dict)  # {"fp16": "6.3 MB", ...}
    memory_bytes_by_precision: dict = field(default_factory=dict)  # {"fp16": 6400000, ...}

    # Device
    device_memory_gb: float = 0
    device_tflops_fp16: float = 0
    supported_precisions: list = field(default_factory=list)

    # Performance — primary metric (tg for LLMs, fps for vision)
    tier: int = 0  # 1 = real benchmark, 2 = estimated
    benchmark: BenchmarkResult | None = None  # Tier 1 primary result
    secondary_benchmark: BenchmarkResult | None = None  # Tier 1 secondary (pp512 when primary is tg128)
    estimated_fps: float | None = None
    estimated_fps_range: tuple[float, float] | None = None  # (low, high)
    metric: str = "fps"  # "fps" or "tok/s"

    # Fit check
    fits_fp32: bool | None = None
    fits_fp16: bool | None = None
    fits_int8: bool | None = None
    fits_4bit: bool | None = None

    # Scaling reference (for Tier 2)
    reference_benchmark: BenchmarkResult | None = None
    scaling_notes: str = ""

    # Data provenance — used by display to communicate confidence clearly
    model_source: str = "database"    # "database" | "huggingface" | "file" | "unknown"
    tier2_strategy: int = 0           # 1=FLOPs/same-device, 2=TFLOPS/same-model, 3=rough


def _load_benchmarks() -> list[BenchmarkResult]:
    """Load benchmarks from SQLite and project into legacy structure."""
    results = []
    for entry in data_access.list_benchmarks():
        bid = entry.benchmark_id
        # Recover input_size from benchmark_id if present
        input_size = None
        parts = bid.split("__")
        if len(parts) >= 5:
            candidate = parts[4]
            if not candidate.startswith("bs"):
                input_size = candidate

        # Map metric names back to legacy display values
        metric_map = {
            "tok_s_tg": "tok/s",
            "tok_s_pp": "tok/s",
            "samples_s": "samples/s",
        }
        metric = metric_map.get(entry.metric, entry.metric)

        results.append(BenchmarkResult(
            model=entry.model_id,
            device=entry.device_id,
            precision=entry.precision,
            framework=entry.framework,
            metric=metric,
            value=entry.value,
            input_size=input_size,
            batch_size=1,
            source=entry.source_name,
            notes=entry.notes or "",
            benchmark_id=bid,
        ))
    return results


def _load_devices() -> dict:
    return {d.device_id: d.model_dump() for d in data_access.list_devices()}


def _load_models_db() -> dict:
    return {m.model_id: m.model_dump() for m in data_access.list_models()}


def get_best_tflops(device_spec: dict) -> float:
    """Get the best available TFLOPS for inference scaling.

    Prefers FP16 (most ML inference runs FP16). Falls back to FP32 for GPUs
    where FP16 is throttled (e.g. Pascal consumer) or unknown.
    """
    gpu = device_spec.get("gpu", {})
    fp16 = gpu.get("tflops_fp16")
    fp32 = gpu.get("tflops_fp32")

    if fp16 and fp16 > 0:
        return fp16
    if fp32 and fp32 > 0:
        return fp32
    return 0


def _resolve_model_key(model_name: str, models_db: dict) -> str | None:
    """Find the canonical model key from a name or alias."""
    normalized = model_name.lower().strip()
    if model_name in models_db:
        return model_name
    for key, entry in models_db.items():
        aliases = [a.lower() for a in entry.get("aliases", [])]
        if normalized in aliases:
            return key
    if normalized.endswith(".pt"):
        return _resolve_model_key(normalized[:-3], models_db)
    return None


def _is_tg_benchmark(b: BenchmarkResult) -> bool:
    """True if this is a text-generation benchmark (tg128) rather than prompt-processing (pp512)."""
    if b.input_size and b.input_size.startswith("tg"):
        return True
    if "text generation" in b.notes.lower():
        return True
    return False


def _is_pp_benchmark(b: BenchmarkResult) -> bool:
    """True if this is a prompt-processing benchmark (pp512)."""
    if b.input_size and b.input_size.startswith("pp"):
        return True
    if "prompt processing" in b.notes.lower():
        return True
    return False


def _has_flash_attention(b: BenchmarkResult) -> bool:
    """True if this benchmark used Flash Attention (generally faster)."""
    if b.benchmark_id and "__fa" in b.benchmark_id and "__nofa" not in b.benchmark_id:
        return True
    if b.notes and "with flash attention" in b.notes.lower():
        return True
    return False


def _best_in_group(candidates: list[BenchmarkResult]) -> BenchmarkResult | None:
    """Pick the best benchmark from a group: prefer Flash Attention, then highest value."""
    if not candidates:
        return None
    fa = [b for b in candidates if _has_flash_attention(b)]
    pool = fa if fa else candidates
    return max(pool, key=lambda b: b.value)


def tier1_lookup(
    model_name: str,
    device_id: str,
    benchmarks: list[BenchmarkResult],
    models_db: dict,
) -> tuple[BenchmarkResult | None, BenchmarkResult | None]:
    """Tier 1: exact match in benchmark database.

    Returns (primary, secondary):
    - LLMs:   primary = tg128 (text generation), secondary = pp512 (prompt processing)
    - Vision: primary = best fps benchmark, secondary = None

    For LLMs, text generation speed is what users experience during chat.
    Prompt processing (pp) is up to 30× faster and would be misleading as a headline.
    """
    model_key = _resolve_model_key(model_name, models_db) or model_name
    model_type = models_db.get(model_key, {}).get("model_type", "")

    candidates = [
        b for b in benchmarks
        if (_resolve_model_key(b.model, models_db) or b.model) == model_key
        and b.device == device_id
    ]

    if not candidates:
        return None, None

    if model_type == "llm":
        tg = [b for b in candidates if _is_tg_benchmark(b)]
        pp = [b for b in candidates if _is_pp_benchmark(b)]

        primary = _best_in_group(tg) or candidates[0]
        secondary = _best_in_group(pp)
        return primary, secondary

    # Non-LLM: return best single match
    return _best_in_group(candidates) or candidates[0], None


def tier2_estimate(
    profile: ModelProfile,
    device_id: str,
    benchmarks: list[BenchmarkResult],
    devices: dict,
    models_db: dict,
) -> tuple[float | None, tuple[float, float] | None, BenchmarkResult | None, str, int]:
    """Tier 2: estimate performance by scaling from nearest known benchmark.

    Strategy:
    1. Find benchmarks on the same device → scale by FLOPs ratio
    2. Find benchmarks of the same model on different device → scale by TFLOPS ratio
    3. Find any benchmark on same device → scale by FLOPs ratio (least accurate)

    For LLMs, only tg128 (text generation) benchmarks are used as references —
    pp512 values are 10-30× higher and would produce wildly inflated estimates.

    Returns: (estimated_value, (low, high), reference_benchmark, notes, strategy)
    where strategy: 1=FLOPs/same-device, 2=TFLOPS/same-model, 3=rough
    """
    if profile.flops is None and profile.parameters is None:
        return None, None, None, "Insufficient model info for estimation", 0

    device_spec = devices.get(device_id)
    if device_spec is None:
        return None, None, None, f"Unknown device: {device_id}", 0

    is_llm = (profile.model_type == "llm")
    device_tflops = get_best_tflops(device_spec)

    def _is_valid_ref(b: BenchmarkResult) -> bool:
        """For LLMs, only use tg benchmarks as scaling references."""
        if is_llm and _is_pp_benchmark(b):
            return False
        return True

    # Strategy 1a: Same device, LLM → scale by parameter ratio (memory-bandwidth bound)
    # LLM text-generation speed ≈ BW / (2 × param_bytes).  Because same device = same
    # bandwidth, scaling by param ratio is the most reliable cross-model estimate.
    if is_llm and profile.parameters:
        llm_refs = []
        for b in benchmarks:
            if b.device != device_id:
                continue
            if not _is_valid_ref(b):
                continue
            ref_key = _resolve_model_key(b.model, models_db)
            if ref_key is None:
                continue
            ref_entry = models_db.get(ref_key, {})
            if ref_entry.get("model_type") != "llm":
                continue
            ref_params = ref_entry.get("parameters")
            if not ref_params:
                continue
            llm_refs.append((b, ref_params))

        if llm_refs:
            # Use the benchmark whose model is closest in parameter count
            llm_refs.sort(key=lambda x: abs(x[1] - profile.parameters))
            ref_bench, ref_params = llm_refs[0]
            scale = ref_params / profile.parameters
            est = ref_bench.value * scale
            low = est * 0.7
            high = est * 1.3
            notes = (
                f"Scaled from {ref_bench.model} ({ref_bench.value:.0f} {ref_bench.metric}) "
                f"on same device by parameter ratio ({ref_params/1e9:.1f}B / {profile.parameters/1e9:.1f}B)"
            )
            return est, (low, high), ref_bench, notes, 1

    # Strategy 1b: Same device, vision model → scale by FLOPs ratio
    if profile.flops:
        best_ref = None
        best_ratio = float("inf")

        for b in benchmarks:
            if b.device != device_id:
                continue
            if not _is_valid_ref(b):
                continue
            ref_key = _resolve_model_key(b.model, models_db)
            if ref_key is None:
                continue
            ref_entry = models_db.get(ref_key, {})
            ref_flops = ref_entry.get("flops")
            if ref_flops is None:
                continue

            ratio = abs(1 - ref_flops / profile.flops) if profile.flops else float("inf")
            if ratio < best_ratio:
                best_ratio = ratio
                best_ref = (b, ref_flops)

        if best_ref:
            ref_bench, ref_flops = best_ref
            scale = ref_flops / profile.flops
            est = ref_bench.value * scale
            low = est * 0.7
            high = est * 1.3
            notes = (
                f"Scaled from {ref_bench.model} ({ref_bench.value:.0f} {ref_bench.metric}) "
                f"on same device by FLOPs ratio ({ref_flops/1e9:.1f}G / {profile.flops/1e9:.1f}G)"
            )
            return est, (low, high), ref_bench, notes, 1

    # Strategy 2: Same model, different device → scale by TFLOPS
    model_key = profile.name
    for b in benchmarks:
        if not _is_valid_ref(b):
            continue
        b_key = _resolve_model_key(b.model, models_db) or b.model
        if b_key != model_key:
            continue
        ref_device = devices.get(b.device)
        if ref_device is None:
            continue
        ref_tflops = get_best_tflops(ref_device)
        if ref_tflops == 0 or device_tflops == 0:
            continue

        scale = device_tflops / ref_tflops
        ref_bw = ref_device.get("memory", {}).get("bandwidth_gbps") or 1
        dev_bw = device_spec.get("memory", {}).get("bandwidth_gbps") or 1
        bw_factor = min(dev_bw / ref_bw, scale)
        combined_scale = (scale + bw_factor) / 2

        est = b.value * combined_scale
        low = est * 0.6
        high = est * 1.4
        notes = (
            f"Scaled from {b.device} ({b.value:.0f} {b.metric}) "
            f"by TFLOPS ratio ({device_tflops:.1f} / {ref_tflops:.1f}) "
            f"and bandwidth ratio ({dev_bw:.0f} / {ref_bw:.0f} GB/s)"
        )
        return est, (low, high), b, notes, 2

    # Strategy 3: Same device, any model → very rough scale by FLOPs
    if profile.flops:
        for b in benchmarks:
            if b.device != device_id:
                continue
            if not _is_valid_ref(b):
                continue
            ref_key = _resolve_model_key(b.model, models_db)
            if ref_key and models_db.get(ref_key, {}).get("flops"):
                ref_flops = models_db[ref_key]["flops"]
                scale = ref_flops / profile.flops
                est = b.value * scale
                low = est * 0.5
                high = est * 1.5
                notes = (
                    f"Rough estimate scaled from {b.model} on same device. "
                    f"Different model architecture — treat with caution."
                )
                return est, (low, high), b, notes, 3

    return None, None, None, "No suitable reference benchmark found", 0


def estimate(profile: ModelProfile, device_id: str) -> Estimate:
    """Run the full estimation pipeline for a model on a device."""
    benchmarks = _load_benchmarks()
    devices = _load_devices()
    models_db = _load_models_db()

    device_spec = devices.get(device_id)
    if device_spec is None:
        return Estimate(
            model_name=profile.name,
            device_name=f"Unknown ({device_id})",
            device_id=device_id,
            scaling_notes=f"Unknown device: {device_id}. Use --list-devices to see available devices.",
        )

    device_name = device_spec.get("name", device_id)
    device_mem = device_spec.get("memory", {}).get("total_gb") or 0
    device_tflops = get_best_tflops(device_spec)
    supported = device_spec.get("supported_precisions", [])

    # Build memory info
    memory_human = {}
    memory_bytes = {}
    for prec in ["fp32", "fp16", "int8", "4bit"]:
        mem_str = profile.memory_human(prec)
        if mem_str != "Unknown":
            memory_human[prec] = mem_str
        mem_val = profile._memory_for_precision(prec)
        if mem_val is not None:
            memory_bytes[prec] = mem_val

    # Fit checks
    def fits(prec: str) -> bool | None:
        mem = memory_bytes.get(prec)
        if mem is None or device_mem == 0:
            return None
        # Add ~20% overhead for runtime buffers
        total_needed_gb = (mem * 1.2) / (1024**3)
        return total_needed_gb <= device_mem

    result = Estimate(
        model_name=profile.name,
        device_name=device_name,
        device_id=device_id,
        parameters=profile.parameters,
        parameters_human=profile.parameters_human,
        flops=profile.flops,
        flops_human=profile.flops_human,
        model_type=profile.model_type,
        memory_by_precision=memory_human,
        memory_bytes_by_precision=memory_bytes,
        device_memory_gb=device_mem,
        device_tflops_fp16=device_tflops,
        supported_precisions=supported,
        fits_fp32=fits("fp32"),
        fits_fp16=fits("fp16"),
        fits_int8=fits("int8"),
        fits_4bit=fits("4bit"),
        model_source=getattr(profile, "source", "database") or "database",
    )

    # Tier 1: exact lookup
    bench, secondary = tier1_lookup(profile.name, device_id, benchmarks, models_db)
    if bench:
        result.tier = 1
        result.benchmark = bench
        result.secondary_benchmark = secondary
        result.estimated_fps = bench.value
        result.metric = bench.metric
        return result

    # Tier 2: scaling estimate
    est_val, est_range, ref_bench, notes, strategy = tier2_estimate(
        profile, device_id, benchmarks, devices, models_db
    )
    result.tier = 2
    result.estimated_fps = est_val
    result.estimated_fps_range = est_range
    result.reference_benchmark = ref_bench
    result.scaling_notes = notes
    result.tier2_strategy = strategy
    if ref_bench:
        result.metric = ref_bench.metric

    return result
