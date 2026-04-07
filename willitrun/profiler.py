"""Model profiling — extract parameters, FLOPs, and memory footprint."""

from __future__ import annotations

from dataclasses import dataclass

from .loader import ModelInfo


@dataclass
class ModelProfile:
    """Profiled model characteristics."""

    name: str
    parameters: int | None       # total params — drives memory estimates
    flops: int | None
    model_type: str | None
    architecture: str | None
    default_input_size: str | None
    llm_config: dict | None

    # MoE — active params per forward pass (None for dense models)
    active_params: int | None = None
    is_moe: bool = False

    # Computed memory estimates (bytes) — always based on total params
    memory_fp32: int | None = None
    memory_fp16: int | None = None
    memory_fp8: int | None = None
    memory_int8: int | None = None
    memory_4bit: int | None = None

    source: str = "unknown"

    @property
    def parameters_human(self) -> str:
        if self.parameters is None:
            return "Unknown"
        if self.parameters >= 1_000_000_000:
            return f"{self.parameters / 1_000_000_000:.1f}B"
        if self.parameters >= 1_000_000:
            return f"{self.parameters / 1_000_000:.1f}M"
        if self.parameters >= 1_000:
            return f"{self.parameters / 1_000:.1f}K"
        return str(self.parameters)

    @property
    def flops_human(self) -> str:
        if self.flops is None:
            return "Unknown"
        if self.flops >= 1_000_000_000_000:
            return f"{self.flops / 1_000_000_000_000:.1f} TFLOPs"
        if self.flops >= 1_000_000_000:
            return f"{self.flops / 1_000_000_000:.1f} GFLOPs"
        if self.flops >= 1_000_000:
            return f"{self.flops / 1_000_000:.1f} MFLOPs"
        return f"{self.flops} FLOPs"

    def memory_human(self, precision: str) -> str:
        mem = self._memory_for_precision(precision)
        if mem is None:
            return "Unknown"
        return _bytes_human(mem)

    def _memory_for_precision(self, precision: str) -> int | None:
        mapping = {
            "fp32": self.memory_fp32,
            "fp16": self.memory_fp16,
            "bf16": self.memory_fp16,
            "fp8": self.memory_fp8,
            "int8": self.memory_int8,
            "4bit": self.memory_4bit,
            "int4": self.memory_4bit,
        }
        return mapping.get(precision)


def _bytes_human(n: int) -> str:
    if n >= 1024**3:
        return f"{n / 1024**3:.1f} GB"
    if n >= 1024**2:
        return f"{n / 1024**2:.1f} MB"
    if n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n} B"


def profile_model(model_info: ModelInfo) -> ModelProfile:
    """Build a complete profile from ModelInfo."""
    params = model_info.parameters  # total params — drives memory

    # Memory is always based on total params (all weights reside in device memory)
    memory_fp32 = params * 4 if params else None
    memory_fp16 = params * 2 if params else None
    memory_fp8 = params * 1 if params else None
    memory_int8 = params * 1 if params else None
    memory_4bit = int(params * 0.5) if params else None

    return ModelProfile(
        name=model_info.name,
        parameters=params,
        active_params=getattr(model_info, "active_params", None),
        is_moe=getattr(model_info, "is_moe", False),
        flops=model_info.flops,
        model_type=model_info.model_type,
        architecture=model_info.architecture,
        default_input_size=model_info.default_input_size,
        llm_config=model_info.llm_config,
        memory_fp32=memory_fp32,
        memory_fp16=memory_fp16,
        memory_fp8=memory_fp8,
        memory_int8=memory_int8,
        memory_4bit=memory_4bit,
        source=model_info.source,
    )
