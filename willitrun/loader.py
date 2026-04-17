"""Model loading abstraction — resolves model names, files, and HF identifiers."""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class ModelInfo:
    """Normalized model information from any source."""

    name: str
    parameters: int | None = None       # total parameters (use for memory)
    active_params: int | None = None    # active params per forward pass (MoE only; use for speed scaling)
    is_moe: bool = False
    flops: int | None = None
    default_input_size: str | None = None
    model_type: str | None = None  # classification, detection, segmentation, llm
    architecture: str | None = None
    weights_size_mb: float | None = None
    llm_config: dict | None = None
    source: str = "unknown"  # "database", "file", "huggingface"
    extra: dict = field(default_factory=dict)


def _load_models_db() -> dict:
    from ._data import load_models_raw  # noqa: PLC0415
    return load_models_raw()


def resolve_from_database(model_input: str) -> ModelInfo | None:
    """Try to resolve a model name from the built-in models database."""
    db = _load_models_db()
    normalized = model_input.lower().strip()

    # Direct key match
    if model_input in db:
        return _model_info_from_db(model_input, db[model_input])

    # Alias match
    for key, entry in db.items():
        aliases = [a.lower() for a in entry.get("aliases", [])]
        if normalized in aliases:
            return _model_info_from_db(key, entry)

    # Strip .pt extension and retry
    if normalized.endswith(".pt"):
        return resolve_from_database(normalized[:-3])

    return None


def _model_info_from_db(key: str, entry: dict) -> ModelInfo:
    return ModelInfo(
        name=key,
        parameters=entry.get("parameters"),
        flops=entry.get("flops"),
        default_input_size=entry.get("default_input_size"),
        model_type=entry.get("model_type"),
        architecture=entry.get("architecture"),
        weights_size_mb=entry.get("weights_size_mb"),
        llm_config=entry.get("llm_config"),
        source="database",
    )


def resolve_from_file(path: str) -> ModelInfo | None:
    """Try to load model info from a local file (.pt or .onnx)."""
    p = Path(path)
    if not p.exists():
        return None

    if p.suffix == ".pt":
        return _load_pytorch_file(p)
    elif p.suffix == ".onnx":
        return _load_onnx_file(p)

    return None


def _load_pytorch_file(path: Path) -> ModelInfo | None:
    try:
        import torch
    except ImportError:
        return ModelInfo(
            name=path.stem,
            weights_size_mb=path.stat().st_size / (1024 * 1024),
            source="file",
            extra={"path": str(path), "note": "Install torch for detailed profiling"},
        )

    try:
        model = torch.load(path, map_location="cpu", weights_only=False)
        # Handle Ultralytics checkpoint format
        if isinstance(model, dict) and "model" in model:
            model = model["model"]

        if hasattr(model, "parameters"):
            params = sum(p.numel() for p in model.parameters())
        else:
            params = None

        return ModelInfo(
            name=path.stem,
            parameters=params,
            weights_size_mb=path.stat().st_size / (1024 * 1024),
            source="file",
            extra={"path": str(path), "torch_model": model},
        )
    except Exception:
        return ModelInfo(
            name=path.stem,
            weights_size_mb=path.stat().st_size / (1024 * 1024),
            source="file",
            extra={"path": str(path), "note": "Could not load model"},
        )


def _load_onnx_file(path: Path) -> ModelInfo | None:
    try:
        import onnx
    except ImportError:
        return ModelInfo(
            name=path.stem,
            weights_size_mb=path.stat().st_size / (1024 * 1024),
            source="file",
            extra={"path": str(path), "note": "Install onnx for detailed profiling"},
        )

    try:
        model = onnx.load(str(path))
        params = sum(
            _onnx_tensor_size(init) for init in model.graph.initializer
        )
        return ModelInfo(
            name=path.stem,
            parameters=params,
            weights_size_mb=path.stat().st_size / (1024 * 1024),
            source="file",
            extra={"path": str(path)},
        )
    except Exception:
        return ModelInfo(
            name=path.stem,
            weights_size_mb=path.stat().st_size / (1024 * 1024),
            source="file",
            extra={"path": str(path), "note": "Could not parse ONNX model"},
        )


def _onnx_tensor_size(tensor) -> int:
    """Count number of elements in an ONNX tensor."""
    import numpy as np
    from onnx import numpy_helper

    arr = numpy_helper.to_array(tensor)
    return arr.size


def _parse_hf_config(config: dict, model_name: str) -> ModelInfo:
    """Shared logic: turn a HuggingFace config.json dict → ModelInfo."""
    import json as _json  # noqa: F401 (may already be imported)

    # Gemma-4 (and some others) nest text settings under text_config; fall back to root.
    cfg = config.get("text_config") or config

    def _get_first(d: dict, *keys):
        for k in keys:
            if k in d and d[k] is not None:
                return d[k]
        return None

    hidden_size = _get_first(cfg, "hidden_size", "d_model")
    num_layers = _get_first(cfg, "num_hidden_layers", "n_layer", "num_layers")
    vocab_size = _get_first(cfg, "vocab_size")
    intermediate_size = _get_first(cfg, "intermediate_size", "ffn_dim", "ffn_hidden_size")

    # MoE detection
    num_experts = _get_first(cfg, "num_experts", "num_local_experts")
    experts_per_tok = _get_first(cfg, "num_experts_per_tok", "experts_per_token", "top_k")
    # moe_intermediate_size is the per-expert FFN dim (Qwen3-MoE); fall back to intermediate_size (Mixtral)
    expert_ffn_size = _get_first(cfg, "moe_intermediate_size", "expert_intermediate_size") or intermediate_size
    is_moe = bool(
        num_experts
        or any("MoE" in a or "Moe" in a for a in config.get("architectures", []))
        or "moe" in (cfg.get("model_type") or "").lower()
    )

    # Robust transformer parameter estimate
    # Accounts for embedding table + N × (attention QKV+proj + FFN up/gate/down)
    params = None
    active_params = None
    if hidden_size and num_layers and vocab_size:
        embed_params = vocab_size * hidden_size
        attn_per_layer = 4 * hidden_size * hidden_size

        if is_moe and num_experts and expert_ffn_size:
            # Total: all experts are loaded into memory
            total_expert_ffn = num_experts * 3 * hidden_size * expert_ffn_size
            params = embed_params + num_layers * (attn_per_layer + total_expert_ffn)
            # Active: only experts_per_tok experts fire per token (bandwidth-bound speed)
            if experts_per_tok:
                active_expert_ffn = experts_per_tok * 3 * hidden_size * expert_ffn_size
                active_params = embed_params + num_layers * (attn_per_layer + active_expert_ffn)
        elif intermediate_size:
            # Each layer: attn (4·h²) + FFN (3·h·ffn for SwiGLU, or 2·h·ffn)
            # Use 3× for safety (covers both MLP and SwiGLU variants)
            layer_params = attn_per_layer + 3 * hidden_size * intermediate_size
            params = embed_params + num_layers * layer_params
        else:
            layer_params = 12 * hidden_size * hidden_size
            params = embed_params + num_layers * layer_params

    llm_config = {}
    for key in [
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "intermediate_size",
        "vocab_size",
        "max_position_embeddings",
        "num_key_value_heads",
        "num_layers",
    ]:
        if key in cfg:
            llm_config[key] = cfg[key]

    model_type = cfg.get("model_type", config.get("model_type", "unknown"))
    _LLM_ARCHS = {
        "llama", "mistral", "phi", "phi3", "phi4", "gpt2", "gpt_neo",
        "gpt_neox", "gptj", "falcon", "qwen2", "qwen2_moe", "qwen3",
        "qwen3_moe", "gemma", "gemma2", "gemma3", "gemma4", "gemma4_text",
        "mixtral", "starcoder2",
        "internlm", "internlm2", "baichuan", "deepseek", "deepseek_v2",
        "deepseek_v3", "cohere", "command_r", "bloom", "opt", "mpt",
        "stablelm", "persimmon", "olmo", "olmo2", "dbrx", "jamba",
        "arctic", "exaone", "granite",
    }
    if model_type.lower() in _LLM_ARCHS:
        our_type = "llm"
    elif model_type in ("vit", "deit", "beit", "swin", "convnext", "efficientnet",
                        "resnet", "mobilenet_v2", "mobilenet_v3", "efficientformer"):
        our_type = "classification"
    elif model_type in ("detr", "yolos", "conditional_detr", "deta", "rt_detr"):
        our_type = "detection"
    elif model_type in ("whisper", "wav2vec2", "hubert", "speech_encoder_decoder"):
        our_type = "audio"
    else:
        our_type = model_type

    return ModelInfo(
        name=model_name,
        parameters=params,
        active_params=active_params,
        is_moe=is_moe,
        model_type=our_type,
        architecture=model_type,
        llm_config=llm_config if llm_config else None,
        source="huggingface",
        extra={"hf_config": config},
    )


def resolve_from_huggingface(model_name: str) -> ModelInfo | None:
    """Fetch model config from HuggingFace Hub.

    Tries two methods in order:
    1. huggingface_hub library (uses local cache, handles auth)
    2. requests fallback — fetches config.json directly from HF CDN
       (works for public models without any extra dependencies)
    """
    import json

    # Method 1: huggingface_hub (preferred — handles private repos + caching)
    try:
        from huggingface_hub import hf_hub_download
        import logging as _logging
        _hf_log = _logging.getLogger("huggingface_hub")
        _prev   = _hf_log.level
        _hf_log.setLevel(_logging.ERROR)   # silence "Retry 1/5" noise
        try:
            config_path = hf_hub_download(repo_id=model_name, filename="config.json")
            with open(config_path) as f:
                config = json.load(f)
            return _parse_hf_config(config, model_name)
        except Exception as exc:  # noqa: BLE001
            logger.debug("hf_hub_download failed for %s: %s", model_name, exc)
        finally:
            _hf_log.setLevel(_prev)
    except ImportError:
        logger.debug("huggingface_hub not installed; using HTTP fallback")

    # Method 2: direct requests fallback (no huggingface_hub needed)
    try:
        import requests
        url = f"https://huggingface.co/{model_name}/resolve/main/config.json"
        resp = requests.get(url, timeout=15, headers={
            "User-Agent": "willitrun/0.1 (model-memory-estimator)",
        })
        if resp.status_code == 200:
            config = resp.json()
            return _parse_hf_config(config, model_name)
        logger.debug("config.json fetch failed for %s: HTTP %s", model_name, resp.status_code)
    except Exception as exc:  # noqa: BLE001
        logger.debug("config.json fetch error for %s: %s", model_name, exc)

    return None


def resolve_model(model_input: str) -> ModelInfo:
    """Resolve a model from any source: database, local file, or HuggingFace.

    Resolution order:
    1. Built-in database (fast, no dependencies)
    2. Local file (.pt, .onnx)
    3. HuggingFace Hub (requires huggingface_hub)
    """
    # 1. Try database first
    info = resolve_from_database(model_input)
    if info:
        return info

    # 2. Try local file
    if os.path.exists(model_input):
        info = resolve_from_file(model_input)
        if info:
            # Also check database for additional metadata using the filename
            db_info = resolve_from_database(Path(model_input).stem)
            if db_info:
                # Merge: prefer file-derived values, fill gaps from DB
                if info.parameters is None:
                    info.parameters = db_info.parameters
                if info.flops is None:
                    info.flops = db_info.flops
                if info.model_type is None:
                    info.model_type = db_info.model_type
                if info.default_input_size is None:
                    info.default_input_size = db_info.default_input_size
                if info.architecture is None:
                    info.architecture = db_info.architecture
                if info.llm_config is None:
                    info.llm_config = db_info.llm_config
            return info

    # 3. Try HuggingFace
    if "/" in model_input:
        info = resolve_from_huggingface(model_input)
        if info:
            return info

    # Fallback: return minimal info
    return ModelInfo(name=model_input, source="unknown")
