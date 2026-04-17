"""Shared GPU chip name → willitrun device_id mapping.

Imported by ingest scripts that need to normalise GPU names from various
sources (CUDA driver strings, markdown tables, benchmark databases) into
canonical device IDs as defined in data/devices.yaml.

Usage:
    from gpu_map import resolve_gpu, resolve_apple_chip
"""

from __future__ import annotations

import re

# ── NVIDIA / AMD / Intel GPU → device_id ──────────────────────────────────────
# Keys are lowercase, prefix-stripped chip names.
# Entries are sorted from most-specific to least-specific within each family
# so that the substring fallback in resolve_gpu works correctly.
GPU_MAP: dict[str, str] = {
    # RTX 50 series
    "rtx 5090": "rtx-5090-32gb",
    "rtx 5080": "rtx-5080-16gb",
    "rtx 5070 ti": "rtx-5070ti-16gb",
    "rtx 5070": "rtx-5070-12gb",
    # RTX 40 series
    "rtx 4090 d": "rtx-4090-24gb",       # Chinese-market variant, same perf
    "rtx 4090": "rtx-4090-24gb",
    "rtx 4080 super": "rtx-4080s-16gb",
    "rtx 4080": "rtx-4080-16gb",
    "rtx 4070 ti super": "rtx-4070tis-16gb",
    "rtx 4070 ti": "rtx-4070ti-12gb",
    "rtx 4070 super": "rtx-4070s-12gb",
    "rtx 4070": "rtx-4070-12gb",
    "rtx 4060 ti 16gb": "rtx-4060ti-16gb",
    "rtx 4060 ti": "rtx-4060ti-8gb",
    "rtx 4060": "rtx-4060-8gb",
    # RTX 30 series
    "rtx 3090 ti": "rtx-3090-24gb",
    "rtx 3090": "rtx-3090-24gb",
    "rtx 3080 ti": "rtx-3080ti-12gb",
    "rtx 3080": "rtx-3080-10gb",
    "rtx 3070 ti": "rtx-3070ti-8gb",
    "rtx 3070": "rtx-3070-8gb",
    "rtx 3060 ti": "rtx-3060ti-8gb",
    "rtx 3060 12gb": "rtx-3060-12gb",
    "rtx 3060": "rtx-3060-12gb",
    # RTX 20 series
    "rtx 2080 ti": "rtx-2080ti-11gb",
    "rtx 2080 super": "rtx-2080s-8gb",
    "rtx 2080": "rtx-2080-8gb",
    "rtx 2070 super": "rtx-2070s-8gb",
    "rtx 2070": "rtx-2070-8gb",
    "rtx 2060 super": "rtx-2060s-8gb",
    "rtx 2060": "rtx-2060-6gb",
    # GTX series
    "gtx 1080 ti": "gtx-1080ti-11gb",
    "gtx 1080": "gtx-1080-8gb",
    "gtx 1070 ti": "gtx-1070ti-8gb",
    "gtx 1070": "gtx-1070-8gb",
    "gtx 1660 super": "gtx-1660s-6gb",
    "gtx 1660 ti": "gtx-1660ti-6gb",
    "gtx 1660": "gtx-1660-6gb",
    "gtx 1650": "gtx-1650-4gb",
    # Titan
    "titan rtx": "titan-rtx-24gb",
    "titan xp": "titan-xp-12gb",
    "titan v": "titan-v-12gb",
    # Datacenter / server
    "h100 sxm": "h100-80gb",
    "h100 pcie": "h100-80gb",
    "h100 80 gb": "h100-80gb",
    "h100": "h100-80gb",
    "a100 sxm": "a100-80gb",
    "a100 pcie": "a100-80gb",
    "a100 80 gb": "a100-80gb",
    "a100 40 gb": "a100-40gb",
    "a100": "a100-80gb",
    "a6000": "a6000-48gb",
    "rtx a6000": "a6000-48gb",
    "rtx a5000": "a5000-24gb",
    "rtx a4500": "rtx-a4500-20gb",
    "rtx a4000": "a4000-16gb",
    "l40s": "l40s-48gb",
    "l40": "l40-48gb",
    "l4": "l4-24gb",
    "a30": "a30-24gb",
    "a40": "a40-48gb",
    "v100 32 gb": "v100-32gb",
    "v100": "v100-16gb",
    "p100": "tesla-p100-16gb",
    "p40": "tesla-p40-24gb",
    "t4": "t4-16gb",
    # Quadro / RTX Pro / Ada
    "rtx pro 6000 blackwell": "rtx-pro-6000-96gb",
    "rtx 6000 ada": "rtx-6000ada-48gb",
    "rtx 5000 ada": "rtx-5000ada-32gb",
    "rtx 4000 ada": "rtx-4000ada-20gb",
    "quadro rtx 8000": "quadro-rtx8000-48gb",
    "quadro rtx 6000": "quadro-rtx6000-24gb",
    "quadro rtx 4000": "quadro-rtx4000-8gb",
    # Laptop GPUs
    "rtx 4090 laptop": "rtx-4090-laptop-16gb",
    "rtx 4080 laptop": "rtx-4080-laptop-12gb",
    "rtx 4070 laptop": "rtx-4070-laptop-8gb",
    "rtx 4060 laptop": "rtx-4060-laptop-8gb",
    "rtx 3080 laptop": "rtx-3080-laptop-16gb",
    "rtx 3070 laptop": "rtx-3070-laptop-8gb",
    "rtx 3060 laptop": "rtx-3060-laptop-6gb",
    # Jetson
    "jetson agx orin": "jetson-agx-orin-64gb",
    "jetson orin nx": "jetson-orin-nx-16gb",
    "jetson orin nano": "jetson-orin-nano-8gb",
    # Intel Arc — discrete
    "intel arc a770m": "intel-arc-a770m-16gb",
    "arc a770m": "intel-arc-a770m-16gb",
    "intel arc a770": "intel-arc-a770-16gb",
    "arc a770": "intel-arc-a770-16gb",
    "intel arc a750": "intel-arc-a750-8gb",
    "arc a750": "intel-arc-a750-8gb",
    "intel arc pro b60": "intel-arc-pro-b60-24gb",
    "arc pro b60": "intel-arc-pro-b60-24gb",
    "intel arc pro b50": "intel-arc-pro-b50-24gb",
    "arc pro b50": "intel-arc-pro-b50-24gb",
    "intel arc b580": "intel-arc-b580-12gb",
    "arc b580": "intel-arc-b580-12gb",
    "intel arc b570": "intel-arc-b570-10gb",
    "arc b570": "intel-arc-b570-10gb",
    # Intel iGPU
    "intel core ultra 200 series": "intel-core-ultra-200",
    "intel core ultra 200": "intel-core-ultra-200",
    "core ultra 200": "intel-core-ultra-200",
    "intel core ultra 100 series": "intel-core-ultra-100",
    "intel core ultra 100": "intel-core-ultra-100",
    "core ultra 100": "intel-core-ultra-100",
    # AMD RDNA4 discrete
    "radeon rx 9070 xt": "rx-9070-xt-16gb",
    "rx 9070 xt": "rx-9070-xt-16gb",
    "amd rx 9070 xt": "rx-9070-xt-16gb",
    "radeon rx 9070": "rx-9070-16gb",
    "rx 9070": "rx-9070-16gb",
    "amd rx 9070": "rx-9070-16gb",
    # AMD RDNA3 discrete
    "radeon rx 7900 xtx": "rx-7900-xtx-24gb",
    "rx 7900 xtx": "rx-7900-xtx-24gb",
    "amd 7900 xtx": "rx-7900-xtx-24gb",        # geerlingguy: no "RX" prefix
    "radeon rx 7900 xt": "rx-7900-xt-20gb",
    "rx 7900 xt": "rx-7900-xt-20gb",
    "amd 7900 xt": "rx-7900-xt-20gb",          # geerlingguy: no "RX" prefix
    "radeon rx 7900 gre": "rx-7900-gre-16gb",
    "rx 7900 gre": "rx-7900-gre-16gb",
    "amd 7900 gre": "rx-7900-gre-16gb",        # geerlingguy: no "RX" prefix
    "radeon rx 7800 xt": "rx-7800-xt-16gb",
    "rx 7800 xt": "rx-7800-xt-16gb",
    "amd 7800 xt": "rx-7800-xt-16gb",          # geerlingguy: no "RX" prefix
    # AMD RDNA3 budget
    "radeon rx 7600": "rx-7600-8gb",
    "rx 7600": "rx-7600-8gb",
    # AMD RDNA2 budget
    "radeon rx 6650 xt": "rx-6650-xt-8gb",
    "rx 6650 xt": "rx-6650-xt-8gb",
    # AMD RDNA2 discrete
    "radeon rx 6900 xt": "rx-6900-xt-16gb",
    "rx 6900 xt": "rx-6900-xt-16gb",
    "radeon rx 6800 xt": "rx-6800-xt-16gb",
    "rx 6800 xt": "rx-6800-xt-16gb",
    "radeon rx 6700 xt": "rx-6700-xt-12gb",
    "rx 6700 xt": "rx-6700-xt-12gb",
    # AMD APU / iGPU — short-form aliases used in community benchmarks
    "395+": "amd-ryzen-ai-max-395",             # "Framework Desktop Mainboard (395+)"
    "amd ryzen ai max+ 395": "amd-ryzen-ai-max-395",
    "amd ryzen al max+ 395": "amd-ryzen-ai-max-395",   # typo seen in community data
    "ryzen ai max+ 395": "amd-ryzen-ai-max-395",
    "amd ryzen ai 9 300 series": "amd-ryzen-ai-9-300",
    "ryzen ai 9 300 series": "amd-ryzen-ai-9-300",
    "amd ryzen z1 extreme": "amd-ryzen-z1-extreme",
    "ryzen z1 extreme": "amd-ryzen-z1-extreme",
    "amd ryzen 8000 series": "amd-ryzen-8000-apu",
    "ryzen 8000 series": "amd-ryzen-8000-apu",
    "amd ryzen 6000 series": "amd-ryzen-6000-apu",
    "ryzen 6000 series": "amd-ryzen-6000-apu",
    # Apple via MoltenVK / Vulkan — slightly lower than native Metal
    "apple m4 max": "apple-m4-max-36gb",
    "apple m3 ultra": "apple-m1-ultra-64gb",  # no M3 Ultra entry; map nearest
    "apple m3": "apple-m3-8gb",
    "apple m2 ultra": "apple-m2-ultra-64gb",
    "apple m2 pro": "apple-m2-pro-16gb",
    "apple m2": "apple-m2-16gb",
    "apple m1": "apple-m1-16gb",
}

# ── Apple Silicon chip name → device_id ───────────────────────────────────────
# Used for sources that identify hardware by chip name rather than GPU model.
# Keys are lowercase chip names, stripped of core counts and memory sizes.
APPLE_CHIP_MAP: dict[str, str] = {
    "m1": "apple-m1-16gb",
    "m1 pro": "apple-m1-pro-16gb",
    "m1 max": "apple-m1-max-32gb",
    "m1 ultra": "apple-m1-ultra-64gb",
    "m2": "apple-m2-16gb",
    "m2 pro": "apple-m2-pro-16gb",
    "m2 max": "apple-m2-max-32gb",
    "m2 ultra": "apple-m2-ultra-64gb",
    "m3": "apple-m3-8gb",
    "m3 pro": "apple-m3-pro-18gb",
    "m3 max": "apple-m3-max-36gb",
    "m3 ultra": "apple-m1-ultra-64gb",   # no M3 Ultra entry; nearest approximation
    "m4": "apple-m4-pro-24gb",            # base M4 → closest Pro config
    "m4 pro": "apple-m4-pro-24gb",
    "m4 max": "apple-m4-max-36gb",
    "m5": "apple-m5-32gb",
    "m5 max": "apple-m5-max-64gb",
}


def normalize_gpu_name(raw: str) -> str:
    """Strip vendor prefixes and memory annotations from a GPU chip name.

    Examples:
        "GeForce RTX 4090"       → "rtx 4090"
        "NVIDIA RTX 4090 D"      → "rtx 4090 d"
        "RTX 4090 (24 GB)"       → "rtx 4090"
        "AMD Radeon RX 7900 XTX" → "radeon rx 7900 xtx"
        "Intel Arc B580"         → "intel arc b580"
    """
    name = raw.strip().lower()
    for prefix in ("nvidia ", "geforce ", "tesla "):
        if name.startswith(prefix):
            name = name[len(prefix):]
    # Remove parenthetical memory/spec annotations
    name = re.sub(r"\s*\([^)]*\)\s*", " ", name)
    # Remove trailing "gpu" (e.g. "Apple M1 Max GPU")
    name = re.sub(r"\s+gpu$", "", name)
    # Collapse whitespace
    return re.sub(r"\s+", " ", name).strip()


def normalize_apple_chip(raw: str) -> str:
    """Strip core counts and memory sizes from Apple chip names.

    Examples:
        "M3 Ultra 512GB"       → "m3 ultra"
        "M1 Max (32 GPU Core)" → "m1 max"
        "Apple M2 Pro 16GB"    → "m2 pro"
    """
    name = raw.strip().lower()
    if name.startswith("apple "):
        name = name[6:]
    # Remove parenthetical
    name = re.sub(r"\s*\([^)]*\)\s*", " ", name)
    # Remove trailing GB/TB memory sizes
    name = re.sub(r"\s+\d+\s*(?:gb|tb)$", "", name)
    # Remove trailing core counts  "10 core", "38 core gpu", etc.
    name = re.sub(r"\s+\d+\s*(?:gpu\s+)?core(?:s)?.*$", "", name)
    return re.sub(r"\s+", " ", name).strip()


def resolve_gpu(raw_chip: str) -> str | None:
    """Map a raw GPU chip name string to a willitrun device_id.

    Tries, in order:
    1. Exact match after normalization
    2. Match after stripping trailing memory size (e.g. "12gb")
    3. Prepend "rtx" for bare number strings (e.g. "Nvidia 4090" → "rtx 4090")
    4. Longest-key substring match
    Returns None if no match found.
    """
    normalized = normalize_gpu_name(raw_chip)

    if normalized in GPU_MAP:
        return GPU_MAP[normalized]

    # Strip trailing memory annotation
    without_mem = re.sub(r"\s+\d+\s*gb$", "", normalized)
    if without_mem in GPU_MAP:
        return GPU_MAP[without_mem]

    # Sources like geerlingguy write "Nvidia 4090" (no "RTX") — try prepending
    if re.match(r"^[\d]+(\s+(ti|super|xt|xtx|gre))*$", without_mem):
        with_rtx = "rtx " + without_mem
        if with_rtx in GPU_MAP:
            return GPU_MAP[with_rtx]

    # Longest-key substring match (avoids "rtx 4070" matching "rtx 4070 ti")
    for key in sorted(GPU_MAP, key=len, reverse=True):
        if key in normalized:
            return GPU_MAP[key]

    return None


def resolve_apple_chip(raw_chip: str) -> str | None:
    """Map a raw Apple chip name to a willitrun device_id."""
    normalized = normalize_apple_chip(raw_chip)
    if normalized in APPLE_CHIP_MAP:
        return APPLE_CHIP_MAP[normalized]
    # Longest-key substring match
    for key in sorted(APPLE_CHIP_MAP, key=len, reverse=True):
        if key in normalized:
            return APPLE_CHIP_MAP[key]
    return None
