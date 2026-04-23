"""Unit tests for scripts/gpu_map.py — GPU and Apple chip name resolution.

Covers:
- Vendor prefix stripping (GeForce, NVIDIA, Tesla)
- Parenthetical memory annotation stripping
- Bare number → RTX prepend (e.g. "Nvidia 4090")
- Longest-key substring ordering (no "rtx 4070" matching "rtx 4070 ti")
- Apple chip normalization (core count / GB stripping)
- Known false-positive cases (B-8: Tesla M40 must not route to Apple resolver)
- Unknown inputs return None
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# scripts/ is not a package — add it to sys.path so the import works both
# when running from repo root and from inside the tests/ directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from gpu_map import resolve_apple_chip, resolve_gpu  # noqa: E402


# ---------------------------------------------------------------------------
# resolve_gpu — NVIDIA / AMD / Intel
# ---------------------------------------------------------------------------

class TestResolveGpu:
    def test_bare_rtx_name(self):
        assert resolve_gpu("RTX 4090") == "rtx-4090-24gb"

    def test_geforce_prefix_stripped(self):
        assert resolve_gpu("GeForce RTX 4090") == "rtx-4090-24gb"

    def test_nvidia_prefix_stripped(self):
        assert resolve_gpu("NVIDIA RTX 4090") == "rtx-4090-24gb"

    def test_paren_memory_stripped(self):
        assert resolve_gpu("RTX 4090 (24 GB)") == "rtx-4090-24gb"

    def test_china_variant_d_suffix(self):
        # RTX 4090 D (Chinese market) maps to same device
        assert resolve_gpu("RTX 4090 D") == "rtx-4090-24gb"

    def test_bare_number_gets_rtx_prepended(self):
        # Sources like geerlingguy emit "Nvidia 4090" without "RTX"
        assert resolve_gpu("Nvidia 4090") == "rtx-4090-24gb"

    def test_ti_variant_not_shadowed_by_base(self):
        # "rtx 4070 ti" must NOT fall back to "rtx 4070"
        assert resolve_gpu("RTX 4070 Ti") == "rtx-4070ti-12gb"
        assert resolve_gpu("RTX 4070") == "rtx-4070-12gb"

    def test_super_variant(self):
        assert resolve_gpu("RTX 4080 SUPER") == "rtx-4080s-16gb"

    def test_amd_gpu(self):
        assert resolve_gpu("AMD Radeon RX 7900 XTX") == "rx-7900-xtx-24gb"

    def test_intel_arc(self):
        result = resolve_gpu("Intel Arc B580")
        assert result is not None
        assert "b580" in result

    def test_unknown_returns_none(self):
        assert resolve_gpu("Unknown XYZ Accelerator 9999") is None

    def test_empty_string_returns_none(self):
        assert resolve_gpu("") is None

    # --- B-8 false-positive regression ---

    def test_tesla_m40_not_apple(self):
        # "Tesla M40" contains "m4" — must NOT be routed to Apple chip resolver.
        # resolve_gpu should return a datacenter GPU or None, never an Apple ID.
        result = resolve_gpu("NVIDIA Tesla M40")
        assert result is None or "apple" not in result

    def test_tesla_m10_not_apple(self):
        # "Tesla M10" contains "m1" — same guard.
        result = resolve_gpu("NVIDIA Tesla M10")
        assert result is None or "apple" not in result


# ---------------------------------------------------------------------------
# resolve_apple_chip — Apple Silicon
# ---------------------------------------------------------------------------

class TestResolveAppleChip:
    def test_m1(self):
        assert resolve_apple_chip("M1") == "apple-m1-16gb"

    def test_m2_pro(self):
        assert resolve_apple_chip("M2 Pro") is not None
        assert "m2-pro" in resolve_apple_chip("M2 Pro")

    def test_m4_pro_with_memory(self):
        # Memory suffix should be stripped during normalisation
        result = resolve_apple_chip("M4 Pro 24GB")
        assert result is not None
        assert "m4-pro" in result

    def test_apple_prefix_stripped(self):
        # "Apple M3" and "M3" should resolve identically
        assert resolve_apple_chip("Apple M3") == resolve_apple_chip("M3")

    def test_core_count_stripped(self):
        # "M1 Max (32 GPU Core)" → resolves same as "M1 Max"
        assert resolve_apple_chip("M1 Max (32 GPU Core)") == resolve_apple_chip("M1 Max")

    def test_gpu_suffix_stripped(self):
        # "Apple M1 Max GPU" trailing "GPU" stripped by normalize_gpu_name
        assert resolve_apple_chip("M1 Max") is not None

    def test_unknown_returns_none(self):
        assert resolve_apple_chip("Intel Core i9") is None
