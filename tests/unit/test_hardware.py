# tests/unit/test_hardware.py
"""Tests for hardware detection and VRAM estimation."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from src.utils.hardware import (
    HardwareProfile,
    _detect_cuda,
    _match_fallback_tier,
    _normalize_gpu_name,
    detect_hardware,
    estimate_vram_usage,
)

# ---------------------------------------------------------------------------
# GPU name normalization
# ---------------------------------------------------------------------------


class TestNormalizeGpuName:
    """Tests for GPU name normalization."""

    def test_nvidia_geforce_prefix(self) -> None:
        assert _normalize_gpu_name("NVIDIA GeForce RTX 5060") == "RTX_5060"

    def test_nvidia_prefix_only(self) -> None:
        assert _normalize_gpu_name("NVIDIA RTX A4000") == "RTX_A4000"

    def test_double_prefix(self) -> None:
        # Some drivers report "NVIDIA GeForce ..."
        result = _normalize_gpu_name("NVIDIA GeForce RTX 3090")
        assert result == "RTX_3090"

    def test_hyphens_to_underscores(self) -> None:
        assert _normalize_gpu_name("RTX 2080-Ti") == "RTX_2080_Ti"

    def test_plain_name(self) -> None:
        assert _normalize_gpu_name("RTX 5060") == "RTX_5060"

    def test_laptop_gpu_suffix_removed(self) -> None:
        result = _normalize_gpu_name("NVIDIA GeForce RTX 5060 Laptop GPU")
        assert result == "RTX_5060"


# ---------------------------------------------------------------------------
# CUDA detection
# ---------------------------------------------------------------------------


class TestDetectCuda:
    """Tests for CUDA probing."""

    def test_uses_total_memory_property(self, monkeypatch: pytest.MonkeyPatch) -> None:
        props = SimpleNamespace(
            name="NVIDIA GeForce RTX 5060 Laptop GPU",
            total_memory=8 * 1024**3,
            major=12,
            minor=0,
        )
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: True,
                current_device=lambda: 0,
                get_device_properties=lambda _idx: props,
                memory_allocated=lambda _idx: 0,
            )
        )

        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        result = _detect_cuda()

        assert result is not None
        assert result["device"] == "cuda"
        assert result["device_name"] == props.name
        assert result["vram_gb"] == pytest.approx(8.0)
        assert result["architecture"] == "blackwell"
        assert result["supports_bf16"] is True


# ---------------------------------------------------------------------------
# Fallback tier matching
# ---------------------------------------------------------------------------


class TestFallbackTier:
    """Tests for VRAM-based fallback tier matching."""

    TIERS = [
        {
            "vram_up_to_gb": 6,
            "recommended": {"batch_size": 1, "precision": "16-mixed"},
        },
        {
            "vram_up_to_gb": 8,
            "recommended": {"batch_size": 2, "precision": "bf16-mixed"},
        },
        {
            "vram_up_to_gb": 16,
            "recommended": {"batch_size": 8, "precision": "bf16-mixed"},
        },
    ]

    def test_exact_match(self) -> None:
        result = _match_fallback_tier(8.0, self.TIERS)
        assert result["batch_size"] == 2

    def test_below_threshold(self) -> None:
        result = _match_fallback_tier(5.0, self.TIERS)
        assert result["batch_size"] == 1

    def test_between_tiers(self) -> None:
        result = _match_fallback_tier(10.0, self.TIERS)
        assert result["batch_size"] == 8

    def test_above_all_tiers(self) -> None:
        result = _match_fallback_tier(48.0, self.TIERS)
        # Should fall through to the ultra-conservative default
        assert result["batch_size"] == 1


# ---------------------------------------------------------------------------
# VRAM estimation
# ---------------------------------------------------------------------------


class TestEstimateVram:
    """Tests for VRAM usage estimation."""

    def test_default_returns_all_components(self) -> None:
        result = estimate_vram_usage()
        expected_keys = {
            "base_model",
            "lora_adapters",
            "optimizer_states",
            "activations",
            "total",
        }
        assert set(result.keys()) == expected_keys

    def test_total_is_sum_of_components(self) -> None:
        result = estimate_vram_usage()
        component_sum = (
            result["base_model"]
            + result["lora_adapters"]
            + result["optimizer_states"]
            + result["activations"]
        )
        assert abs(result["total"] - component_sum) < 0.02

    def test_bf16_halves_base_model(self) -> None:
        bf16 = estimate_vram_usage(precision="bf16-mixed")
        fp32 = estimate_vram_usage(precision="32")
        # bf16 base model should be roughly half of fp32
        assert bf16["base_model"] < fp32["base_model"]
        assert abs(bf16["base_model"] * 2 - fp32["base_model"]) < 0.1

    def test_gradient_checkpointing_reduces_activations(self) -> None:
        with_gc = estimate_vram_usage(gradient_checkpointing=True)
        without_gc = estimate_vram_usage(gradient_checkpointing=False)
        assert with_gc["activations"] < without_gc["activations"]

    def test_larger_batch_increases_activations(self) -> None:
        small = estimate_vram_usage(batch_size=1)
        large = estimate_vram_usage(batch_size=4)
        assert large["activations"] > small["activations"]

    def test_8gb_gpu_estimate_fits_recommended_microbatch(self) -> None:
        """Verify the revised 8GB profile stays within budget."""
        result = estimate_vram_usage(
            model_params_m=650,
            lora_params_m=16,
            precision="bf16-mixed",
            batch_size=1,
            seq_length=1024,
            gradient_checkpointing=True,
        )
        assert result["total"] < 8.0, (
            f"Estimated VRAM {result['total']:.2f} GB exceeds 8 GB budget. Breakdown: {result}"
        )


# ---------------------------------------------------------------------------
# Hardware detection (integration-light: uses actual system)
# ---------------------------------------------------------------------------


class TestDetectHardware:
    """Tests for the full detection pipeline."""

    def test_returns_hardware_profile(self) -> None:
        profile = detect_hardware()
        assert isinstance(profile, HardwareProfile)

    def test_device_is_valid(self) -> None:
        profile = detect_hardware()
        assert profile.device in ("cuda", "mps", "cpu")

    def test_summary_is_string(self) -> None:
        profile = detect_hardware()
        summary = profile.summary()
        assert isinstance(summary, str)
        assert "Device:" in summary

    def test_cpu_fallback_no_crash(self) -> None:
        """Even without a GPU, detection should succeed."""
        profile = detect_hardware()
        assert profile.device_name is not None
        assert profile.batch_size >= 1
