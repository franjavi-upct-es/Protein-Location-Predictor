# src/utils/hardware.py
"""
Hardware detection and resource-aware configuration.

Detects the available GPU (or CPU / Apple Silicon), profiles its VRAM,
and returns recommended training parameters by matching against known
GPU profiles or falling back to VRAM-based tiers.

Usage::

    from src.utils.hardware import detect_hardware, HardwareProfile
    hw = detect_hardware(cfg)
    print(hw.device, hw.precision, hw.batch_size)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class HardwareProfile:
    """Resolved hardware configuration for the current machine."""

    device: str  # "cuda", "mps", "cpu"
    device_name: str  # Human-readable, e.g., "NVIDIA RTX 5060"
    vram_gb: float  # Total VRAM in GB (0 for CPU)
    vram_free_gb: float  # Free VRAM at detection time
    architecture: str  # "blackwell", "ampere", "apple_silicon", "cpu"

    # Recommended training parameters
    precision: str = "32"  # "bf16-mixed", "16-mixed", "32"
    batch_size: int = 1
    gradient_checkpointing: bool = True
    cpu_offload: bool = False
    max_sequence_length: int = 512

    # Capabilities
    supports_bf16: bool = False
    supports_fp16: bool = False

    matched_profile: str | None = None  # Name of the matched GPU profile, if any

    def summary(self) -> str:
        """Return a human-readable summary of the hardware profile."""
        lines = [
            f"Device:           {self.device_name} ({self.device})",
            (f"VRAM:             {self.vram_gb:.1f} GB total, {self.vram_free_gb:.1f} GB free"),
            f"Architecture:     {self.architecture}",
            f"Precision:        {self.precision}",
            f"Batch size:       {self.batch_size}",
            f"Grad checkpoint:  {self.gradient_checkpointing}",
            f"CPU offload:      {self.cpu_offload}",
            f"Max seq length:   {self.max_sequence_length}",
        ]
        if self.matched_profile:
            lines.append(f"Matched profile: {self.matched_profile}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# GPU detection helpers
# ---------------------------------------------------------------------------


def _detect_cuda() -> dict[str, Any] | None:
    """Detect CUDA GPU and return its properties, or None if unavailable."""
    try:
        import torch
    except Exception as e:
        logger.debug(f"PyTorch import failed during CUDA detection: {e}")
        return None

    try:
        if not torch.cuda.is_available():
            return None

        device_idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_idx)
        vram_total = props.total_memory / (1024**3)

        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(device_idx)
            vram_free = free_bytes / (1024**3)
            vram_total = total_bytes / (1024**3)
        except Exception:
            vram_free = (props.total_memory - torch.cuda.memory_allocated(device_idx)) / (1024**3)

        # Determine architecture from copute capability
        major, minor = props.major, props.minor
        arch_map = {
            7: "turing",  # SM 7.x = Turing (RTX 20xx)
            8: "ampere",  # SM 8.x = Ampere (RTX 30xx) / Ada (8.9)
            9: "ada_lovelace",  # SM 9.0 = Ada Lovelace (RTX 40xx)
            10: "blackwell",  # SM 10.x = Blackwell (RTX 50xx)
            12: "blackwell",  # SM 12.x = Blackwell (RTX 50xx, alternate)
        }
        # Ada Lovelace is SM 8.9, Ampere is SM 8.0-8.6
        if major == 8 and minor >= 9:
            architecture = "ada_lovelace"
        else:
            architecture = arch_map.get(major, f"unknown_sm{major}")

        # bf16 support: Ampere (SM 8.0+) and later
        supports_bf16 = major >= 8
        supports_fp16 = major >= 7

        return {
            "device": "cuda",
            "device_name": props.name,
            "vram_gb": vram_total,
            "vram_free_gb": vram_free,
            "architecture": architecture,
            "supports_bf16": supports_bf16,
            "supports_fp16": supports_fp16,
        }
    except Exception as e:
        logger.warning(f"CUDA appears available but device probing failed: {e}")
        return None


def _detect_mps() -> dict[str, Any] | None:
    """Detect Apple Silicon MPS device."""
    try:
        import torch

        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            return None

        # Apple Silicon uses unified memory — estimate available
        import subprocess  # nosec B404

        result = subprocess.run(  # nosec B603 B607
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        total_ram_gb = int(result.stdout.strip()) / (1024**3) if result.returncode == 0 else 8.0
        # Rough heuristic: ~75% of unified memory is usable for GPU tasks
        usable_gb = total_ram_gb * 0.75

        return {
            "device": "mps",
            "device_name": f"Apple Silicon ({total_ram_gb:.0f} GB unified)",
            "vram_gb": usable_gb,
            "vram_free_gb": usable_gb * 0.8,
            "architecture": "apple_silicon",
            "supports_bf16": False,  # MPS bf16 support is incomplete
            "supports_fp16": True,
        }
    except Exception as e:
        logger.debug(f"MPS detection failed: {e}")
        return None


def _cpu_fallback() -> dict[str, Any]:
    """Return a CPU-only hardware profile."""
    return {
        "device": "cpu",
        "device_name": "CPU",
        "vram_gb": 0.0,
        "vram_free_gb": 0.0,
        "architecture": "cpu",
        "supports_bf16": False,
        "supports_fp16": False,
    }


# ---------------------------------------------------------------------------
# Profile matching
# ---------------------------------------------------------------------------


def _normalize_gpu_name(name: str) -> str:
    """
    Normalize a GPU name for matching against profile keys.

    Examples::

        'NVIDIA GeForce RTX 5060' -> 'RTX_5060'
        'NVIDIA RTX A4000'        -> 'RTX_A4000'
    """
    # Remove vendor prefix
    name = re.sub(r"(?i)^(NVIDIA|GeForce)\s+", "", name).strip()
    name = re.sub(r"(?i)^GeForce\s+", "", name).strip()
    # Laptop SKUs should match the base GPU profile names.
    name = re.sub(r"(?i)\s+Laptop GPU$", "", name).strip()
    # Replace spaces and hyphens with underscores
    name = re.sub(r"[\s\-]+", "_", name)
    return name


def _match_profile(
    gpu_info: dict[str, Any],
    profiles: dict[str, dict],
) -> dict[str, Any] | None:
    """Try to match the detected GPU against a known profile."""
    normalized = _normalize_gpu_name(gpu_info["device_name"])
    logger.debug(f"Matching GPU '{gpu_info['device_name']}' -> normalized '{normalized}'")

    for profile_name, profile_data in profiles.items():
        if normalized.lower() == profile_name.lower().replace(" ", "_"):
            logger.info(f"Matched GPU profile: {profile_name}")
            result: dict[str, Any] = profile_data.get("recommended", {}).copy()
            result["matched_profile"] = profile_name
            return result

    return None


def _match_fallback_tier(vram_gb: float, tiers: list[dict]) -> dict[str, Any]:
    """Match against VRAM-based fallback tiers."""
    for tier in sorted(tiers, key=lambda t: t["vram_up_to_gb"]):
        if vram_gb <= tier["vram_up_to_gb"]:
            logger.info(f"Using fallback VRAM tier: <= {tier['vram_up_to_gb']} GB")
            result: dict[str, Any] = tier.get("recommended", {}).copy()
            result["matched_profile"] = f"fallback_{tier['vram_up_to_gb']}gb"
            return result

    # Ultra-conservative if nothing matches
    return {
        "precision": "32",
        "batch_size": 1,
        "gradient_checkpointing": True,
        "cpu_offload": True,
        "max_sequence_length": 512,
        "matched_profile": "fallback_minimum",
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_hardware(cfg: Any = None) -> HardwareProfile:
    """
    Detect hardware and return a resolved HardwareProfile.

    The detections order is: CUDA -> MPS -> CPU.
    Training parameters are determine by:
        1. Named GPU profile match (from configs/hardware/gpu_profiles.yaml)
        2. VRAM-based fallback tier
        3. Ultra-conservative CPU defaults

    Args:
        cfg: A DotDict configuration (from load_config). If provided,
             hardware profiles and fallback tiers are read from it.
             Otherwise, only auto-detection is used.

    Returns:
        A HardwareProfile dataclass with all resolved parameters.
    """
    # Detect the hardware
    gpu_info = _detect_cuda() or _detect_mps() or _cpu_fallback()
    logger.info(
        f"Detected: {gpu_info['device_name']} "
        f"({gpu_info['vram_gb']:.1f} GB, {gpu_info['architecture']})"
    )

    # Resolve training parameters
    recommended: dict[str, Any] = {}

    if cfg is not None and "hardware" in cfg:
        hw_cfg = cfg["hardware"]
        profiles = hw_cfg.get("profiles", {})
        fallback_tiers = hw_cfg.get("fallback_tiers", [])

        if gpu_info["device"] != "cpu":
            recommended = _match_profile(gpu_info, profiles) or {}

        if not recommended and gpu_info["vram_gb"] > 0:
            recommended = _match_fallback_tier(gpu_info["vram_gb"], fallback_tiers)

    # CPU fallback
    if gpu_info["device"] == "cpu":
        recommended = {
            "precision": "32",
            "batch_size": 1,
            "gradient_checkpointing": False,
            "cpu_offload": True,
            "max_sequence_length": 512,
            "matched_profile": "cpu",
        }

    # Validate precision against hardware capabilities
    precision = recommended.get("precision", "32")
    if "bf16" in precision and not gpu_info.get("supports_bf16", False):
        logger.warning(
            f"bf16 requested but not supported by {gpu_info['device_name']}. Falling back to fp16."
        )
        precision = precision.replace("bf16", "16")
        recommended["precision"] = precision

    # Build the profile
    matched_profile = recommended.pop("matched_profile", None)

    profile = HardwareProfile(
        device=gpu_info["device"],
        device_name=gpu_info["device_name"],
        vram_gb=gpu_info["vram_gb"],
        vram_free_gb=gpu_info["vram_free_gb"],
        architecture=gpu_info["architecture"],
        supports_bf16=gpu_info.get("supports_bf16", False),
        supports_fp16=gpu_info.get("supports_fp16", False),
        matched_profile=matched_profile,
        **recommended,
    )

    logger.info(f"Hardware profile resolved:\n{profile.summary()}")
    return profile


def estimate_vram_usage(
    model_params_m: float = 150,
    lora_params_m: float = 4,
    precision: str = "bf16-mixed",
    batch_size: int = 8,
    seq_length: int = 1024,
    gradient_checkpointing: bool = True,
) -> dict[str, float]:
    """
    Estimate VRAM usage for a training configuration.

    Returns a dict with component-wise memory estimates in GB.
    This is an approximation — actual usage may vary.

    Args:
        model_params_m: Base model parameters in millions.
        lora_params_m: LoRA adapter parameters in millions.
        precision: Training precision mode.
        batch_size: Per-GPU batch size.
        seq_length: Maximum sequence length.
        gradient_checkpointing: Whether gradient checkpointing is enabled.

    Returns:
        Dict with keys: base_model, lora_adapters, optimizer, activations,
        total
    """
    bytes_per_param = 2 if "16" in precision or "bf16" in precision else 4

    # Base model (frozen weights, stored in reduced precision)
    base_model_gb = (model_params_m * 1e6 * bytes_per_param) / (1024**3)

    # LoRA adapters (trainable, need gradients)
    lora_gb = (lora_params_m * 1e6 * bytes_per_param) / (1024**3)

    # Optimizer states (AdamW: 2 states per trainable param, in fp32)
    optimizer_gb = (lora_params_m * 1e6 * 4 * 2) / (1024**3)

    # Activation memory (rough estimate)
    # ESM-2 hidden_dim=640, 30 layers, per-token per-layer
    hidden_dim = 640
    num_layers = 30
    act_per_token = hidden_dim * num_layers * bytes_per_param
    total_tokens = batch_size * seq_length
    activations_gb = (total_tokens * act_per_token) / (1024**3)

    if gradient_checkpointing:
        activations_gb *= 0.6  # ~40% reduction

    total = base_model_gb + lora_gb + optimizer_gb + activations_gb

    return {
        "base_model": round(base_model_gb, 2),
        "lora_adapters": round(lora_gb, 2),
        "optimizer_states": round(optimizer_gb, 2),
        "activations": round(activations_gb, 2),
        "total": round(total, 2),
    }
