# src/models/esm_lora.py
"""
ESM-2 backbone with LoRA adapters via PEFT, with optional QLoRA support.

Loads the pre-trained ESM-2 model and injects low-rank adapter matrices
into the attention layers, enabling parameter-efficient fine-tuning that
fits within 8 GB VRAM. When quantization is enabled in the configuration,
the base model is loaded in 4-bit (NF4 by default) via bitsandbytes,
which dramatically reduces VRAM usage and unlocks larger backbones on
consumer GPUs (the QLoRA path).

Usage::

    from src.models.esm_lora import build_esm_lora_backbone
    backbone = build_esm_lora_backbone(cfg)
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Protocol, cast

import torch
import torch.nn as nn

from src.utils.config import DotDict
from src.utils.logging import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from transformers import PreTrainedModel
else:
    PreTrainedModel = nn.Module


class _HasLastHiddenState(Protocol):
    """Structural type for transformer outputs used by the pooling helper."""

    last_hidden_state: torch.Tensor


class QuantizationRuntimeUnsupportedError(RuntimeError):
    """Raised when the local CUDA stack cannot run QLoRA quantization."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_esm_lora_backbone(
    cfg: DotDict, enable_gradient_checkpointing: bool = True
) -> PreTrainedModel:
    """
    Build an ESM-2 backbone with LoRA adapters.

    If ``cfg.model.quantization.enabled`` is true, the base model is
    loaded in 4-bit (or 8-bit) precision via bitsandbytes and prepared
    for k-bit training before LoRA is applied. This is the QLoRA path
    and requires a CUDA-capable GPU.

    Args:
        cfg: Project configuration. Needs ``model.backbone`` and
            ``model.lora`` sections, and optionally
            ``model.quantization``.
        enable_gradient_checkpointing: If True, enables gradient
            checkpointing on the ESM-2 encoder for ~40% activation
            memory reduction. When QLoRA is enabled, gradient
            checkpointing is delegated to ``prepare_model_for_kbit_training``
            so that input gradients are correctly enabled.

    Returns:
        A PEFT-wrapped ESM-2 model with LoRA adapters applied to the
        configured target modules. Only LoRA parameters are trainable.
    """
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import EsmModel

    backbone_cfg = cfg.model.backbone
    lora_cfg = cfg.model.lora
    quant_cfg = cfg.model.get("quantization", {}) or {}
    use_quantization = bool(quant_cfg.get("enabled", False))
    use_sdpa = bool(backbone_cfg.get("use_sdpa_attention", False))
    model_name = backbone_cfg.name

    if use_sdpa:
        from src.models.sdpa_patch import patch_esm_sdpa

        patch_esm_sdpa()

    logger.info(f"Loading pre-trained backbone: {model_name}")

    if use_quantization:
        base_model = _load_quantized_backbone(
            model_name=model_name,
            quant_cfg=quant_cfg,
            enable_gradient_checkpointing=enable_gradient_checkpointing,
        )
    else:
        base_model = EsmModel.from_pretrained(
            model_name,
            add_pooling_layer=False,
        )
        if enable_gradient_checkpointing:
            base_model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled.")

    # Configure LoRA
    target_modules = _resolve_lora_target_modules(
        base_model=base_model,
        configured_target_modules=list(lora_cfg.target_modules),
        use_quantization=use_quantization,
    )
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=lora_cfg.rank,
        lora_alpha=lora_cfg.alpha,
        lora_dropout=lora_cfg.dropout,
        target_modules=target_modules,
        bias=lora_cfg.get("bias", "none"),
    )

    # Apply LoRA adapters
    model = get_peft_model(cast(PreTrainedModel, base_model), lora_config)

    # Log parameter counts
    trainable, total = model.get_nb_trainable_parameters()
    logger.info(
        f"LoRA applied: {trainable:,} trainable / {total:,} total parameters "
        f"({100 * trainable / total:.2f}%)"
    )
    logger.info(f"Target modules: {target_modules}")
    logger.info(f"LoRA rank={lora_cfg.rank}, alpha={lora_cfg.alpha}")
    if use_quantization:
        logger.info(
            f"Quantization active: method={quant_cfg.get('method', 'nf4')}, "
            f"compute_dtype={quant_cfg.get('compute_dtype', 'bfloat16')}"
        )

    # from_pretrained() returns models in eval mode by default.
    model.train()

    return cast(PreTrainedModel, model)


def _resolve_lora_target_modules(
    base_model: nn.Module,
    configured_target_modules: list[str],
    use_quantization: bool,
) -> list[str]:
    """
    Resolve configured LoRA target patterns against the loaded backbone.

    In QLoRA mode, some modules can be replaced by bitsandbytes wrappers
    whose weights are still plain ``torch.nn.Parameter`` objects because
    the corresponding checkpoint entries were missing and got freshly
    initialized during load. PEFT's 4-bit adapter path assumes true
    ``Params4bit`` weights and crashes on these modules, so we filter
    them out here while preserving the original broad config patterns.
    """
    if not use_quantization:
        return configured_target_modules

    resolved_targets: list[str] = []
    skipped_targets: list[str] = []
    seen: set[str] = set()

    for module_name, module in base_model.named_modules():
        if not module_name or not _matches_lora_target(module_name, configured_target_modules):
            continue

        if not _supports_peft_bnb_lora_target(module):
            skipped_targets.append(module_name)
            continue

        if module_name not in seen:
            resolved_targets.append(module_name)
            seen.add(module_name)

    if resolved_targets:
        if skipped_targets:
            logger.warning(f"Skipping incompatible quantized LoRA targets: {skipped_targets}")
        return resolved_targets

    return configured_target_modules


def _matches_lora_target(module_name: str, configured_target_modules: list[str]) -> bool:
    """Return True if a module name matches any configured target."""
    return any(
        module_name == target or module_name.endswith(f".{target}")
        for target in configured_target_modules
    )


def _supports_peft_bnb_lora_target(module: nn.Module) -> bool:
    """Return whether PEFT's bitsandbytes adapter path can wrap a module."""
    try:
        import bitsandbytes as bnb
    except ImportError:
        return True

    if isinstance(module, bnb.nn.Linear4bit):
        return all(hasattr(module.weight, attr) for attr in ("compress_statistics", "quant_type"))

    if isinstance(module, bnb.nn.Linear8bitLt):
        return hasattr(module, "state") and hasattr(module, "index")

    return True


def get_embedding_dim(cfg: DotDict) -> int:
    """Return the embedding dimension from config."""
    return int(cfg.model.backbone.get("embedding_dim", 640))


def extract_sequence_representation(
    model_output: _HasLastHiddenState,
    attention_mask: torch.Tensor,
    pooling: str = "mean",
) -> torch.Tensor:
    """
    Extract a fixed-size representation from ESM-2 token-level outputs.

    Args:
        model_output: Output from the ESM-2 forward pass
            (must have ``last_hidden_state``).
        attention_mask: Attention mask tensor of shape (B, L).
        pooling: Pooling strategy. One of:
            - "mean": Mean of all non-padding tokens.
            - "cls": CLS token (index 0).
            - "mean_cls": Concatenation of mean pooling and CLS.

    Returns:
        Tensor of shape (B, D) or (B, 2*D) for "mean_cls".
    """
    hidden_states = cast(torch.Tensor, model_output.last_hidden_state)

    if pooling == "cls":
        return hidden_states[:, 0, :]  # (B, D)

    elif pooling == "mean":
        # Mask padding tokens before averaging
        mask = attention_mask.unsqueeze(-1).float()  # (B, L, 1)
        summed = (hidden_states * mask).sum(dim=1)  # (B, D)
        counts = mask.sum(dim=1).clamp(min=1)  # (B, 1)
        return summed / counts  # (B, D)

    elif pooling == "mean_cls":
        cls_emb = hidden_states[:, 0, :]
        mask = attention_mask.unsqueeze(-1).float()
        summed = (hidden_states * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        mean_emb = summed / counts
        return torch.cat([cls_emb, mean_emb], dim=-1)  # (B, 2*D)

    else:
        raise ValueError(f"Unknown pooling strategy: '{pooling}'")


# ---------------------------------------------------------------------------
# Quantization helpers
# ---------------------------------------------------------------------------


def get_quantization_runtime_issue() -> str | None:
    """
    Return a human-readable QLoRA runtime compatibility issue, if any.

    This performs a lightweight preflight check before bitsandbytes tries
    to quantize weights on the active CUDA device. It is intentionally
    conservative: when the installed PyTorch build does not contain CUDA
    kernels for the detected GPU architecture, we fail fast with a clear
    message instead of letting the load crash deep inside CUDA.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                r"[\s\S]*is not compatible with the current PyTorch "
                r"installation\."
            ),
            category=UserWarning,
        )

        if not torch.cuda.is_available():
            return (
                "Quantization requires a CUDA-capable GPU but none was "
                "detected. Either disable model.quantization.enabled or run "
                "on a CUDA host."
            )

        try:
            device_idx = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(device_idx)
            major, minor = torch.cuda.get_device_capability(device_idx)
        except Exception as e:
            return (
                "Quantization requires a working CUDA runtime, but probing "
                f"the active device failed: {e}"
            )

    get_arch_list = getattr(torch.cuda, "get_arch_list", None)
    supported_arches = list(get_arch_list()) if callable(get_arch_list) else []
    supported_arches = [arch for arch in supported_arches if arch.startswith(("sm_", "compute_"))]

    required_sm = f"sm_{major}{minor}"
    required_compute = f"compute_{major}{minor}"
    if (
        supported_arches
        and required_sm not in supported_arches
        and required_compute not in supported_arches
    ):
        supported = ", ".join(supported_arches)
        return (
            f"Detected GPU '{device_name}' ({required_sm}), but the "
            "installed PyTorch build does not include CUDA kernels "
            f"for that architecture. Supported architectures: "
            f"{supported}. Install a newer CUDA-enabled PyTorch build "
            "for this GPU and a matching bitsandbytes wheel, or "
            "disable model.quantization.enabled."
        )

    return None


def assert_quantization_runtime_supported() -> None:
    """Raise if the local CUDA stack cannot run QLoRA quantization."""
    issue = get_quantization_runtime_issue()
    if issue is not None:
        raise QuantizationRuntimeUnsupportedError(issue)


def _load_quantized_backbone(
    model_name: str,
    quant_cfg: DotDict | dict[str, Any],
    enable_gradient_checkpointing: bool,
) -> PreTrainedModel:
    """
    Load an ESM-2 backbone in k-bit precision and prepare it for training.

    Performs three steps:

    1. Build a ``BitsAndBytesConfig`` from the quantization config.
    2. Load the backbone via ``from_pretrained`` with that config.
    3. Run ``prepare_model_for_kbit_training`` so that LayerNorms stay in
       fp32, input gradients are enabled (required by gradient
       checkpointing on quantized layers), and the model is ready for
       LoRA injection.

    Args:
        model_name: HuggingFace identifier of the ESM-2 checkpoint.
        quant_cfg: ``cfg.model.quantization`` section.
        enable_gradient_checkpointing: Whether to enable gradient
            checkpointing inside ``prepare_model_for_kbit_training``.

    Returns:
        A backbone module ready for LoRA wrapping.
    """
    from peft import prepare_model_for_kbit_training
    from transformers import EsmModel

    bnb_config = _build_bnb_config(quant_cfg)
    method = str(quant_cfg.get("method", "nf4")).lower()
    logger.info(f"Loading backbone in {method} precision via bitsandbytes")

    base_model = EsmModel.from_pretrained(
        model_name,
        add_pooling_layer=False,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
    )

    # PEFT helper: keeps LayerNorms in fp32 for stability, enables input
    # gradients (needed by activation checkpointing on quantized layers),
    # and optionally enables gradient checkpointing with use_reentrant=False.
    prepared_model = prepare_model_for_kbit_training(
        base_model,
        use_gradient_checkpointing=enable_gradient_checkpointing,
    )
    if enable_gradient_checkpointing:
        logger.info("Gradient checkpointing enabled (via prepare_model_for_kbit_training).")

    return cast(PreTrainedModel, prepared_model)


def _build_bnb_config(quant_cfg: DotDict | dict[str, Any]) -> Any:
    """
    Build a ``BitsAndBytesConfig`` from the project quantization section.

    Validates that the runtime supports quantization (CUDA + bitsandbytes
    installed) and raises with an actionable error message otherwise.

    Args:
        quant_cfg: ``cfg.model.quantization`` section.

    Returns:
        A ``transformers.BitsAndBytesConfig`` instance.

    Raises:
        QuantizationRuntimeUnsupportedError: If the CUDA runtime is missing
            or incompatible with the active GPU.
        ImportError: If bitsandbytes is not installed.
        ValueError: If the requested method is unknown.
    """
    assert_quantization_runtime_supported()

    try:
        import bitsandbytes  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Quantization requires the 'bitsandbytes' package. Install it with: uv add bitsandbytes"
        ) from e

    from transformers import BitsAndBytesConfig

    method = str(quant_cfg.get("method", "nf4")).lower()
    compute_dtype_str = str(quant_cfg.get("compute_dtype", "bfloat16")).lower()
    double_quant = bool(quant_cfg.get("double_quant", True))

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if compute_dtype_str not in dtype_map:
        raise ValueError(
            f"Unknown compute_dtype '{compute_dtype_str}'. Supported: {sorted(dtype_map)}"
        )
    compute_dtype = dtype_map[compute_dtype_str]

    if method in ("nf4", "fp4"):
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=method,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=double_quant,
        )
    elif method == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    else:
        raise ValueError(
            f"Unknown quantization method: '{method}'. Supported: 'nf4', 'fp4', 'int8'."
        )
