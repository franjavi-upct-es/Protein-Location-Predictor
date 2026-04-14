# src/models/esm_lora.py
"""
ESM-2 backbone with LoRA adapters via PEFT, with optional QLoRA support.

Loads the pre-trained ESM-2 model and injects low-rank adapter matrices
into the attention layers, enabling parameter-efficient fine-tuning that
fits within 8 GB VRAM. When quantization is enabled in the configuration,
the base model is loaded in 4-bit (NF4 by default) via bitsandbytes,
which dramatically reduces VRAM usage and unlocks larger backbones on
consumer GPUs (the QLoRA path).

Usage:
    from src.models.esm_lora import build_esm_lora_backbone
    backbone = build_esm_lora_backbone(cfg)
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from src.utils.config import DotDict
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_esm_lora_backbone(
    cfg: DotDict, enable_gradient_checkpointing: bool = True
) -> nn.Module:
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
    model_name = backbone_cfg.name

    logger.info(f"Loading pre-trained backbone: {model_name}")

    if use_quantization:
        base_model = _load_quantized_backbone(
            model_name=model_name,
            quant_cfg=quant_cfg,
            enable_gradient_checkpointing=enable_gradient_checkpointing,
        )
    else:
        base_model = EsmModel.from_pretrained(model_name)
        if enable_gradient_checkpointing:
            base_model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled.")

    # Configure LoRA
    target_modules = list(lora_cfg.target_modules)
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=lora_cfg.rank,
        lora_alpha=lora_cfg.alpha,
        lora_dropout=lora_cfg.dropout,
        target_modules=target_modules,
        bias=lora_cfg.get("bias", "none"),
    )

    # Apply LoRA adapters
    model = get_peft_model(base_model, lora_config)

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

    return model


def get_embedding_dim(cfg: DotDict) -> int:
    """Return the embedding dimension from config."""
    return int(cfg.model.backbone.get("embedding_dim", 640))


def extract_sequence_representation(
    model_output: Any, attention_mask: torch.Tensor, pooling: str = "mean"
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
    hidden_states = model_output.last_hidden_state  # (B, L, D)

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


def _load_quantized_backbone(
    model_name: str,
    quant_cfg: DotDict | dict[str, Any],
    enable_gradient_checkpointing: bool,
) -> nn.Module:
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
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
    )

    # PEFT helper: keeps LayerNorms in fp32 for stability, enables input
    # gradients (needed by activation checkpointing on quantized layers),
    # and optionally enables gradient checkpointing with use_reentrant=False.
    base_model = prepare_model_for_kbit_training(
        base_model,
        use_gradient_checkpointing=enable_gradient_checkpointing,
    )
    if enable_gradient_checkpointing:
        logger.info(
            "Gradient checkpointing enabled "
            "(via prepare_model_for_kbit_training)."
        )

    return base_model


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
        RuntimeError: If CUDA is not available.
        ImportError: If bitsandbytes is not installed.
        ValueError: If the requested method is unknown.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Quantization requires a CUDA-capable GPU but none was "
            "detected. Either disable model.quantization.enabled or run "
            "on a CUDA host."
        )

    try:
        import bitsandbytes  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Quantization requires the 'bitsandbytes' package. Install "
            "it with: uv add bitsandbytes"
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
            f"Unknown compute_dtype '{compute_dtype_str}'. "
            f"Supported: {sorted(dtype_map)}"
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
            f"Unknown quantization method: '{method}'. "
            "Supported: 'nf4', 'fp4', 'int8'."
        )
