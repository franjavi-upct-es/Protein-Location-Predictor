# src/models/esm_lora.py
"""
ESM-2 backbone with LoRA adapters via PEFT.

Loads the pre-trained ESM-2 model and injects low-rank adapter matrices
into the attention layers, enabling parameter-efficient fine-tuning that
fits within 8 BG VRAM.

Usage:
    from src.models.esm_lora import build_esm_lora_backbone
    backbone = build_esm_lora_backbone(cfg, hw_profile)
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from src.utils.config import DotDict
from src.utils.logging import get_logger

logger = get_logger(__name__)


def build_esm_lora_backbone(
    cfg: DotDict, enable_gradient_checkpointing: bool = True
) -> nn.Module:
    """
    Build a ESM-2 backbone with LoRA adapters.

    Args:
        cfg: Project configuration
            (needs model.backbone and model.lora sections).
        enable_gradient_checkpointing: If True, enables gradient checkpointing
            on the ESM-2 encode for ~40% activation memory reduction.

    Returns:
        A PEFT-wrapped ESM-2 model with LoRA adapaters applied to the
        configured target modules. Only LoRA parameters are trainable.
    """
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import EsmModel

    backbone_cfg = cfg.model.backbone
    lora_cfg = cfg.model.lora
    model_name = backbone_cfg.name

    # Load pre-trained ESM-2
    logger.info(f"Loading pre-trained backbone: {model_name}")
    base_model = EsmModel.from_pretrained(model_name)

    # Enable gradient checkpointing before wrapping with PEFT
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
    logger.info(f"LoRA rank{lora_cfg.rank}, alpha={lora_cfg.alpha}")

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
    Extract a fixed-size representation from ESm-2 token-level outputs.

    Args:
        model_output: Output from the ESM-2 forward pass (.last_hidden_state).
        attention_mask: Attention mask tensor (B, L).
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
        # Mask padding tolens before averaging
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
