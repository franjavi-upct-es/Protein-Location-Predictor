# src/training/auto_batch_size.py
"""
Empirical batch-size auto-tuning via OOM binary search.

The hardware profile in ``configs/hardware/gpu_profiles.yaml`` is a
useful first guess but it cannot account for runtime variables such as
the chosen sequence length, LoRA rank, gradient checkpointing toggle,
quantization, or fragmentation pressure from other processes. This
module finds the largest batch size that actually runs on the current
machine for the current configuration by trying candidates and catching
``torch.cuda.OutOfMemoryError``.

The probe is intentionally fast: it builds the real model once, then
runs a forward + backward + zero-grad pass on a synthetic batch. No
data download, no tokenizer roundtrip, no optimizer state allocation
beyond what a single step requires. Total runtime is typically a few
seconds even for ESM-2 t33.

Results are cached on disk so a re-run with the same configuration is
instant. The cache key is a hash of the relevant configuration fields.

Usage (programmatic):
    from src.training.auto_batch_size import find_max_batch_size
    bs = find_max_batch_size(cfg, max_seq_length=1024)

Usage (CLI):
    uv run python -m src.training.auto_batch_size
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import torch

from src.utils.config import DotDict, load_config
from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

# Default candidate batch sizes to probe, in descending order
_DEFAULT_CANDIDATES = (32, 24, 16, 12, 8, 6, 4, 3, 2, 1)

# Safety margin: take the largest fitting size and divide by this factor
# to leave headroom for fragmentation and small variations across batches
_SAFETY_DIVISOR = 1


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------


def _config_fingerprint(cfg: DotDict, max_seq_length: int) -> str:
    """Build a stable hash of the fields that influence VRAM usage."""
    relevant = {
        "backbone_name": str(cfg.model.backbone.get("name")),
        "lora_rank": int(cfg.model.lora.get("rank", 8)),
        "lora_target_modules": list(cfg.model.lora.get("target_modules", [])),
        "use_sdpa": bool(cfg.model.backbone.get("use_sdpa_attention", False)),
        "quantization": dict(cfg.model.get("quantization", {}) or {}),
        "gradient_checkpointing": bool(cfg.training.get("gradient_checkpointing", True)),
        "precision": str(cfg.training.get("precision", "32")),
        "max_seq_length": int(max_seq_length),
    }
    payload = json.dumps(relevant, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _cache_path(cfg: DotDict) -> Path:
    """Return the path of the auto-batch-size cache file."""
    project_root = Path(cfg.get("project_root", "."))
    return project_root / ".cache" / "auto_batch_size.json"


def _load_cached(cfg: DotDict, fingerprint: str) -> int | None:
    """Return the cached batch size for ``fingerprint`` or None."""
    path = _cache_path(cfg)
    if not path.exists():
        return None
    try:
        cache = json.loads(path.read_text())
        return int(cache.get(fingerprint)) if fingerprint in cache else None
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


def _store_cached(cfg: DotDict, fingerprint: str, batch_size: int) -> None:
    """Persist a result into the cache file."""
    path = _cache_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    cache: dict[str, int] = {}
    if path.exists():
        try:
            cache = json.loads(path.read_text())
        except json.JSONDecodeError:
            cache = {}
    cache[fingerprint] = int(batch_size)
    path.write_text(json.dumps(cache, indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------


def _make_synthetic_batch(
    batch_size: int, seq_length: int, num_classes: int, device: str
) -> dict[str, torch.Tensor]:
    """Build a synthetic batch matching the model's expected interface."""
    return {
        "input_ids": torch.randint(4, 28, (batch_size, seq_length), device=device),
        "attention_mask": torch.ones((batch_size, seq_length), dtype=torch.long, device=device),
        "labels": torch.randint(0, 2, (batch_size, num_classes), device=device).float(),
    }


def _try_batch_size(
    model: Any,
    batch_size: int,
    seq_length: int,
    num_classes: int,
    device: str,
) -> bool:
    """
    Try a single forward + backward pass at the requested batch size.

    Returns True if it fits, False on OOM. Always frees the allocated
    memory before returning.
    """
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        batch = _make_synthetic_batch(batch_size, seq_length, num_classes, device)

        # Forward
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

        # Synthetic loss (BCE on the multi-label logits) + backward
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch["labels"])
        loss.backward()

        # Cleanup
        for p in model.parameters():
            if p.grad is not None:
                p.grad = None
        del logits, loss, batch
        gc.collect()
        torch.cuda.empty_cache()
        return True

    except torch.cuda.OutOfMemoryError:
        for p in model.parameters():
            if p.grad is not None:
                p.grad = None
        gc.collect()
        torch.cuda.empty_cache()
        return False
    except RuntimeError as e:
        # Some bnb / SDPA paths raise RuntimeError instead of OutOfMemoryError
        if "out of memory" in str(e).lower():
            for p in model.parameters():
                if p.grad is not None:
                    p.grad = None
            gc.collect()
            torch.cuda.empty_cache()
            return False
        raise


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def find_max_batch_size(
    cfg: DotDict,
    max_seq_length: int,
    num_classes: int = 10,
    candidates: tuple[int, ...] = _DEFAULT_CANDIDATES,
    use_cache: bool = True,
) -> int:
    """
    Find the largest batch size that fits in VRAM for the current config.

    Args:
        cfg: Project configuration. Must contain ``model`` and ``training``.
        max_seq_length: Sequence length to probe with.
        num_classes: Number of output classes for the synthetic head.
        candidates: Batch sizes to probe, in descending order.
        use_cache: If True, read from and write to the on-disk cache.

    Returns:
        The largest batch size that ran a forward+backward pass without
        OOM. Returns 1 if even batch size 1 fails (caller should treat
        this as an error condition).
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA not available — auto batch-size returning 1 (CPU fallback)")
        return 1

    fingerprint = _config_fingerprint(cfg, max_seq_length)

    if use_cache:
        cached = _load_cached(cfg, fingerprint)
        if cached is not None:
            logger.info(f"Auto batch-size: cache hit (fingerprint={fingerprint}) -> {cached}")
            return cached

    logger.info(
        f"Auto batch-size: probing (fingerprint={fingerprint}, seq_length={max_seq_length})..."
    )

    # Build the real model once. This dominates probe time but is the
    # only way to get realistic VRAM estimates.
    from src.models.classifier_head import ClassifierHead
    from src.models.esm_lora import (
        build_esm_lora_backbone,
        extract_sequence_representation,
        get_embedding_dim,
    )

    backbone = build_esm_lora_backbone(
        cfg,
        enable_gradient_checkpointing=cfg.training.get("gradient_checkpointing", True),
    )
    emb_dim = get_embedding_dim(cfg)
    head = ClassifierHead.from_config(cfg, emb_dim, num_classes)

    class _Probe(torch.nn.Module):
        def __init__(
            self,
            backbone_module: torch.nn.Module,
            head_module: torch.nn.Module,
        ) -> None:
            super().__init__()
            self.backbone = backbone_module
            self.head = head_module

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
        ) -> torch.Tensor:
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            pooled = extract_sequence_representation(
                out,
                attention_mask,
                pooling="mean",
            )
            return cast(torch.Tensor, self.head(pooled))

    model = _Probe(backbone, head).to("cuda")
    model.train()

    # Probe candidates from largest to smallest
    fitted = 1
    for candidate in candidates:
        logger.info(f"  Trying batch_size={candidate}...")
        if _try_batch_size(model, candidate, max_seq_length, num_classes, "cuda"):
            fitted = candidate
            logger.info(f"  -> batch_size={candidate} fits")
            break
        logger.info(f"  -> batch_size={candidate} OOM")

    # Apply safety divisor (currently 1, kept for future tuning)
    final = max(1, fitted // _SAFETY_DIVISOR)

    logger.info(f"Auto batch-size result: {final} (fingerprint={fingerprint})")

    # Cleanup the probe model
    del model, backbone, head
    gc.collect()
    torch.cuda.empty_cache()

    if use_cache:
        _store_cached(cfg, fingerprint, final)

    return final


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for standalone batch-size probing."""
    parser = argparse.ArgumentParser(
        description="Find the largest batch size that fits on this GPU."
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=None,
        help="Sequence length to probe. Defaults to the configured max.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore the on-disk cache and re-probe.",
    )
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=[],
        help="Config overrides in key=value format.",
    )
    args = parser.parse_args()

    cfg = load_config(mode="training", overrides=args.overrides)
    setup_logging(level=cfg.project.log_level)

    seq_length = args.seq_length
    if seq_length is None:
        seq_length = cfg.training.get("max_sequence_length") or int(
            cfg.model.backbone.get("max_position_embeddings", 1024)
        )

    bs = find_max_batch_size(cfg, max_seq_length=seq_length, use_cache=not args.no_cache)
    print(f"\nMax batch size for seq_length={seq_length}: {bs}")


if __name__ == "__main__":
    main()
