# src/models/sdpa_patch.py
"""
Optional SDPA enablement helper for ESM-2.

Recent Hugging Face Transformers releases expose ESM attention through
the standard ``attn_implementation`` hook. This module keeps the
project's existing ``patch_esm_sdpa()`` / ``unpatch_esm_sdpa()`` API,
but now implements it by wrapping ``EsmModel.from_pretrained`` so that
future loads default to ``attn_implementation="sdpa"`` unless the caller
explicitly asks for a different implementation.

The patch is process-local, idempotent, and only affects models loaded
after it has been applied. ``unpatch_esm_sdpa()`` restores the original
``from_pretrained`` implementation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from src.utils.logging import get_logger

logger = get_logger(__name__)


_PATCH_FLAG = "_sdpa_patched"
_ORIGINAL_FROM_PRETRAINED_ATTR = "_original_from_pretrained_pre_sdpa"
_FromPretrainedCallable = Callable[..., Any]


def patch_esm_sdpa() -> bool:
    """
    Make future ``EsmModel.from_pretrained`` calls default to SDPA.

    Returns True when the helper is available and patched (including when
    it was already patched), otherwise False.
    """
    try:
        from transformers import EsmModel
    except ImportError:
        logger.warning("transformers not available — SDPA patch not applied")
        return False

    if getattr(EsmModel, _PATCH_FLAG, False):
        logger.debug("ESM SDPA patch already applied — skipping")
        return True

    original_method = cast(Any, EsmModel.from_pretrained)
    original = cast(
        _FromPretrainedCallable,
        getattr(original_method, "__func__", original_method),
    )
    setattr(EsmModel, _ORIGINAL_FROM_PRETRAINED_ATTR, original)
    type.__setattr__(
        EsmModel,
        "from_pretrained",
        classmethod(_esm_from_pretrained_sdpa),
    )
    setattr(EsmModel, _PATCH_FLAG, True)

    logger.info(
        "Patched EsmModel.from_pretrained to default to "
        "attn_implementation='sdpa'"
    )
    return True


def unpatch_esm_sdpa() -> bool:
    """
    Restore the original ``EsmModel.from_pretrained`` implementation.

    Returns True if a restoration happened, False otherwise.
    """
    try:
        from transformers import EsmModel
    except ImportError:
        return False

    if not getattr(EsmModel, _PATCH_FLAG, False):
        return False

    original = getattr(EsmModel, _ORIGINAL_FROM_PRETRAINED_ATTR, None)
    if original is None:
        return False

    type.__setattr__(EsmModel, "from_pretrained", classmethod(original))
    delattr(EsmModel, _ORIGINAL_FROM_PRETRAINED_ATTR)
    setattr(EsmModel, _PATCH_FLAG, False)
    logger.info("Restored stock EsmModel.from_pretrained")
    return True


def is_esm_sdpa_patched() -> bool:
    """Return True if future ESM loads will default to SDPA."""
    try:
        from transformers import EsmModel
    except ImportError:
        return False
    return bool(getattr(EsmModel, _PATCH_FLAG, False))


def _esm_from_pretrained_sdpa(
    cls: Any,
    pretrained_model_name_or_path: str,
    *model_args: Any,
    **kwargs: Any,
) -> Any:
    """Wrapper that defaults ESM loads to SDPA when unspecified."""
    original = cast(
        _FromPretrainedCallable | None,
        getattr(cls, _ORIGINAL_FROM_PRETRAINED_ATTR, None),
    )
    if original is None:
        raise RuntimeError(
            "ESM SDPA patch lost reference to the original "
            "from_pretrained implementation."
        )

    kwargs.setdefault("attn_implementation", "sdpa")
    return original(
        cls,
        pretrained_model_name_or_path,
        *model_args,
        **kwargs,
    )
