# tests/unit/test_sdpa_patch.py
"""Tests for the ESM SDPA enablement helper."""

from __future__ import annotations

from typing import Any

from transformers import EsmModel

from src.models.sdpa_patch import (
    is_esm_sdpa_patched,
    patch_esm_sdpa,
    unpatch_esm_sdpa,
)


def test_patch_injects_sdpa_when_unspecified(
    monkeypatch,
) -> None:
    """The patch should default future ESM loads to SDPA."""
    captured: dict[str, Any] = {}

    def fake_from_pretrained(cls, name: str, *args: Any, **kwargs: Any) -> dict:
        captured["name"] = name
        captured["kwargs"] = kwargs
        return {"name": name, "kwargs": kwargs}

    monkeypatch.setattr(
        EsmModel,
        "from_pretrained",
        classmethod(fake_from_pretrained),
    )

    assert patch_esm_sdpa() is True
    assert is_esm_sdpa_patched() is True

    EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")

    assert captured["name"] == "facebook/esm2_t6_8M_UR50D"
    assert captured["kwargs"]["attn_implementation"] == "sdpa"

    assert unpatch_esm_sdpa() is True
    assert is_esm_sdpa_patched() is False


def test_patch_respects_explicit_attention_choice(
    monkeypatch,
) -> None:
    """An explicit attention implementation should win over the default."""
    captured: dict[str, Any] = {}

    def fake_from_pretrained(cls, name: str, *args: Any, **kwargs: Any) -> dict:
        captured["kwargs"] = kwargs
        return {"name": name, "kwargs": kwargs}

    monkeypatch.setattr(
        EsmModel,
        "from_pretrained",
        classmethod(fake_from_pretrained),
    )

    assert patch_esm_sdpa() is True

    EsmModel.from_pretrained(
        "facebook/esm2_t6_8M_UR50D",
        attn_implementation="eager",
    )

    assert captured["kwargs"]["attn_implementation"] == "eager"

    assert unpatch_esm_sdpa() is True
