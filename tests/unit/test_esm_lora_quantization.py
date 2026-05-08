# tests/unit/test_esm_lora_quantization.py
"""Tests for the QLoRA quantization path of the ESM-2 backbone builder."""

from __future__ import annotations

import importlib.util
from typing import Any

import pytest
import torch

from src.models.esm_lora import _build_bnb_config, build_esm_lora_backbone
from src.utils.config import DotDict


def _bitsandbytes_available() -> bool:
    return importlib.util.find_spec("bitsandbytes") is not None


# ---------------------------------------------------------------------------
# _build_bnb_config — pure validation logic
# ---------------------------------------------------------------------------


class TestBuildBnbConfigValidation:
    """Tests that don't require a GPU or bitsandbytes."""

    def test_raises_without_cuda(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If CUDA is not available, _build_bnb_config must raise loudly."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        with pytest.raises(RuntimeError, match="CUDA-capable GPU"):
            _build_bnb_config({"method": "nf4"})

    def test_unknown_compute_dtype_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Pretending CUDA + bnb are present, an unknown dtype must raise."""
        monkeypatch.setattr(
            "src.models.esm_lora.assert_quantization_runtime_supported", lambda: None
        )
        if not _bitsandbytes_available():
            pytest.skip("bitsandbytes not installed")
        with pytest.raises(ValueError, match="Unknown compute_dtype"):
            _build_bnb_config({"method": "nf4", "compute_dtype": "float128"})

    def test_unknown_method_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "src.models.esm_lora.assert_quantization_runtime_supported", lambda: None
        )
        if not _bitsandbytes_available():
            pytest.skip("bitsandbytes not installed")
        with pytest.raises(ValueError, match="Unknown quantization method"):
            _build_bnb_config({"method": "int2"})


# ---------------------------------------------------------------------------
# _build_bnb_config — happy path (requires CUDA + bnb)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
class TestBuildBnbConfigHappyPath:
    """Happy-path tests that require an actual CUDA + bitsandbytes setup."""

    def setup_method(self) -> None:
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        if not _bitsandbytes_available():
            pytest.skip("bitsandbytes not installed")

    def test_nf4_default(self) -> None:
        cfg = {"method": "nf4"}
        bnb = _build_bnb_config(cfg)
        assert bnb.load_in_4bit is True
        assert bnb.bnb_4bit_quant_type == "nf4"
        assert bnb.bnb_4bit_use_double_quant is True
        assert bnb.bnb_4bit_compute_dtype == torch.bfloat16

    def test_int8(self) -> None:
        cfg = {"method": "int8"}
        bnb = _build_bnb_config(cfg)
        assert bnb.load_in_8bit is True

    def test_double_quant_disabled(self) -> None:
        cfg = {"method": "nf4", "double_quant": False}
        bnb = _build_bnb_config(cfg)
        assert bnb.bnb_4bit_use_double_quant is False


# ---------------------------------------------------------------------------
# End-to-end smoke test (very small backbone, real GPU)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.slow
class TestQLoRABackboneSmoke:
    """
    End-to-end smoke test that loads the smallest ESM-2 variant (8 M
    parameters) in 4-bit, applies LoRA, and runs a forward pass.

    Skipped automatically when CUDA or bitsandbytes are not available.
    Uses ``facebook/esm2_t6_8M_UR50D`` to keep the test fast and tiny.
    """

    BACKBONE = "facebook/esm2_t6_8M_UR50D"

    def setup_method(self) -> None:
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        if not _bitsandbytes_available():
            pytest.skip("bitsandbytes not installed")

    def _make_cfg(self) -> DotDict:
        return DotDict.from_dict(
            {
                "model": {
                    "backbone": {
                        "name": self.BACKBONE,
                        "embedding_dim": 320,
                    },
                    "lora": {
                        "rank": 4,
                        "alpha": 8,
                        "dropout": 0.0,
                        "target_modules": ["query", "key", "value", "dense"],
                        "bias": "none",
                    },
                    "quantization": {
                        "enabled": True,
                        "method": "nf4",
                        "compute_dtype": "bfloat16",
                        "double_quant": True,
                    },
                }
            }
        )

    def test_loads_and_runs_forward_pass(self) -> None:
        cfg = self._make_cfg()
        model = build_esm_lora_backbone(cfg, enable_gradient_checkpointing=True)
        model = model.to("cuda")
        model.eval()

        # Build a fake batch of 2 short sequences
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.BACKBONE)
        encoded = tokenizer(
            ["MSKGEELFTGVVPILVELDG", "MQIFVKTLTGKTITLEVEPSDT"],
            return_tensors="pt",
            padding=True,
        ).to("cuda")

        with torch.no_grad():
            output: Any = model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
            )

        # PEFT-wrapped models still expose the underlying outputs
        assert hasattr(output, "last_hidden_state")
        assert output.last_hidden_state.shape[0] == 2
        assert output.last_hidden_state.shape[2] == 320

    def test_only_lora_params_are_trainable(self) -> None:
        cfg = self._make_cfg()
        model = build_esm_lora_backbone(cfg, enable_gradient_checkpointing=True)
        trainable, total = model.get_nb_trainable_parameters()
        assert trainable > 0
        assert trainable < total
        # LoRA should be a small fraction of the full model
        assert trainable / total < 0.05
