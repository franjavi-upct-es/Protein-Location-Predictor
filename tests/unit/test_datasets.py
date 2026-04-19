# tests/unit/test_datasets.py
"""Tests for the ProteinDataset and collation utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.datasets import (
    ProteinDataModule,
    ProteinDataset,
    dynamic_padding_collate,
)
from src.utils.config import DotDict

# ---------------------------------------------------------------------------
# Mock tokenizer (avoids downloading ESM-2 for unit tests)
# ---------------------------------------------------------------------------


class MockTokenizer:
    """Minimal tokenizer mock that returns fixed-size tensors."""

    def __call__(
        self,
        sequence: str,
        return_tensors: str = "pt",
        padding: str = "max_length",
        truncation: bool = True,
        max_length: int = 64,
    ) -> dict[str, torch.Tensor]:
        seq_len = min(len(sequence) + 2, max_length)  # +2 for CLS/EOS
        if padding is False:
            input_ids = torch.randint(3, 30, (1, seq_len))
            attention_mask = torch.ones(1, seq_len, dtype=torch.long)
        else:
            input_ids = torch.ones(1, max_length, dtype=torch.long)
            input_ids[0, :seq_len] = torch.randint(3, 30, (seq_len,))
            attention_mask = torch.zeros(1, max_length, dtype=torch.long)
            attention_mask[0, :seq_len] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "accession": ["P001", "P002", "P003"],
            "sequence": ["MSKGEEL" * 10, "MQIFVKT" * 5, "MGLSDGE" * 8],
            "locations": [["Nucleus", "Cytoplasm"], ["Membrane"], ["Nucleus"]],
        }
    )


@pytest.fixture()
def label_list() -> list[str]:
    return ["Cytoplasm", "Membrane", "Nucleus"]


@pytest.fixture()
def tokenizer() -> MockTokenizer:
    return MockTokenizer()


# ---------------------------------------------------------------------------
# ProteinDataset
# ---------------------------------------------------------------------------


class TestProteinDataset:
    """Tests for the ProteinDataset class."""

    def test_length(
        self,
        sample_df: pd.DataFrame,
        tokenizer: MockTokenizer,
        label_list: list[str],
    ) -> None:
        ds = ProteinDataset(sample_df, tokenizer, label_list, max_length=64)
        assert len(ds) == 3

    def test_item_keys(
        self,
        sample_df: pd.DataFrame,
        tokenizer: MockTokenizer,
        label_list: list[str],
    ) -> None:
        ds = ProteinDataset(sample_df, tokenizer, label_list, max_length=64)
        item = ds[0]
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item

    def test_label_vector_shape(
        self,
        sample_df: pd.DataFrame,
        tokenizer: MockTokenizer,
        label_list: list[str],
    ) -> None:
        ds = ProteinDataset(sample_df, tokenizer, label_list, max_length=64)
        item = ds[0]
        assert item["labels"].shape == (len(label_list),)

    def test_multi_hot_encoding(
        self,
        sample_df: pd.DataFrame,
        tokenizer: MockTokenizer,
        label_list: list[str],
    ) -> None:
        ds = ProteinDataset(sample_df, tokenizer, label_list, max_length=64)
        # First protein: Nucleus + Cytoplasm
        labels = ds[0]["labels"]
        cyto_idx = label_list.index("Cytoplasm")
        nuc_idx = label_list.index("Nucleus")
        mem_idx = label_list.index("Membrane")
        assert labels[cyto_idx] == 1.0
        assert labels[nuc_idx] == 1.0
        assert labels[mem_idx] == 0.0

    def test_single_label_encoding(
        self,
        sample_df: pd.DataFrame,
        tokenizer: MockTokenizer,
        label_list: list[str],
    ) -> None:
        ds = ProteinDataset(sample_df, tokenizer, label_list, max_length=64)
        # Second protein: Membrane only
        labels = ds[1]["labels"]
        assert labels.sum() == 1.0
        assert labels[label_list.index("Membrane")] == 1.0

    def test_with_locations_str(self, tokenizer: MockTokenizer, label_list: list[str]) -> None:
        df = pd.DataFrame(
            {
                "accession": ["P001"],
                "sequence": ["MSKGEEL" * 10],
                "locations_str": ["Nucleus|Cytoplasm"],
            }
        )
        ds = ProteinDataset(df, tokenizer, label_list, max_length=64)
        labels = ds[0]["labels"]
        assert labels[label_list.index("Nucleus")] == 1.0
        assert labels[label_list.index("Cytoplasm")] == 1.0

    def test_external_features(
        self,
        sample_df: pd.DataFrame,
        tokenizer: MockTokenizer,
        label_list: list[str],
    ) -> None:
        ext = np.random.rand(3, 5).astype(np.float32)
        ds = ProteinDataset(
            sample_df,
            tokenizer,
            label_list,
            max_length=64,
            external_features=ext,
        )
        item = ds[0]
        assert "external_features" in item
        assert item["external_features"].shape == (5,)


# ---------------------------------------------------------------------------
# Dynamic padding collation
# ---------------------------------------------------------------------------


class TestDynamicPaddingCollate:
    """Tests for batch collation with dynamic padding."""

    def test_batch_shape(
        self,
        sample_df: pd.DataFrame,
        tokenizer: MockTokenizer,
        label_list: list[str],
    ) -> None:
        ds = ProteinDataset(sample_df, tokenizer, label_list, max_length=64)
        batch = [ds[i] for i in range(3)]
        collated = dynamic_padding_collate(batch)

        assert collated["input_ids"].shape[0] == 3
        assert collated["attention_mask"].shape[0] == 3
        assert collated["labels"].shape == (3, len(label_list))

    def test_padding_alignment(
        self,
        sample_df: pd.DataFrame,
        tokenizer: MockTokenizer,
        label_list: list[str],
    ) -> None:
        ds = ProteinDataset(sample_df, tokenizer, label_list, max_length=64)
        batch = [ds[i] for i in range(3)]
        collated = dynamic_padding_collate(batch)

        # All input_ids in the batch should have the same length
        assert collated["input_ids"].shape[1] == collated["attention_mask"].shape[1]

    def test_dynamic_padding_does_not_pad_to_dataset_max_length(
        self,
        sample_df: pd.DataFrame,
        tokenizer: MockTokenizer,
        label_list: list[str],
    ) -> None:
        ds = ProteinDataset(sample_df, tokenizer, label_list, max_length=128)
        batch = [ds[i] for i in range(3)]
        collated = dynamic_padding_collate(batch)

        assert collated["input_ids"].shape[1] < 128


# ---------------------------------------------------------------------------
# Lightning DataModule
# ---------------------------------------------------------------------------


@pytest.fixture()
def datamodule_cfg(tmp_path: Path) -> DotDict:
    splits_dir = tmp_path / "data" / "splits"
    splits_dir.mkdir(parents=True)

    pd.DataFrame(
        {
            "accession": ["P001", "P002"],
            "sequence": ["MSKGEEL" * 6, "MQIFVKT" * 5],
            "locations_str": ["Nucleus|Cytoplasm", "Membrane"],
        }
    ).to_csv(splits_dir / "train.csv", index=False)
    pd.DataFrame(
        {
            "accession": ["P003"],
            "sequence": ["MGLSDGE" * 6],
            "locations_str": ["Nucleus"],
        }
    ).to_csv(splits_dir / "val.csv", index=False)
    pd.DataFrame(
        {
            "accession": ["P004"],
            "sequence": ["MTEYKLV" * 6],
            "locations_str": ["Cytoplasm"],
        }
    ).to_csv(splits_dir / "test.csv", index=False)

    return DotDict.from_dict(
        {
            "project_root": str(tmp_path),
            "paths": {"splits_dir": "data/splits"},
            "training": {
                "batch_size": 2,
                "num_workers": 0,
                "pin_memory": False,
            },
            "model": {
                "backbone": {
                    "name": "facebook/esm2_t30_150M_UR50D",
                    "max_position_embeddings": 64,
                }
            },
        }
    )


class TestProteinDataModule:
    """Tests for the Lightning DataModule wrapper."""

    def test_constructor_sets_cfg(
        self,
        datamodule_cfg: DotDict,
        label_list: list[str],
    ) -> None:
        dm = ProteinDataModule(
            datamodule_cfg,
            label_list=label_list,
            tokenizer=MockTokenizer(),
        )

        assert dm.cfg is datamodule_cfg
        assert dm.batch_size == 2
        assert dm.max_length == 64

    def test_setup_loads_fit_and_test_splits(
        self,
        datamodule_cfg: DotDict,
        label_list: list[str],
    ) -> None:
        dm = ProteinDataModule(
            datamodule_cfg,
            label_list=label_list,
            tokenizer=MockTokenizer(),
        )

        dm.setup()

        assert dm.train_dataset is not None
        assert dm.val_dataset is not None
        assert dm.test_dataset is not None
        assert len(dm.train_dataset) == 2
        assert len(dm.val_dataset) == 1
        assert len(dm.test_dataset) == 1

    def test_uses_training_max_sequence_length(
        self,
        datamodule_cfg: DotDict,
        label_list: list[str],
    ) -> None:
        datamodule_cfg.training.max_sequence_length = 32
        dm = ProteinDataModule(
            datamodule_cfg,
            label_list=label_list,
            tokenizer=MockTokenizer(),
        )

        assert dm.max_length == 32
