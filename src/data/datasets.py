# src/data/datasets.py
"""
PyTorch Dataset and Lightning DataModule for protein sequences.

Provides on-the-fly tokenization with the ESM-2 tokenizer and multi-hot
label enconding for multi-label classification.

Usage::

    from src.data.datasets import ProteinDataModule
    dm = ProteinDataModule(cfg, label_list=["Nucleus", "Cytoplasm", ...])
    dm.setup()
    for batch in dm.train_dataloader():
        input_ids, attention_mask = labels = batch
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.utils.config import DotDict, resolve_path
from src.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class ProteinDataset(Dataset):
    """
    Dataset for protein sequences with multi-label localization target.

    Tokenizes sequences on-the-fly using the ESM-2 tokenizer and encodes
    labels as multi-hot vectors.

    Args:
        df: DataFrame with 'sequence' and 'locations' (list) or
            'locations_str' (pipe-separated) columns.
        tokenizer: HuggingFace tokenizer (ESM-2).
        label_list: Ordered list of all possible location labels.
        max_length: Maximum sequence length for tokenization.
        external_features: Optional numpy array of shape
            (n_samples, n_features) with precomputed external
            features (biophysical, etc.).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: Any,
        label_list: list[str],
        max_length: int = 1024,
        external_features: np.ndarray | None = None,
    ) -> None:
        self.sequences = df["sequence"].tolist()
        self.accessions = df["accession"].tolist()
        self.tokenizer = tokenizer
        self.label_list = label_list
        self.label_to_idx = {label: i for i, label in enumerate(label_list)}
        self.max_length = max_length
        self.external_features = external_features

        # Parse locations
        if "locations" in df.columns:
            self.locations = df["locations"].tolist()
        elif "locations_str" in df.columns:
            self.locations = (
                df["locations_str"]
                .apply(lambda s: s.split("|") if isinstance(s, str) else [])
                .tolist()
            )
        else:
            raise ValueError("DataFrame must have 'locations' or 'locations_str' column")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sequence = self.sequences[idx]
        locations = self.locations[idx]

        # Tokenize
        encoding = self.tokenizer(
            sequence,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.max_length,
        )

        # Multi-hot label vector
        label_vector = torch.zeros(len(self.label_list), dtype=torch.float32)
        for loc in locations:
            if loc in self.label_to_idx:
                label_vector[self.label_to_idx[loc]] = 1.0

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": label_vector,
        }

        if self.external_features is not None:
            item["external_features"] = torch.tensor(
                self.external_features[idx], dtype=torch.float32
            )

        return item


# ---------------------------------------------------------------------------
# Collation (dynamic padding for efficiency)
# ---------------------------------------------------------------------------


def dynamic_padding_collate(
    batch: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """
    Collate function that pads to the longest sequence in the batch.

    This is more memory-efficient than padding to max_length for every batch,
    especially when sequence lengths vary significantly.
    """
    # Find the actual max length in this batch
    max_len = max(item["input_ids"].size(0) for item in batch)

    collated: dict[str, list[torch.Tensor]] = {key: [] for key in batch[0]}

    for item in batch:
        seq_len = item["input_ids"].size(0)
        pad_len = max_len - seq_len

        if pad_len > 0:
            # Pad input_ids with tokenizer pad token (usually 1 for ESM-2)
            item["input_ids"] = torch.cat(
                [
                    item["input_ids"],
                    torch.ones(pad_len, dtype=item["input_ids"].dtype),
                ]
            )
            item["attention_mask"] = torch.cat(
                [
                    item["attention_mask"],
                    torch.zeros(pad_len, dtype=item["attention_mask"].dtype),
                ]
            )

        for key in collated:
            collated[key].append(item[key])

    return {key: torch.stack(vals) for key, vals in collated.items()}


# ---------------------------------------------------------------------------
# Lightning DataModule
# ---------------------------------------------------------------------------

try:
    import pytorch_lightning as pl

    class ProteinDataModule(pl.LightningDataModule):
        """
        Lightning DataModule for protein localization.

        Loads pre-split CSV files, creates DataSets with the ESM-2 tokenizer,
        and provides DataLoaders with dynamic padding.

        Args:
            cfg: Project configuration.
            label_list: Ordered list of all location labels.
            tokenizer: ESM-2 tokenizer instance. If None, laoded from config.
        """

        def __init__(
            self,
            cfg: DotDict,
            label_list: list[str],
            tokenizer: Any = None,
        ) -> None:
            super().__init__()
            self.cfg = cfg
            self.label_list = label_list
            self._tokenizer = tokenizer

            # Training params
            training_cfg = cfg.get("training", {})
            self.batch_size = training_cfg.get("batch_size", 2)
            self.num_workers = training_cfg.get("num_workers", 4)
            self.pin_memory = training_cfg.get("pin_memory", True)
            self.use_length_bucketing = training_cfg.get("use_length_bucketing", False)
            self.length_bucket_jitter = training_cfg.get("length_bucket_jitter", 0.05)
            backbone_max_length = cfg.model.backbone.get("max_position_embeddings", 1024)
            requested_max_length = training_cfg.get("max_sequence_length")
            self.max_length = (
                min(requested_max_length, backbone_max_length)
                if requested_max_length is not None
                else backbone_max_length
            )

            self.train_dataset: ProteinDataset | None = None
            self.val_dataset: ProteinDataset | None = None
            self.test_dataset: ProteinDataset | None = None

        @property
        def tokenizer(self) -> Any:
            if self._tokenizer is None:
                from transformers import AutoTokenizer

                model_name = self.cfg.model.backbone.name
                logger.info(f"Loading tokenizer: {model_name}")
                self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            return self._tokenizer

        def _load_split(self, name: str) -> pd.DataFrame:
            """Load a split CSV and reconstruct locations list."""
            path = resolve_path(self.cfg, "paths.splits_dir") / f"{name}.csv"
            df = pd.read_csv(path)
            if "locations_str" in df.columns:
                df["locations"] = df["locations_str"].apply(
                    lambda s: s.split("|") if isinstance(s, str) else []
                )

            return df

        def setup(self, stage: str | None = None) -> None:
            """Load data and create datasets."""
            from src.data.external_features import (
                compute_all_external_features,
                get_external_feature_dim,
            )

            has_ext = get_external_feature_dim(self.cfg) > 0

            if stage in ("fit", None):
                train_df = self._load_split("train")
                val_df = self._load_split("val")
                train_ext = (
                    compute_all_external_features(train_df["sequence"].tolist(), self.cfg)
                    if has_ext
                    else None
                )
                val_ext = (
                    compute_all_external_features(val_df["sequence"].tolist(), self.cfg)
                    if has_ext
                    else None
                )
                self.train_dataset = ProteinDataset(
                    train_df,
                    self.tokenizer,
                    self.label_list,
                    self.max_length,
                    external_features=train_ext,
                )
                self.val_dataset = ProteinDataset(
                    val_df,
                    self.tokenizer,
                    self.label_list,
                    self.max_length,
                    external_features=val_ext,
                )
                logger.info(
                    f"Loaded train ({len(self.train_dataset)}) "
                    f"and val ({len(self.val_dataset)}) datasets"
                )

            if stage in ("test", None):
                test_df = self._load_split("test")
                test_ext = (
                    compute_all_external_features(test_df["sequence"].tolist(), self.cfg)
                    if has_ext
                    else None
                )
                self.test_dataset = ProteinDataset(
                    test_df,
                    self.tokenizer,
                    self.label_list,
                    self.max_length,
                    external_features=test_ext,
                )
                logger.info(f"Loaded test ({len(self.test_dataset)}) dataset")

        def train_dataloader(self) -> DataLoader:
            assert self.train_dataset is not None
            if self.use_length_bucketing:
                from src.data.samplers import LengthBucketBatchSampler

                lengths = [len(seq) for seq in self.train_dataset.sequences]
                seed = self.cfg.project.get("seed", 42)
                batch_sampler = LengthBucketBatchSampler(
                    lengths=lengths,
                    batch_size=self.batch_size,
                    shuffle=True,
                    seed=seed,
                    drop_last=True,
                    jitter_fraction=self.length_bucket_jitter,
                )
                return DataLoader(
                    self.train_dataset,
                    batch_sampler=batch_sampler,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    collate_fn=dynamic_padding_collate,
                )
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=dynamic_padding_collate,
                drop_last=True,
            )

        def val_dataloader(self) -> DataLoader:
            assert self.val_dataset is not None
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=dynamic_padding_collate,
            )

        def test_dataloader(self) -> DataLoader:
            assert self.test_dataset is not None
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=dynamic_padding_collate,
            )

except ImportError:
    logger.debug("pytorch_lightning not available — ProteinDataModule disabled")
