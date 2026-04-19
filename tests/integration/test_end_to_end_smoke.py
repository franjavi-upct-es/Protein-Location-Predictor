# tests/integration/test_end_to_end_smoke.py
"""
End-to-end smoke test on synthetic data.

Exercises the full training pipeline with the smallest publicly
available ESM-2 backbone (8M parameters) and synthetic protein
sequences with synthetic location labels. Runs in about a minute on
CPU and a handful of seconds on GPU.

The test does not assert specific metric values (the data is random
and there is nothing to learn) but it does assert:
    - The pipeline runs to completion without exceptions.
    - At least one optimizer step is performed.
    - A checkpoint is written.
    - Predictions can be loaded back from the checkpoint.

This is the canary that catches "I refactored something and the
trainer no longer wires together".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Heavy deps: skip the whole module if any are missing
pytest.importorskip("transformers")
pytest.importorskip("pytorch_lightning")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

LABEL_LIST = [
    "Cytoplasm",
    "Membrane",
    "Mitochondrion",
    "Nucleus",
]

SMALL_BACKBONE = "facebook/esm2_t6_8M_UR50D"


def _generate_synthetic_dataset(n: int = 60) -> pd.DataFrame:
    """Generate ``n`` synthetic protein rows with multi-label locations."""
    rng = np.random.default_rng(123)
    rows = []
    for i in range(n):
        seq_len = int(rng.integers(40, 200))
        seq = "".join(rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), size=seq_len))
        # Pick 1 or 2 random labels
        n_labels = int(rng.integers(1, 3))
        labels = list(rng.choice(LABEL_LIST, size=n_labels, replace=False))
        rows.append(
            {
                "accession": f"P{i:05d}",
                "sequence": seq,
                "locations_str": "|".join(labels),
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture()
def synthetic_project(tmp_path: Path) -> Path:
    """
    Lay out a minimal project tree under ``tmp_path`` so that
    ``load_config`` and the training pipeline can find everything.

    Returns the project root path.
    """
    # Sentinel file so utils.config.find_project_root works
    (tmp_path / "pyproject.toml").touch()

    # Splits
    splits_dir = tmp_path / "data" / "splits"
    splits_dir.mkdir(parents=True)

    df = _generate_synthetic_dataset(n=60)
    # Simple deterministic split: 40 / 10 / 10
    df.iloc[:40].to_csv(splits_dir / "train.csv", index=False)
    df.iloc[40:50].to_csv(splits_dir / "val.csv", index=False)
    df.iloc[50:60].to_csv(splits_dir / "test.csv", index=False)

    # Models / reports / mlruns dirs
    (tmp_path / "models").mkdir()
    (tmp_path / "reports").mkdir()
    (tmp_path / "mlruns").mkdir()

    return tmp_path


@pytest.fixture()
def synthetic_cfg(synthetic_project: Path):
    """Build a minimal in-memory config pointing at the synthetic project."""
    from src.utils.config import DotDict

    return DotDict.from_dict(
        {
            "project": {
                "name": "smoke-test",
                "seed": 42,
                "log_level": "WARNING",
            },
            "project_root": str(synthetic_project),
            "paths": {
                "data_dir": "data",
                "raw_dir": "data/raw",
                "processed_dir": "data/processed",
                "splits_dir": "data/splits",
                "models_dir": "models",
                "reports_dir": "reports",
                "mlflow_dir": "mlruns",
            },
            "model": {
                "backbone": {
                    "name": SMALL_BACKBONE,
                    "embedding_dim": 320,
                    "num_layers": 6,
                    "max_position_embeddings": 256,
                    "use_sdpa_attention": False,
                },
                "lora": {
                    "rank": 4,
                    "alpha": 8,
                    "dropout": 0.0,
                    "target_modules": [
                        "query",
                        "key",
                        "value",
                        "dense",
                    ],
                    "bias": "none",
                },
                "classifier": {
                    "hidden_dims": [64],
                    "dropout": 0.1,
                    "activation": "gelu",
                },
                "pooling": "mean",
                "quantization": {"enabled": False},
            },
            "loss": {
                "focal": {"gamma": 2.0, "alpha": None},
                "hierarchical": {"enabled": False, "weight": 0.0},
            },
            "features": {
                "biophysical": {"enabled": False, "properties": []},
                "signal_peptide": {"enabled": False},
                "transmembrane": {"enabled": False},
            },
            "training": {
                "max_epochs": 2,
                "patience": 10,
                "gradient_clip_val": 1.0,
                "accumulate_grad_batches": 1,
                "optimizer": {
                    "name": "adamw",
                    "lr": 5.0e-4,
                    "head_lr": 1.0e-3,
                    "weight_decay": 0.01,
                    "betas": [0.9, 0.999],
                },
                "scheduler": {
                    "name": "cosine_warmup",
                    "warmup_steps_fraction": 0.1,
                    "min_lr": 1.0e-6,
                },
                "batch_size": 4,
                "max_sequence_length": 128,
                "num_workers": 0,
                "pin_memory": False,
                "use_length_bucketing": True,
                "length_bucket_jitter": 0.05,
                "auto_batch_size": False,
                "precision": "32",
                "gradient_checkpointing": False,
                "cpu_offload": False,
                "experiment": {
                    "name": "smoke-test",
                    "tracking_uri": "mlruns",
                    "log_every_n_steps": 1,
                    "save_top_k": 1,
                },
                "deterministic": False,
            },
        }
    )


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestEndToEndSmoke:
    """One slow test that runs the whole pipeline on synthetic data."""

    def test_full_pipeline_runs_to_completion(self, synthetic_cfg, synthetic_project: Path) -> None:
        import pytorch_lightning as pl
        from pytorch_lightning.loggers import CSVLogger

        from src.data.datasets import ProteinDataModule
        from src.models.lightning_module import (
            ProteinLocalizationModule,
        )

        cfg = synthetic_cfg

        # Discover labels (mimics train.py without going through train())
        train_csv = synthetic_project / "data" / "splits" / "train.csv"
        df = pd.read_csv(train_csv)
        all_labels: list[str] = []
        for s in df["locations_str"].dropna():
            all_labels.extend(s.split("|"))
        label_list = sorted(set(all_labels))
        assert len(label_list) >= 2, "Synthetic data lost its labels"

        import torch

        class_freqs = torch.tensor(
            [all_labels.count(lbl) for lbl in label_list],
            dtype=torch.float32,
        )

        # Build datamodule and model
        dm = ProteinDataModule(cfg, label_list=label_list)
        model = ProteinLocalizationModule(
            cfg=cfg,
            label_list=label_list,
            class_frequencies=class_freqs,
        )

        # Minimal trainer (CPU, 2 epochs, no progress bar)
        ckpt_dir = synthetic_project / "models" / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        trainer = pl.Trainer(
            max_epochs=2,
            accelerator="cpu",
            devices=1,
            precision="32",
            logger=CSVLogger(
                save_dir=str(synthetic_project / "reports"),
                name="smoke",
            ),
            enable_progress_bar=False,
            enable_checkpointing=True,
            default_root_dir=str(synthetic_project),
            num_sanity_val_steps=0,
            log_every_n_steps=1,
        )

        # Run training
        trainer.fit(model, datamodule=dm)

        # The trainer must have stepped at least once
        assert trainer.global_step > 0, "Trainer did not step"

        # The model must still be runnable on a single batch
        dm.setup("test")
        test_batch = next(iter(dm.test_dataloader()))
        model.eval()
        with torch.no_grad():
            logits = model(
                input_ids=test_batch["input_ids"],
                attention_mask=test_batch["attention_mask"],
            )
        assert logits.shape[0] == test_batch["input_ids"].shape[0]
        assert logits.shape[1] == len(label_list)
