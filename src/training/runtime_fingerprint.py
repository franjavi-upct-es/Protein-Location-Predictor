# src/training/runtime_fingerprint.py
"""
Runtime fingerprinting for reproducibility.

Captures everything needed to reproduce a training run two months
later: GPU + drivers, package versions, git commit SHA, hash of the
resolved configuration, and hashes of the dataset split files. The
fingerprint is logged to the active experiment tracker (MLflow if
present, otherwise the console) and persisted as a JSON artifact
alongside the model checkpoint.

This is the missing piece between "I have a checkpoint" and "I know
how that checkpoint was produced".

Usage::

    from src.training.runtime_fingerprint import RuntimeFingerprintCallback
    trainer = pl.Trainer(callbacks=[RuntimeFingerprintCallback(cfg)])
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytorch_lightning as pl

from src.utils.config import DotDict, resolve_path
from src.utils.logging import get_logger

logger = get_logger(__name__)


class RuntimeFingerprintCallback(pl.Callback):
    """
    Capture and persist a runtime fingerprint at training start.

    The fingerprint includes:
        - Timestamp (UTC, ISO-8601)
        - Host platform info (OS, machine, Python version)
        - GPU info (name, driver, CUDA version, compute capability)
        - Versions of relevant Python packages
        - Git commit SHA + dirty flag
        - SHA-256 of the resolved configuration
        - SHA-256 of each split CSV file (train/val/test)

    The fingerprint is written to ``reports/fingerprints/<run_id>.json``
    and logged to MLflow as a parameter dict (truncated keys) plus an
    artifact (full JSON).

    Args:
        cfg: Project configuration.
    """

    def __init__(self, cfg: DotDict) -> None:
        super().__init__()
        self.cfg = cfg
        self._fingerprint: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Lightning hook
    # ------------------------------------------------------------------

    def on_train_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        try:
            self._fingerprint = self._build_fingerprint()
            self._persist_to_disk(trainer)
            self._log_to_tracker(trainer)
            logger.info("Runtime fingerprint captured.")
        except Exception as e:  # never block training on a logging failure
            logger.warning(f"Failed to capture runtime fingerprint: {e}")

    # ------------------------------------------------------------------
    # Public accessors (useful for tests)
    # ------------------------------------------------------------------

    @property
    def fingerprint(self) -> dict[str, Any] | None:
        return self._fingerprint

    # ------------------------------------------------------------------
    # Builders
    # ------------------------------------------------------------------

    def _build_fingerprint(self) -> dict[str, Any]:
        return {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "platform": self._platform_info(),
            "gpu": self._gpu_info(),
            "packages": self._package_versions(),
            "git": self._git_info(),
            "config_sha256": self._config_hash(),
            "splits": self._split_hashes(),
            "env": {
                "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "CUBLAS_WORKSPACE_CONFIG": os.environ.get(
                    "CUBLAS_WORKSPACE_CONFIG"
                ),
            },
        }

    @staticmethod
    def _platform_info() -> dict[str, str]:
        return {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
            "processor": platform.processor(),
        }

    @staticmethod
    def _gpu_info() -> dict[str, Any]:
        info: dict[str, Any] = {"available": False}
        try:
            import torch

            if not torch.cuda.is_available():
                return info

            props = torch.cuda.get_device_properties(0)
            info.update(
                {
                    "available": True,
                    "device_name": props.name,
                    "compute_capability": (f"{props.major}.{props.minor}"),
                    "total_memory_gb": round(
                        props.total_memory / (1024**3), 2
                    ),
                    "torch_cuda_version": torch.version.cuda,
                    "torch_cudnn_version": (
                        torch.backends.cudnn.version()
                        if torch.backends.cudnn.is_available()
                        else None
                    ),
                }
            )

            # nvidia-smi for the driver version (cheap, optional)
            try:
                out = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=driver_version",
                        "--format=csv,noheader",
                    ],
                    stderr=subprocess.DEVNULL,
                    text=True,
                    timeout=2,
                )
                info["driver_version"] = out.strip().splitlines()[0]
            except (
                FileNotFoundError,
                subprocess.CalledProcessError,
                subprocess.TimeoutExpired,
            ):
                info["driver_version"] = None
        except ImportError:
            pass

        return info

    @staticmethod
    def _package_versions() -> dict[str, str]:
        from importlib.metadata import PackageNotFoundError, version

        packages = [
            "torch",
            "transformers",
            "peft",
            "accelerate",
            "bitsandbytes",
            "pytorch-lightning",
            "torchmetrics",
            "pandas",
            "numpy",
            "scikit-learn",
            "mlflow",
        ]
        result: dict[str, str] = {}
        for pkg in packages:
            try:
                result[pkg] = version(pkg)
            except PackageNotFoundError:
                result[pkg] = "not installed"
        return result

    @staticmethod
    def _git_info() -> dict[str, Any]:
        info: dict[str, Any] = {"available": False}
        try:
            sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=2,
            ).strip()
            dirty_output = subprocess.check_output(
                ["git", "status", "--porcelain"],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=2,
            )
            info.update(
                {
                    "available": True,
                    "commit_sha": sha,
                    "dirty": bool(dirty_output.strip()),
                }
            )
            try:
                branch = subprocess.check_output(
                    ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                    stderr=subprocess.DEVNULL,
                    text=True,
                    timeout=2,
                ).strip()
                info["branch"] = branch
            except subprocess.CalledProcessError:
                pass
        except (
            FileNotFoundError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
        ):
            pass
        return info

    def _config_hash(self) -> str:
        # Drop project_root because it varies per machine
        cfg_copy = dict(self.cfg)
        cfg_copy.pop("project_root", None)
        payload = json.dumps(cfg_copy, sort_keys=True, default=str).encode(
            "utf-8"
        )
        return hashlib.sha256(payload).hexdigest()

    def _split_hashes(self) -> dict[str, str | None]:
        result: dict[str, str | None] = {}
        try:
            splits_dir = resolve_path(self.cfg, "paths.splits_dir")
        except Exception:
            return result
        for name in ("train", "val", "test"):
            path = splits_dir / f"{name}.csv"
            if path.exists():
                result[name] = _hash_file(path)
            else:
                result[name] = None
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist_to_disk(self, trainer: pl.Trainer) -> None:
        if self._fingerprint is None:
            return
        try:
            reports_dir = resolve_path(self.cfg, "paths.reports_dir")
        except Exception:
            return

        out_dir = reports_dir / "fingerprints"
        out_dir.mkdir(parents=True, exist_ok=True)

        run_id = getattr(getattr(trainer, "logger", None), "run_id", None)
        if not run_id:
            run_id = self._fingerprint["timestamp_utc"].replace(":", "-")

        out_path = out_dir / f"{run_id}.json"
        out_path.write_text(
            json.dumps(self._fingerprint, indent=2, sort_keys=True)
        )
        logger.info(f"Runtime fingerprint written to {out_path}")

    def _log_to_tracker(self, trainer: pl.Trainer) -> None:
        if self._fingerprint is None:
            return
        pl_logger = getattr(trainer, "logger", None)
        if pl_logger is None:
            return

        # Log a small flat subset as hyperparameters / params
        flat = {
            "fp.git_sha": self._fingerprint.get("git", {}).get(
                "commit_sha", "unknown"
            )[:12],
            "fp.git_dirty": self._fingerprint.get("git", {}).get(
                "dirty", False
            ),
            "fp.gpu": self._fingerprint.get("gpu", {}).get(
                "device_name", "cpu"
            ),
            "fp.cuda": self._fingerprint.get("gpu", {}).get(
                "torch_cuda_version", "n/a"
            ),
            "fp.config_sha": self._fingerprint["config_sha256"][:12],
            "fp.torch": self._fingerprint["packages"].get("torch", "n/a"),
            "fp.transformers": self._fingerprint["packages"].get(
                "transformers", "n/a"
            ),
        }
        try:
            if hasattr(pl_logger, "log_hyperparams"):
                pl_logger.log_hyperparams(flat)
        except Exception as e:
            logger.debug(f"Could not log fingerprint params to logger: {e}")


def _hash_file(path: Path, chunk_size: int = 65536) -> str:
    """Return the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()
