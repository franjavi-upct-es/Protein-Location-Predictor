# src/utils/reproducibility.py
"""
Reproducibility utilities.

Ensures determinisitc behaviour across runs by seeding all relevant
random number generators and configuring PyTorch backends.

Usage:
    from src.utils.reproducibility import seed_everything
    seed_everything(42)
"""

from __future__ import annotations

import os
import random
from contextlib import suppress

import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)


def seed_everything(seed: int = 42, deterministic: bool = True) -> None:
    """
    Seed all random number generators for reproducibility.

    Args:
        seed: The seed value to use everywhere.
        deterministic: If True, also configure PyTorch for deterministic
                       operations (may reduce performance slightly).
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # PyTorch 2.0+ determinisitc algorithms
            if hasattr(torch, "use_deterministic_algorithms"):
                with suppress(Exception):
                    torch.use_deterministic_algorithms(True, warn_only=True)

        logger.info(
            f"Random seed set to {seed} (determinisitc={deterministic})"
        )
    except ImportError:
        logger.warning(
            "PyTorch not installed — only Python/NumPy seeds were set."
        )
