# src/utils/logging.py
"""
Structured logging configuration.

Provides a consistent logging interface across all modules with:
  - Console output with color-coded levels
  - Optional file output with rotation
  - Structured context fields (e.g. epoch, batch, GPU memory)

Usage::

    from src.utils.logging import get_logger
    logger = get_logger(__name__)
    logger.info("Training started", extra={"epoch": 1, "lr": 2e-4})
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# ----------------------------------------------------------------------------
# Custom formatter with optional color support
# ----------------------------------------------------------------------------

_COLORS = {
    "DEBUG": "\033[36m",  # Cyan
    "INFO": "\033[32m",  # Green
    "WARNING": "\033[33m",  # Yellow
    "ERROR": "\033[31m",  # Red
    "CRITICAL": "\033[35m",  # Magenta
}
_RESET = "\033[0m"


class ConsoleFormatter(logging.Formatter):
    """Human-readable formatter with color for interactive terminals."""

    def __init__(self, use_color: bool = True) -> None:
        super().__init__()
        self.use_color = use_color and sys.stderr.isatty()

    def format(self, record: logging.LogRecord) -> str:
        level = record.levelname
        name = record.name.split(".")[-1]  # Short module name

        if self.use_color:
            color = _COLORS.get(level, "")
            prefix = f"{color}{level:<8}{_RESET} [{name}]"
        else:
            prefix = f"{level:<8} [{name}]"

        msg = f"{prefix} {record.getMessage()}"

        # Append structured extra fields if present
        _STANDARD_KEYS = {
            "name",
            "msg",
            "args",
            "created",
            "pathname",
            "filename",
            "module",
            "funcName",
            "levelno",
            "levelname",
            "lineno",
            "exc_info",
            "exc_text",
            "stack_info",
            "thread",
            "threadName",
            "process",
            "processName",
            "message",
            "relativeCreated",
            "msecs",
            "taskName",
        }
        extras = {
            k: v
            for k, v in record.__dict__.items()
            if k not in _STANDARD_KEYS and not k.startswith("_")
        }
        if extras:
            pairs = " ".join(f"{k}={v}" for k, v in extras.items())
            msg += f"  | {pairs}"

        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            msg += f"\n{record.exc_text}"

        return msg


class FileFormatter(logging.Formatter):
    """Machine-readable formatter for log files."""

    FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    def __init__(self) -> None:
        super().__init__(fmt=self.FORMAT, datefmt=self.DATE_FORMAT)


# ----------------------------------------------------------------------------
# Logger setup
# ----------------------------------------------------------------------------

_CONFIGURED = False


def setup_logging(level: str = "INFO", log_file: Path | str | None = None) -> None:
    """
    Configure the root logger for the project.

    Should be called once at application startup.

    Args:
        level: Logging level (DEBUG, INFO WARNING, ERROR, CRITICAL).
        log_file: Optional path to a log file. If provided, a file handler
                  is added alongside the console handler.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    root_logger = logging.getLogger("src")
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Clear any existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(ConsoleFormatter(use_color=True))
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(FileFormatter())
        root_logger.addHandler(file_handler)

    # Suppress noisy third-party loggers
    for noisy in ("transformers", "urllib3", "matplotlib", "mlflow"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance scoped to the given module name.

    The logger is a child of the 'src' root logger, so it inherits
    the handlers and level configured by ``setup_logging()``.

    Args:
        name: Typically ``__name__`` from the calling module.

    Returns:
        A logging.Logger instance.
    """
    if not _CONFIGURED:
        setup_logging()

    # Ensure the logger is under the 'src' namespace
    if not name.startswith("src"):
        name = f"src.{name}"
    return logging.getLogger(name)
