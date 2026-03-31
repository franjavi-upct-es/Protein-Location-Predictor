# tests/unit/test_logging.py
"""Tests for the logging system."""

from __future__ import annotations

import logging

from src.utils.logging import (
    ConsoleFormatter,
    FileFormatter,
    get_logger,
)


class TestGetLogger:
    """Tests for logger instantiation."""

    def test_returns_logger(self) -> None:
        logger = get_logger("test_module")
        assert isinstance(logger, logging.Logger)

    def test_logger_namespace(self) -> None:
        logger = get_logger("my_module")
        assert logger.name.startswith("src.")

    def test_already_namespaced(self) -> None:
        logger = get_logger("src.data.download")
        assert logger.name == "src.data.download"


class TestConsoleFormatter:
    """Tests for the console formatter."""

    def test_formats_message(self) -> None:
        fmt = ConsoleFormatter(use_color=False)
        record = logging.LogRecord(
            name="src.test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="hello world",
            args=(),
            exc_info=None,
        )
        output = fmt.format(record)
        assert "hello world" in output
        assert "INFO" in output


class TestFileFormatter:
    """Tests for the file formatter."""

    def test_includes_timestamp(self) -> None:
        fmt = FileFormatter()
        record = logging.LogRecord(
            name="src.test",
            level=logging.WARNING,
            pathname="",
            lineno=0,
            msg="disk full",
            args=(),
            exc_info=None,
        )
        output = fmt.format(record)
        assert "WARNING" in output
        assert "disk full" in output
        # Timestamp format: YYYY-MM-DD HH:MM:SS
        assert "-" in output.split("|")[0]
