"""Structured logging for Gweta.

This module provides consistent logging configuration
with support for both structured (JSON) and plain text formats.
"""

import logging
import sys
from typing import Any

from gweta.core.config import get_settings


class StructuredFormatter(logging.Formatter):
    """JSON-formatted log output for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        import json
        from datetime import datetime

        log_data: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add extra fields if present
        if hasattr(record, "extra_data"):
            log_data.update(record.extra_data)

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


class PlainFormatter(logging.Formatter):
    """Human-readable plain text log format."""

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


def setup_logging(level: str = "INFO", format: str = "plain") -> None:
    """Configure root logging for Gweta.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        format: Output format ("plain" or "structured")
    """
    root = logging.getLogger("gweta")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(root.level)

        if format == "structured":
            handler.setFormatter(StructuredFormatter())
        else:
            handler.setFormatter(PlainFormatter())

        root.addHandler(handler)
        root.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger for a module.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Only configure if not already configured
    if not logger.handlers:
        settings = get_settings()

        # Set level
        level = getattr(logging, settings.log_level.upper(), logging.INFO)
        logger.setLevel(level)

        # Create handler
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(level)

        # Set formatter based on settings
        if settings.log_format == "structured":
            handler.setFormatter(StructuredFormatter())
        else:
            handler.setFormatter(PlainFormatter())

        logger.addHandler(handler)

        # Don't propagate to root logger
        logger.propagate = False

    return logger


class LogContext:
    """Context manager for adding extra data to log messages.

    Example:
        >>> logger = get_logger(__name__)
        >>> with LogContext(logger, source="crawl", url="https://example.com"):
        ...     logger.info("Starting crawl")
    """

    def __init__(self, logger: logging.Logger, **extra: Any) -> None:
        """Initialize LogContext.

        Args:
            logger: The logger to add context to
            **extra: Extra fields to include in log messages
        """
        self.logger = logger
        self.extra = extra
        self._old_factory: Any = None

    def __enter__(self) -> "LogContext":
        """Enter context and set up extra data."""
        old_factory = logging.getLogRecordFactory()
        extra = self.extra

        def record_factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
            record = old_factory(*args, **kwargs)
            record.extra_data = extra  # type: ignore[attr-defined]
            return record

        self._old_factory = old_factory
        logging.setLogRecordFactory(record_factory)
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit context and restore original factory."""
        if self._old_factory:
            logging.setLogRecordFactory(self._old_factory)
