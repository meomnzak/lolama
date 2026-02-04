"""Logging utilities for LoLaMA."""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger.

    Child loggers inherit their level from the root 'lolama' logger,
    which defaults to INFO. Use set_verbosity() to change globally.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    # Ensure root lolama logger is configured once
    root = logging.getLogger("lolama")
    if not root.handlers:
        handler: logging.StreamHandler = logging.StreamHandler(sys.stderr)
        formatter: logging.Formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        root.addHandler(handler)
        root.setLevel(logging.INFO)

    return logging.getLogger(name)


def set_verbosity(level: int | str) -> None:
    """Set global verbosity level for all lolama loggers.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR) or int
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger("lolama")
    if not root.handlers:
        handler: logging.StreamHandler = logging.StreamHandler(sys.stderr)
        formatter: logging.Formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        root.addHandler(handler)
    root.setLevel(level)


# Convenience loggers for common modules
def get_model_logger() -> logging.Logger:
    """Get logger for model operations."""
    return get_logger("lolama.model")


def get_data_logger() -> logging.Logger:
    """Get logger for data loading operations."""
    return get_logger("lolama.data")


def get_generation_logger() -> logging.Logger:
    """Get logger for generation operations."""
    return get_logger("lolama.generation")
