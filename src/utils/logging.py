"""Logging helpers: one ``pipeline.log`` per run plus console output."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

LOGGER_NAME = "gem3d"
_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def get_logger(name: str | None = None) -> logging.Logger:
    """Return the project logger or a child of it (e.g. ``gem3d.colmap``)."""
    return logging.getLogger(f"{LOGGER_NAME}.{name}" if name else LOGGER_NAME)


def configure_run_logging(log_file: Path, level: int = logging.INFO) -> logging.Logger:
    """Send project log records to ``log_file`` and stderr.

    Existing handlers are replaced so repeated calls (e.g. Streamlit reruns)
    do not duplicate output.
    """
    logger = get_logger()
    logger.setLevel(level)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    log_file.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(_FORMAT)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(formatter)
    logger.addHandler(console)

    logger.propagate = False
    return logger
