# core/logger.py

from __future__ import annotations

import logging
import sys
from pathlib import Path

from core.enums import LogLevel


_LOGGERS: dict[str, logging.Logger] = {}


def get_logger(
    name: str = "watermwark",
    level: str | LogLevel = LogLevel.INFO,
    log_file: str | None = None,
) -> logging.Logger:
    """
    Create or return a configured project logger.
    """

    if isinstance(level, LogLevel):
        level = level.value

    cache_key = f"{name}:{level}:{log_file}"

    if cache_key in _LOGGERS:
        return _LOGGERS[cache_key]

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _LOGGERS[cache_key] = logger
    return logger