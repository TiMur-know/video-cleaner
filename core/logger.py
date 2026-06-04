# core/logger.py

from __future__ import annotations

import logging
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from utils.enums import LogLevel


_LOGGERS: dict[str, logging.Logger] = {}
DEFAULT_LOGGER_NAME = "watermwark"
DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
__all__ = [
    "AppLogger",
    "configure_logger",
    "get_app_logger",
    "get_logger",
    "level_number",
    "log_detector_event",
    "log_event",
    "log_postprocessing_event",
    "normalize_level",
    "summarize_for_log",
]


class AppLogger:
    def __init__(
        self,
        name: str = DEFAULT_LOGGER_NAME,
        level: str | LogLevel = LogLevel.INFO,
        log_file: str | None = None,
    ) -> None:
        self.logger = configure_logger(
            name=name,
            level=level,
            log_file=log_file,
        )

    def event(
        self,
        component: str,
        event: str,
        payload: Mapping[str, Any] | None = None,
        level: str | LogLevel = LogLevel.INFO,
    ) -> None:
        log_event(
            component=component,
            event=event,
            payload=payload,
            level=level,
            logger=self.logger,
        )

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.logger.debug(message, *args, **kwargs)

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.logger.info(message, *args, **kwargs)

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.logger.warning(message, *args, **kwargs)

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.logger.error(message, *args, **kwargs)


def get_logger(
    name: str = "watermwark",
    level: str | LogLevel = LogLevel.INFO,
    log_file: str | None = None,
) -> logging.Logger:
    """
    Create or return a configured project logger.
    """

    return configure_logger(name=name, level=level, log_file=log_file)


def get_app_logger(
    name: str = DEFAULT_LOGGER_NAME,
    level: str | LogLevel = LogLevel.INFO,
    log_file: str | None = None,
) -> AppLogger:
    return AppLogger(name=name, level=level, log_file=log_file)


def configure_logger(
    name: str = DEFAULT_LOGGER_NAME,
    level: str | LogLevel = LogLevel.INFO,
    log_file: str | None = None,
) -> logging.Logger:
    level_name = normalize_level(level)
    cache_key = f"{name}:{level_name}:{log_file}"

    if cache_key in _LOGGERS:
        return _LOGGERS[cache_key]

    logger = logging.getLogger(name)
    logger.setLevel(level_name)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt=DEFAULT_LOG_FORMAT,
        datefmt=DEFAULT_DATE_FORMAT,
    )

    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level_name)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level_name)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _LOGGERS[cache_key] = logger
    return logger


def normalize_level(level: str | LogLevel) -> str:
    if isinstance(level, LogLevel):
        return level.value

    return str(level).upper()


def log_event(
    component: str,
    event: str,
    payload: Mapping[str, Any] | None = None,
    level: str | LogLevel = LogLevel.INFO,
    logger: logging.Logger | None = None,
) -> None:
    target_logger = logger or get_logger()
    target_logger.log(
        level_number(level),
        "%s %s: %s",
        component,
        event,
        summarize_for_log(payload or {}),
    )


def log_detector_event(
    detector_name: str,
    event: str,
    payload: Mapping[str, Any] | None = None,
) -> None:
    log_event(detector_name, event, payload)


def log_postprocessing_event(
    module_name: str,
    event: str,
    payload: Mapping[str, Any] | None = None,
) -> None:
    log_event(module_name, event, payload)


def summarize_for_log(value: Any, max_items: int = 8) -> Any:
    if value is None:
        return None

    if isinstance(value, np.ndarray):
        summary: dict[str, Any] = {
            "type": "ndarray",
            "shape": value.shape,
            "dtype": str(value.dtype),
            "size": int(value.size),
        }

        if value.size > 0 and np.issubdtype(value.dtype, np.number):
            summary["min"] = float(np.min(value))
            summary["max"] = float(np.max(value))

        return summary

    if isinstance(value, Path):
        return {
            "type": "path",
            "path": str(value),
            "exists": value.exists(),
            "is_file": value.is_file(),
            "is_dir": value.is_dir(),
        }

    if isinstance(value, Mapping):
        return {
            str(key): summarize_for_log(item, max_items=max_items)
            for key, item in list(value.items())[:max_items]
        }

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        value_list = list(value)

        return {
            "type": type(value).__name__,
            "length": len(value_list),
            "preview": [
                summarize_for_log(item, max_items=max_items)
                for item in value_list[:max_items]
            ],
        }

    return value


def level_number(level: str | LogLevel) -> int:
    level_name = normalize_level(level)
    value = logging.getLevelName(level_name)

    if isinstance(value, int):
        return value

    raise ValueError(f"Unsupported log level: {level}")
