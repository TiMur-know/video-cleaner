# core/__init__.py

from __future__ import annotations

from typing import Any


_EXPORT_MODULES = {
    "AppConfig": "core.config",
    "ModuleConfig": "core.config",
    "PathConfig": "core.config",
    "RuntimeConfig": "core.config",
    "default_config": "core.config",
    "infer_mode_from_path": "core.config",
    "load_config": "core.config",
    "save_config": "core.config",
    "to_dict": "core.config",
    "AppLogger": "core.logger",
    "get_app_logger": "core.logger",
    "get_logger": "core.logger",
    "log_detector_event": "core.logger",
    "log_event": "core.logger",
    "log_postprocessing_event": "core.logger",
    "summarize_for_log": "core.logger",
    "Registry": "core.registry",
    "PREPROCESSORS": "core.registry",
    "DETECTORS": "core.registry",
    "TRACKERS": "core.registry",
    "INPAINTERS": "core.registry",
    "POSTPROCESSORS": "core.registry",
    "PIPELINES": "core.registry",
}

__all__ = list(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)

    if module_name is None:
        raise AttributeError(f"module 'core' has no attribute {name!r}")

    from importlib import import_module

    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value
