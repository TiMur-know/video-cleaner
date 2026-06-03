# controls/preprocessing_control.py

from __future__ import annotations

from typing import Any

from controls.control_utils import set_if_exists
from core.config import AppConfig


def disable_preprocessing(config: AppConfig) -> None:
    for preprocessing in [config.image.preprocessing, config.video.preprocessing]:
        for module_name in [
            "resize",
            "denoise",
            "gamma",
            "clahe",
            "edge_enhance",
            "fft_enhance",
            "patchify",
        ]:
            _set_nested_attr(preprocessing, module_name, "enabled", False)

        preprocessing.enable_main_path = False
        preprocessing.enable_fft_path = False
        preprocessing.enable_patch_path = False


def set_preprocessing_logs(
    config: AppConfig,
    enabled: bool,
) -> None:
    for preprocessing in [config.image.preprocessing, config.video.preprocessing]:
        for module_name in [
            "resize",
            "denoise",
            "gamma",
            "clahe",
            "edge_enhance",
            "fft_enhance",
            "patchify",
        ]:
            _set_nested_attr(preprocessing, module_name, "log_events", enabled)


def _set_nested_attr(
    parent: Any,
    child_name: str,
    attr_name: str,
    value: Any,
) -> None:
    child = getattr(parent, child_name, None)

    if child is not None:
        set_if_exists(child, attr_name, value)