# controls/preprocessing_control.py

from __future__ import annotations

from typing import Any

from controls.control_utils import set_if_exists, set_nested_attr
from core.config import AppConfig


PREPROCESSING_MODULES = [
    "resize",
    "denoise",
    "gamma",
    "clahe",
    "edge_enhance",
    "fft_enhance",
    "patchify",
]

MAIN_PATH_MODULES = [
    "resize",
    "denoise",
    "gamma",
    "clahe",
    "edge_enhance",
]


def apply_preprocessing_controls(
    config: AppConfig,
    enabled_modules: list[str] | None,
    tuning: dict[str, Any] | None = None,
) -> None:
    tuning = tuning or {}

    if enabled_modules is None:
        for preprocessing in [config.image.preprocessing, config.video.preprocessing]:
            apply_preprocessing_tuning(preprocessing, tuning)
        return

    selected = set(enabled_modules)

    for preprocessing in [config.image.preprocessing, config.video.preprocessing]:
        preprocessing.enable_main_path = any(
            module_name in selected for module_name in MAIN_PATH_MODULES
        )
        preprocessing.enable_fft_path = "fft_enhance" in selected
        preprocessing.enable_patch_path = "patchify" in selected

        for module_name in PREPROCESSING_MODULES:
            set_nested_attr(
                preprocessing,
                module_name,
                "enabled",
                module_name in selected,
            )

        apply_preprocessing_tuning(preprocessing, tuning)


def apply_preprocessing_tuning(
    preprocessing: Any,
    tuning: dict[str, Any],
) -> None:
    resize = preprocessing.resize
    set_if_exists(resize, "width", _positive_int_or_none(tuning.get("resize_width")))
    set_if_exists(resize, "height", _positive_int_or_none(tuning.get("resize_height")))
    set_if_exists(resize, "keep_aspect_ratio", bool(tuning.get("resize_keep_aspect", True)))
    set_if_exists(resize, "only_downscale", bool(tuning.get("resize_only_downscale", True)))

    denoise = preprocessing.denoise
    set_if_exists(denoise, "kernel_size", _odd_int(tuning.get("denoise_kernel_size"), 5))
    set_if_exists(denoise, "bilateral_sigma_color", float(tuning.get("denoise_strength", 50)))
    set_if_exists(denoise, "bilateral_sigma_space", float(tuning.get("denoise_strength", 50)))
    set_if_exists(denoise, "tv_weight", float(tuning.get("denoise_tv_weight", 0.08)))

    gamma = preprocessing.gamma
    set_if_exists(gamma, "gamma", float(tuning.get("gamma", 1.0)))
    set_if_exists(gamma, "gain", float(tuning.get("gain", 1.0)))

    clahe = preprocessing.clahe
    set_if_exists(clahe, "clip_limit", float(tuning.get("clahe_clip_limit", 2.0)))
    tile_size = int(tuning.get("clahe_tile_size", 8))
    set_if_exists(clahe, "tile_grid_size", (tile_size, tile_size))

    edge_enhance = preprocessing.edge_enhance
    set_if_exists(edge_enhance, "strength", float(tuning.get("edge_strength", 1.0)))
    set_if_exists(edge_enhance, "blur_kernel_size", _odd_int(tuning.get("edge_blur_kernel_size"), 5))

    fft_enhance = preprocessing.fft_enhance
    set_if_exists(fft_enhance, "strength", float(tuning.get("fft_strength", 0.7)))
    set_if_exists(fft_enhance, "high_pass_radius", int(tuning.get("fft_high_pass_radius", 20)))


def disable_preprocessing(config: AppConfig) -> None:
    for preprocessing in [config.image.preprocessing, config.video.preprocessing]:
        for module_name in PREPROCESSING_MODULES:
            _set_nested_attr(preprocessing, module_name, "enabled", False)

        preprocessing.enable_main_path = False
        preprocessing.enable_fft_path = False
        preprocessing.enable_patch_path = False


def set_preprocessing_logs(
    config: AppConfig,
    enabled: bool,
) -> None:
    for preprocessing in [config.image.preprocessing, config.video.preprocessing]:
        for module_name in PREPROCESSING_MODULES:
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


def _positive_int_or_none(value: Any) -> int | None:
    if value in (None, "", 0):
        return None

    value = int(value)

    if value <= 0:
        return None

    return value


def _odd_int(value: Any, default: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default

    number = max(1, number)

    if number % 2 == 0:
        number += 1

    return number
