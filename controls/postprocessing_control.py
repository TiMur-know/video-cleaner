# controls/postprocessing_control.py

from __future__ import annotations

from typing import Any

from controls.control_utils import set_if_exists, set_nested_attr
from core.config import AppConfig


POSTPROCESSING_MODULES = [
    "color_matching",
    "seam_blending",
    "artifact_removal",
    "sharpening",
    "temporal_smoothing",
]


def apply_postprocessing_controls(
    config: AppConfig,
    enabled_modules: list[str] | None,
    tuning: dict[str, Any] | None = None,
) -> None:
    tuning = tuning or {}

    if enabled_modules is None:
        for postprocessing in [config.image.postprocessing, config.video.postprocessing]:
            apply_postprocessing_tuning(postprocessing, tuning)
        return

    selected = set(enabled_modules)

    for postprocessing in [config.image.postprocessing, config.video.postprocessing]:
        postprocessing.enabled = bool(selected)

        postprocessing.enable_color_matching = "color_matching" in selected
        postprocessing.enable_seam_blending = "seam_blending" in selected
        postprocessing.enable_artifact_removal = "artifact_removal" in selected
        postprocessing.enable_sharpening = "sharpening" in selected
        postprocessing.enable_temporal_smoothing = "temporal_smoothing" in selected

        for module_name in POSTPROCESSING_MODULES:
            set_nested_attr(
                postprocessing,
                module_name,
                "enabled",
                module_name in selected,
            )

        apply_postprocessing_tuning(postprocessing, tuning)


def apply_postprocessing_tuning(
    postprocessing: Any,
    tuning: dict[str, Any],
) -> None:
    color_matching = postprocessing.color_matching
    set_if_exists(
        color_matching,
        "max_shift",
        float(tuning.get("color_match_max_shift", 40)),
    )
    set_if_exists(
        color_matching,
        "max_scale",
        float(tuning.get("color_match_max_scale", 2.0)),
    )

    seam_blending = postprocessing.seam_blending
    set_if_exists(seam_blending, "strength", float(tuning.get("seam_strength", 1.0)))
    set_if_exists(
        seam_blending,
        "dilate_iterations",
        int(tuning.get("seam_dilate_iterations", 2)),
    )
    set_if_exists(
        seam_blending,
        "blur_kernel_size",
        _odd_int(tuning.get("seam_blur_kernel_size"), 21),
    )

    artifact_removal = postprocessing.artifact_removal
    set_if_exists(
        artifact_removal,
        "kernel_size",
        _odd_int(tuning.get("artifact_kernel_size"), 3),
    )
    set_if_exists(
        artifact_removal,
        "dilate_mask_iterations",
        int(tuning.get("artifact_dilate_iterations", 1)),
    )

    sharpening = postprocessing.sharpening
    set_if_exists(
        sharpening,
        "strength",
        float(tuning.get("sharpen_strength", 0.4)),
    )
    set_if_exists(
        sharpening,
        "blur_kernel_size",
        _odd_int(tuning.get("sharpen_blur_kernel_size"), 5),
    )

    temporal_smoothing = postprocessing.temporal_smoothing
    set_if_exists(
        temporal_smoothing,
        "alpha",
        float(tuning.get("temporal_alpha", 0.7)),
    )
    set_if_exists(
        temporal_smoothing,
        "window_size",
        _odd_int(tuning.get("temporal_window_size"), 3),
    )


def disable_postprocessing(config: AppConfig) -> None:
    for postprocessing in [config.image.postprocessing, config.video.postprocessing]:
        postprocessing.enabled = False
        postprocessing.enable_color_matching = False
        postprocessing.enable_seam_blending = False
        postprocessing.enable_artifact_removal = False
        postprocessing.enable_sharpening = False
        postprocessing.enable_temporal_smoothing = False

        for module_name in POSTPROCESSING_MODULES:
            _set_nested_attr(postprocessing, module_name, "enabled", False)


def set_postprocessing_logs(
    config: AppConfig,
    enabled: bool,
) -> None:
    for postprocessing in [config.image.postprocessing, config.video.postprocessing]:
        for module_name in POSTPROCESSING_MODULES:
            _set_nested_attr(postprocessing, module_name, "log_events", enabled)


def _set_nested_attr(
    parent: Any,
    child_name: str,
    attr_name: str,
    value: Any,
) -> None:
    child = getattr(parent, child_name, None)

    if child is not None:
        set_if_exists(child, attr_name, value)


def _odd_int(value: Any, default: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default

    number = max(1, number)

    if number % 2 == 0:
        number += 1

    return number
