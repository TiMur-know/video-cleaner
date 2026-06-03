# controls/mask_processing_control.py

from __future__ import annotations

from typing import Any

from controls.control_utils import (
    clamp_float,
    set_if_exists,
    set_nested_attr,
)
from core.config import AppConfig


def apply_mask_processing_tuning(
    config: AppConfig,
    sensitivity: int | float = 50,
    mask_expand: int | float = 2,
    min_area: int | float = 25,
    fusion_strictness: int | float = 50,
) -> None:
    """
    Tune mask processing/refinement after detection and before inpainting.

    Uses the same global knobs as detection:
        sensitivity
        mask_expand
        min_area
        fusion_strictness
    """
    sensitivity = clamp_float(sensitivity, 0, 100)
    fusion_strictness = clamp_float(fusion_strictness, 0, 100)

    mask_expand = int(max(0, mask_expand))
    min_area = int(max(0, min_area))

    for mask_processing in mask_processing_configs(config):
        apply_box_filter_tuning(
            mask_processing=mask_processing,
            sensitivity=sensitivity,
            mask_expand=mask_expand,
            min_area=min_area,
        )

        apply_cleanup_tuning(
            mask_processing=mask_processing,
            mask_expand=mask_expand,
            min_area=min_area,
        )

        apply_fusion_tuning(
            mask_processing=mask_processing,
            fusion_strictness=fusion_strictness,
        )


def mask_processing_configs(config: AppConfig) -> list[Any]:
    """
    Return image/video mask_processing configs if they exist.

    Expected config paths:
        config.image.mask_processing
        config.video.mask_processing
    """
    configs: list[Any] = []

    image_mask_processing = getattr(config.image, "mask_processing", None)
    video_mask_processing = getattr(config.video, "mask_processing", None)

    if image_mask_processing is not None:
        configs.append(image_mask_processing)

    if video_mask_processing is not None:
        configs.append(video_mask_processing)

    return configs


def apply_box_filter_tuning(
    mask_processing: Any,
    sensitivity: float,
    mask_expand: int,
    min_area: int,
) -> None:
    """
    Tune the local box filter.

    Higher sensitivity:
        lower diff threshold
        more likely to keep faint/transparent watermark pixels

    Lower sensitivity:
        higher diff threshold
        less likely to keep background noise
    """
    box_filter = getattr(mask_processing, "box_filter", None)

    if box_filter is None:
        return

    diff_threshold = clamp_float(
        0.085 - sensitivity * 0.0006,
        0.015,
        0.12,
    )

    adaptive_c = clamp_float(
        -1.0 - sensitivity * 0.05,
        -8.0,
        1.0,
    )

    set_if_exists(box_filter, "diff_threshold", diff_threshold)
    set_if_exists(box_filter, "adaptive_c", adaptive_c)
    set_if_exists(box_filter, "box_padding", mask_expand)
    set_if_exists(box_filter, "min_component_area", min_area)
    set_if_exists(box_filter, "dilate_iterations", mask_expand)


def apply_cleanup_tuning(
    mask_processing: Any,
    mask_expand: int,
    min_area: int,
) -> None:
    """
    Tune final mask cleanup before inpainting.
    """
    cleanup = getattr(mask_processing, "cleanup", None)

    if cleanup is None:
        return

    set_if_exists(cleanup, "min_area", min_area)
    set_if_exists(cleanup, "dilate_iterations", mask_expand)
    set_if_exists(cleanup, "dilate_kernel_size", 5)
    set_if_exists(cleanup, "kernel_size", 3)


def apply_fusion_tuning(
    mask_processing: Any,
    fusion_strictness: float,
) -> None:
    """
    Tune mask fusion.

    Lower fusion_strictness:
        union-like behavior

    Higher fusion_strictness:
        requires more detector agreement
    """
    fusion = getattr(mask_processing, "fusion", None)

    if fusion is None:
        return

    min_votes = 1 if fusion_strictness < 50 else 2

    weighted_threshold = clamp_float(
        0.25 + fusion_strictness * 0.005,
        0.10,
        0.90,
    )

    set_if_exists(fusion, "min_votes", min_votes)
    set_if_exists(fusion, "weighted_threshold", weighted_threshold)


def enable_mask_processing(
    config: AppConfig,
    enabled: bool = True,
) -> None:
    for mask_processing in mask_processing_configs(config):
        set_if_exists(mask_processing, "enabled", enabled)


def disable_mask_processing(config: AppConfig) -> None:
    enable_mask_processing(config, enabled=False)


def set_mask_processing_mode(
    config: AppConfig,
    mode: str,
) -> None:
    """
    Suggested modes:
        off
        light
        balanced
        strict
    """
    for mask_processing in mask_processing_configs(config):
        match mode:
            case "off":
                set_if_exists(mask_processing, "enabled", False)

            case "light":
                set_if_exists(mask_processing, "enabled", True)
                set_if_exists(mask_processing, "use_box_filter", True)
                set_if_exists(mask_processing, "use_fusion", True)
                set_if_exists(mask_processing, "use_cleanup", True)
                set_if_exists(mask_processing, "include_detection_masks", False)
                set_if_exists(mask_processing, "include_support_masks", True)
                set_if_exists(mask_processing, "fallback_to_detection_mask", True)

                set_nested_attr(mask_processing, "box_filter", "support_mode", "none")
                set_nested_attr(mask_processing, "fusion", "method", "union")

            case "balanced":
                set_if_exists(mask_processing, "enabled", True)
                set_if_exists(mask_processing, "use_box_filter", True)
                set_if_exists(mask_processing, "use_fusion", True)
                set_if_exists(mask_processing, "use_cleanup", True)
                set_if_exists(mask_processing, "include_detection_masks", False)
                set_if_exists(mask_processing, "include_support_masks", True)
                set_if_exists(mask_processing, "fallback_to_detection_mask", True)

                set_nested_attr(mask_processing, "box_filter", "support_mode", "union")
                set_nested_attr(mask_processing, "fusion", "method", "union")

            case "strict":
                set_if_exists(mask_processing, "enabled", True)
                set_if_exists(mask_processing, "use_box_filter", True)
                set_if_exists(mask_processing, "use_fusion", True)
                set_if_exists(mask_processing, "use_cleanup", True)
                set_if_exists(mask_processing, "include_detection_masks", False)
                set_if_exists(mask_processing, "include_support_masks", True)
                set_if_exists(mask_processing, "fallback_to_detection_mask", False)

                set_nested_attr(mask_processing, "box_filter", "support_mode", "intersect")
                set_nested_attr(mask_processing, "fusion", "method", "vote")

            case _:
                raise ValueError(f"Unsupported mask processing mode: {mode}")


def set_mask_processing_logs(
    config: AppConfig,
    enabled: bool,
) -> None:
    for mask_processing in mask_processing_configs(config):
        set_if_exists(mask_processing, "log_events", enabled)

        for child_name in [
            "box_filter",
            "fusion",
            "cleanup",
        ]:
            set_nested_attr(
                mask_processing,
                child_name,
                "log_events",
                enabled,
            )