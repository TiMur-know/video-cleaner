# controls/postprocessing_control.py

from __future__ import annotations

from typing import Any

from controls.control_utils import set_if_exists
from core.config import AppConfig


def disable_postprocessing(config: AppConfig) -> None:
    for postprocessing in [config.image.postprocessing, config.video.postprocessing]:
        postprocessing.enable_color_matching = False
        postprocessing.enable_seam_blending = False
        postprocessing.enable_artifact_removal = False
        postprocessing.enable_sharpening = False
        postprocessing.enable_temporal_smoothing = False

        for module_name in [
            "color_matching",
            "seam_blending",
            "artifact_removal",
            "sharpening",
            "temporal_smoothing",
        ]:
            _set_nested_attr(postprocessing, module_name, "enabled", False)


def set_postprocessing_logs(
    config: AppConfig,
    enabled: bool,
) -> None:
    for postprocessing in [config.image.postprocessing, config.video.postprocessing]:
        for module_name in [
            "color_matching",
            "seam_blending",
            "artifact_removal",
            "sharpening",
            "temporal_smoothing",
        ]:
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