# controls/inpainting_control.py

from __future__ import annotations

from typing import Any

from controls.control_utils import set_if_exists
from controls.enums import Inpainter
from core.config import AppConfig


def set_inpainter(config: AppConfig, name: Inpainter | str) -> None:
    if isinstance(name, str):
        try:
            name = Inpainter(name)
        except Exception as exc:
            raise ValueError(f"Unsupported inpainter: {name}") from exc

    config.image.inpainting.inpainter = name.value
    config.video.inpainting.inpainter = name.value

    for inpainting in [config.image.inpainting, config.video.inpainting]:
        inpainting.enable_opencv = name in (Inpainter.AUTO, Inpainter.OPENCV)
        inpainting.enable_lama = name in (Inpainter.AUTO, Inpainter.LAMA)
        inpainting.enable_sdxl = name in (Inpainter.AUTO, Inpainter.SDXL)
        inpainting.enable_flux = name in (Inpainter.AUTO, Inpainter.FLUX)


def disable_inpainting(config: AppConfig) -> None:
    for inpainting in [config.image.inpainting, config.video.inpainting]:
        inpainting.inpainter = "opencv"

        inpainting.enable_opencv = True
        inpainting.enable_lama = False
        inpainting.enable_sdxl = False
        inpainting.enable_flux = False

        for module_name in ["opencv", "lama", "sdxl", "flux"]:
            _set_nested_attr(inpainting, module_name, "enabled", False)


def set_device_for_inpainting(
    config: AppConfig,
    device_value: str,
) -> None:
    for inpainting in [config.image.inpainting, config.video.inpainting]:
        _set_nested_attr(inpainting, "lama", "device", device_value)
        _set_nested_attr(inpainting, "sdxl", "device", device_value)
        _set_nested_attr(inpainting, "flux", "device", device_value)


def _set_nested_attr(
    parent: Any,
    child_name: str,
    attr_name: str,
    value: Any,
) -> None:
    child = getattr(parent, child_name, None)

    if child is not None:
        set_if_exists(child, attr_name, value)