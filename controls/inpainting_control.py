# controls/inpainting_control.py

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from controls.control_utils import set_if_exists
from utils.enums import Inpainter

if TYPE_CHECKING:
    from core.config import AppConfig


INPAINTER_MODULES = ["opencv", "lama", "stable_diffusion", "sdxl", "flux"]


def apply_inpainter_tuning_dict(
    config: AppConfig,
    tuning: dict[str, Any] | None = None,
) -> None:
    if not tuning:
        return

    for inpainting in [config.image.inpainting, config.video.inpainting]:
        set_if_exists(
            inpainting,
            "enabled",
            bool(tuning.get("pipeline_enabled", inpainting.enabled)),
        )
        set_if_exists(
            inpainting,
            "fallback_enabled",
            bool(tuning.get("fallback_enabled", inpainting.fallback_enabled)),
        )

        for key, value in tuning.items():
            if "." not in key:
                continue

            module_name, field_name = key.split(".", 1)

            if module_name not in INPAINTER_MODULES:
                continue

            module_config = getattr(inpainting, module_name, None)

            if module_config is None:
                continue

            set_if_exists(
                module_config,
                field_name,
                normalize_inpainter_value(field_name, value),
            )


def normalize_inpainter_value(field_name: str, value: Any) -> Any:
    if field_name == "seed":
        if value is None:
            return None

        seed = int(value)
        return None if seed < 0 else seed

    if field_name in {
        "modulo",
        "dilate_mask_iterations",
        "mask_kernel_size",
        "num_inference_steps",
        "resize_to_multiple_of",
    }:
        return int(value)

    if field_name in {
        "radius",
        "guidance_scale",
        "strength",
    }:
        return float(value)

    return value


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
        inpainting.enable_stable_diffusion = name in (
            Inpainter.AUTO,
            Inpainter.STABLE_DIFFUSION,
        )
        inpainting.enable_sdxl = name in (Inpainter.AUTO, Inpainter.SDXL)
        inpainting.enable_flux = name in (Inpainter.AUTO, Inpainter.FLUX)


def disable_inpainting(config: AppConfig) -> None:
    for inpainting in [config.image.inpainting, config.video.inpainting]:
        inpainting.inpainter = "opencv"

        inpainting.enable_opencv = True
        inpainting.enable_lama = False
        inpainting.enable_stable_diffusion = False
        inpainting.enable_sdxl = False
        inpainting.enable_flux = False

        for module_name in ["opencv", "lama", "stable_diffusion", "sdxl", "flux"]:
            _set_nested_attr(inpainting, module_name, "enabled", False)


def set_device_for_inpainting(
    config: AppConfig,
    device_value: str,
) -> None:
    for inpainting in [config.image.inpainting, config.video.inpainting]:
        _set_nested_attr(inpainting, "lama", "device", device_value)
        _set_nested_attr(inpainting, "stable_diffusion", "device", device_value)
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
