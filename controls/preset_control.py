# controls/preset_control.py

from __future__ import annotations

from controls.control_utils import combine_detector_stages
from controls.detection_control import set_detectors
from controls.inpainting_control import set_inpainter
from controls.tracking_control import set_tracker
from core.config import AppConfig
from core.presets import (
    apply_app_parameter_overrides,
    get_preset_config,
    preset_configs as shared_preset_configs,
)
from utils.enums import Preset


def apply_preset(config: AppConfig, preset: Preset | str) -> None:
    preset = normalize_preset(preset)

    if preset == Preset.DEFAULT:
        return

    preset_config = get_preset_config(preset.value)

    set_inpainter(config, preset_config["inpainter"])
    set_tracker(config, preset_config["tracker"])

    set_detectors(
        config,
        combine_detector_stages(
            proposal_detectors=preset_config["proposal_detectors"],
            refiner_detectors=preset_config["refiner_detectors"],
        ),
    )

    apply_app_parameter_overrides(
        config,
        preset_config.get("parameters", {}),
    )


def normalize_preset(preset: Preset | str) -> Preset:
    if isinstance(preset, Preset):
        return preset

    try:
        return Preset(preset)
    except Exception as exc:
        raise ValueError(f"Unsupported preset: {preset}") from exc


def preset_configs() -> dict[Preset, dict[str, object]]:
    return {
        Preset(name): values
        for name, values in shared_preset_configs().items()
    }
