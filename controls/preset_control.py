# controls/preset_control.py

from __future__ import annotations

from typing import Any

from controls.control_utils import combine_detector_stages
from controls.detection_control import set_detectors
from controls.inpainting_control import set_inpainter
from controls.tracking_control import set_tracker
from core.config import AppConfig
from utils.enums import Inpainter, Preset, Tracker


def apply_preset(config: AppConfig, preset: Preset | str) -> None:
    preset = normalize_preset(preset)

    if preset == Preset.DEFAULT:
        return

    presets = preset_configs()

    if preset not in presets:
        raise ValueError(f"Unsupported preset: {preset}")

    preset_config = presets[preset]

    set_inpainter(config, preset_config["inpainter"])
    set_tracker(config, preset_config["tracker"])

    set_detectors(
        config,
        combine_detector_stages(
            proposal_detectors=preset_config["proposal_detectors"],
            refiner_detectors=preset_config["refiner_detectors"],
        ),
    )


def normalize_preset(preset: Preset | str) -> Preset:
    if isinstance(preset, Preset):
        return preset

    try:
        return Preset(preset)
    except Exception as exc:
        raise ValueError(f"Unsupported preset: {preset}") from exc


def preset_configs() -> dict[Preset, dict[str, Any]]:
    return {
        Preset.FAST: {
            "inpainter": Inpainter.OPENCV,
            "tracker": Tracker.OPTICAL_FLOW,
            "proposal_detectors": ["opencv", "fft", "anomaly"],
            "refiner_detectors": [],
        },
        Preset.BALANCED: {
            "inpainter": Inpainter.AUTO,
            "tracker": Tracker.OPTICAL_FLOW,
            "proposal_detectors": ["opencv", "fft", "anomaly", "paddle_ocr"],
            "refiner_detectors": [],
        },
        Preset.QUALITY: {
            "inpainter": Inpainter.LAMA,
            "tracker": Tracker.XMEM,
            "proposal_detectors": [
                "opencv",
                "grounding_dino",
                "fft",
                "anomaly",
                "paddle_ocr",
            ],
            # You can change this to ["mobile_sam_2"] if that is your new refiner.
            "refiner_detectors": ["sam2"],
        },
    }
