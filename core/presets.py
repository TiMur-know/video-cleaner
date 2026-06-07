# core/presets.py

from __future__ import annotations

from copy import deepcopy
from typing import Any


PresetConfig = dict[str, Any]


PRESET_CONFIGS: dict[str, PresetConfig] = {
    "fast": {
        "description": "Fast preview preset for checking basic detection and output flow.",
        "inpainter": "opencv",
        "tracker": "optical_flow",
        "proposal_detectors": ["opencv", "fft", "anomaly"],
        "refiner_detectors": [],
        "parameters": {},
    },
    "balanced": {
        "description": "General preset with OCR support and moderate runtime cost.",
        "inpainter": "auto",
        "tracker": "optical_flow",
        "proposal_detectors": ["opencv", "fft", "anomaly", "paddle_ocr"],
        "refiner_detectors": [],
        "parameters": {},
    },
    "quality": {
        "description": (
            "Quality preset for text, logos, and signs. It uses rough detectors, "
            "SAM-style refinement, safer mask expansion, LaMa, and blend cleanup."
        ),
        "inpainter": "lama",
        "tracker": "xmem",
        "proposal_detectors": [
            "opencv",
            "grounding_dino",
            "fft",
            "anomaly",
            "paddle_ocr",
        ],
        "refiner_detectors": ["mobile_sam_2", "sam2"],
        "parameters": {
            "detection.refiner_prompt_box_padding": 6,
            "mask_processing.enabled": True,
            "mask_processing.include_support_masks": True,
            "mask_processing.fallback_to_detection_mask": True,
            "mask_processing.candidate_max_area_ratio": 0.45,
            "mask_processing.box_filter.support_mode": "union",
            "mask_processing.box_filter.box_padding": 2,
            "mask_processing.box_filter.dilate_iterations": 2,
            "mask_processing.cleanup.fill_holes": True,
            "mask_processing.cleanup.dilate_iterations": 2,
            "mask_processing.cleanup.dilate_kernel_size": 7,
            "inpainting.lama.dilate_mask_iterations": 3,
            "inpainting.lama.mask_kernel_size": 7,
            "postprocessing.enabled": True,
            "postprocessing.enable_color_matching": True,
            "postprocessing.enable_seam_blending": True,
            "postprocessing.enable_artifact_removal": True,
            "postprocessing.color_matching.enabled": True,
            "postprocessing.seam_blending.enabled": True,
            "postprocessing.artifact_removal.enabled": True,
        },
    },
}


def preset_configs() -> dict[str, PresetConfig]:
    return deepcopy(PRESET_CONFIGS)


def get_preset_config(name: str) -> PresetConfig:
    try:
        return deepcopy(PRESET_CONFIGS[name])
    except KeyError as exc:
        raise ValueError(f"Unsupported preset: {name}") from exc


def apply_mode_parameter_overrides(
    mode_config: Any,
    parameters: dict[str, Any],
) -> None:
    for dotted_path, value in parameters.items():
        set_dotted_attr(mode_config, dotted_path, value)


def apply_app_parameter_overrides(
    config: Any,
    parameters: dict[str, Any],
) -> None:
    for mode_config in [config.image, config.video]:
        apply_mode_parameter_overrides(mode_config, parameters)


def set_dotted_attr(target: Any, dotted_path: str, value: Any) -> None:
    parts = dotted_path.split(".")

    if not parts:
        return

    current = target

    for part in parts[:-1]:
        current = getattr(current, part, None)

        if current is None:
            return

    if hasattr(current, parts[-1]):
        setattr(current, parts[-1], value)
