# ui/handlers/image_handlers.py

from __future__ import annotations

from typing import Any

import numpy as np

from controls.control import (
    apply_detection_tuning,
    apply_detector_tuning_dict,
    apply_preset,
    set_detectors,
    set_device,
    set_inpainter,
)
from controls.control_utils import combine_detector_stages
from core.config import load_config
from pipelines.image_pipeline import ImagePipeline
from ui.components.common import (
    get_file_path,
    to_preview_mask,
    to_preview_rgb,
)


def run_image_from_ui(
    config_path: str | None,
    image_file: Any,
    output_path: str,
    mask_output_path: str,
    preset: str,
    proposal_detectors: list[str] | None,
    refiner_detectors: list[str] | None,
    inpainter: str,
    device: str,
    detection_sensitivity: int | float = 50,
    mask_expand: int | float = 2,
    min_area: int | float = 25,
    fusion_strictness: int | float = 50,
    detector_tuning: dict[str, dict[str, Any]] | None = None,
) -> tuple[
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    dict[str, Any],
]:
    """
    Run image pipeline from Gradio UI.

    Detector behavior:
        no proposal + no refiner:
            detection disabled, output should stay unprocessed

        proposal detectors:
            find possible watermark regions

        refiner detectors:
            improve proposal masks, for example SAM2 / MobileSAM2

        2+ detectors:
            fusion is enabled by control.set_detectors()
    """

    if image_file is None:
        raise ValueError("Choose an image file first.")

    proposal_detectors = proposal_detectors or []
    refiner_detectors = refiner_detectors or []

    detectors = combine_detector_stages(
        proposal_detectors=proposal_detectors,
        refiner_detectors=refiner_detectors,
    )

    detector_mode = resolve_detector_mode(
        proposal_detectors=proposal_detectors,
        refiner_detectors=refiner_detectors,
    )

    image_path = get_file_path(image_file)

    config = load_config(config_path)

    config.runtime.mode = "image"
    config.image.input_path = image_path
    config.image.output_path = output_path
    config.image.mask_output_path = mask_output_path

    apply_preset(config, preset)  # type: ignore[arg-type]

    # UI detector choices override preset detector choices.
    set_detectors(config, detectors)

    apply_detection_tuning(
        config=config,
        sensitivity=detection_sensitivity,
        mask_expand=mask_expand,
        min_area=min_area,
        fusion_strictness=fusion_strictness,
    )

    apply_detector_tuning_dict(
        config=config,
        tuning=detector_tuning,
    )

    set_inpainter(config, inpainter)  # type: ignore[arg-type]
    set_device(config, device)  # type: ignore[arg-type]

    pipeline = ImagePipeline(config.image)

    result = pipeline.run(
        input_path=image_path,
        output_path=output_path,
        context={
            "ui": "visual",
            "mode": "image",
            "preset": preset,
            "proposal_detectors": proposal_detectors,
            "refiner_detectors": refiner_detectors,
            "detectors": detectors,
            "detector_mode": detector_mode,
            "detection_sensitivity": detection_sensitivity,
            "mask_expand": mask_expand,
            "min_area": min_area,
            "fusion_strictness": fusion_strictness,
            "detector_tuning": detector_tuning or {},
            "inpainter": inpainter,
            "device": device,
        },
    )

    metadata = {
        "settings": {
            "preset": preset,
            "proposal_detectors": proposal_detectors,
            "refiner_detectors": refiner_detectors,
            "detectors": detectors,
            "detector_mode": detector_mode,
            "proposal_fusion_enabled": len(proposal_detectors) > 1,
            "refiner_fusion_enabled": len(refiner_detectors) > 1,
            "fusion_enabled": len(detectors) > 1,
            "detection_sensitivity": detection_sensitivity,
            "mask_expand": mask_expand,
            "min_area": min_area,
            "fusion_strictness": fusion_strictness,
            "detector_tuning": detector_tuning or {},
            "inpainter": inpainter,
            "device": device,
        },
        "pipeline": result.metadata,
        "preprocessing": result.preprocessing_metadata,
        "detection": result.detection_metadata,
        "inpainting": result.inpainting_metadata,
        "postprocessing": result.postprocessing_metadata,
    }

    return (
        to_preview_rgb(result.original_image),
        to_preview_rgb(result.preprocessed_image),
        to_preview_mask(result.mask),
        to_preview_rgb(result.inpainted_image),
        to_preview_rgb(result.final_image),
        metadata,
    )


def resolve_detector_mode(
    proposal_detectors: list[str],
    refiner_detectors: list[str],
) -> str:
    proposal_count = len(proposal_detectors)
    refiner_count = len(refiner_detectors)
    total_count = proposal_count + refiner_count

    if total_count == 0:
        return "disabled"

    if proposal_count > 1 and refiner_count > 1:
        return "proposal_fused_refiner_fused"

    if proposal_count > 1 and refiner_count == 1:
        return "proposal_fused_refiner_single"

    if proposal_count == 1 and refiner_count > 1:
        return "proposal_single_refiner_fused"

    if proposal_count > 1:
        return "proposal_fused"

    if refiner_count > 1:
        return "refiner_fused"

    if proposal_count == 1 and refiner_count == 1:
        return "proposal_refiner"

    return "single"