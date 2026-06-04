# ui/handlers/video_handlers.py

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from controls.control import (
    apply_detection_tuning,
    apply_detector_tuning_dict,
    apply_preset,
    set_audio_backend,
    set_detectors,
    set_device,
    set_inpainter,
    set_tracker,
)
from controls.inpainting_control import apply_inpainter_tuning_dict
from controls.mask_processing_control import apply_mask_processing_controls
from controls.postprocessing_control import apply_postprocessing_controls
from controls.preprocessing_control import apply_preprocessing_controls
from controls.control_utils import combine_detector_stages
from core.config import load_config
from pipelines.video_pipeline import VideoPipeline
from ui.components.common import (
    get_file_path,
    to_preview_rgb,
)
from ui.handlers.frame_handlers import is_cancelled
from utils.video_io_utils import read_frame


def run_video_from_ui(
    config_path: str | None,
    video_file: Any,
    output_path: str,
    preset: str,
    proposal_detectors: list[str] | None,
    refiner_detectors: list[str] | None,
    inpainter: str,
    tracker: str,
    device: str,
    audio_backend: str,
    detection_sensitivity: int | float = 50,
    mask_expand: int | float = 2,
    min_area: int | float = 25,
    fusion_strictness: int | float = 50,
    detector_tuning: dict[str, dict[str, Any]] | None = None,
    mask_processing_enabled: bool | None = None,
    mask_processing_stages: list[str] | None = None,
    mask_processing_tuning: dict[str, Any] | None = None,
    inpainter_tuning: dict[str, Any] | None = None,
    preprocessing_modules: list[str] | None = None,
    postprocessing_modules: list[str] | None = None,
    processing_tuning: dict[str, Any] | None = None,
    cancel_token: dict[str, bool] | None = None,
) -> tuple[str | None, np.ndarray | None, dict[str, Any], str]:
    """
    Run video pipeline from Gradio UI.

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

    if video_file is None:
        raise ValueError("Choose a video file first.")

    if is_cancelled(cancel_token):
        return (
            None,
            None,
            {"cancelled": True},
            "Pipeline cancelled before start.",
        )

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

    video_path = get_file_path(video_file)

    config = load_config(config_path)

    config.runtime.mode = "video"
    config.video.input_path = video_path
    config.video.output_path = output_path
    config.video.rebuild.output_path = output_path
    config.video.rebuild.source_video_path = video_path

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

    apply_mask_processing_controls(
        config=config,
        enabled=mask_processing_enabled,
        stages=mask_processing_stages,
        tuning=mask_processing_tuning,
    )

    apply_preprocessing_controls(
        config=config,
        enabled_modules=preprocessing_modules,
        tuning=processing_tuning,
    )

    apply_postprocessing_controls(
        config=config,
        enabled_modules=postprocessing_modules,
        tuning=processing_tuning,
    )

    set_inpainter(config, inpainter)  # type: ignore[arg-type]
    set_device(config, device)  # type: ignore[arg-type]
    apply_inpainter_tuning_dict(
        config=config,
        tuning=inpainter_tuning,
    )
    set_tracker(config, tracker)  # type: ignore[arg-type]
    set_audio_backend(config, audio_backend)  # type: ignore[arg-type]

    pipeline = VideoPipeline(config.video)

    result = pipeline.run(
        input_path=video_path,
        output_path=output_path,
        context={
            "ui": "visual",
            "mode": "video",
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
            "mask_processing_enabled": mask_processing_enabled,
            "mask_processing_stages": mask_processing_stages or [],
            "mask_processing_tuning": mask_processing_tuning or {},
            "inpainter_tuning": inpainter_tuning or {},
            "preprocessing_modules": preprocessing_modules or [],
            "postprocessing_modules": postprocessing_modules or [],
            "processing_tuning": processing_tuning or {},
            "inpainter": inpainter,
            "tracker": tracker,
            "device": device,
            "audio_backend": audio_backend,
            "cancel_token": cancel_token,
        },
    )

    first_frame_preview = load_first_processed_frame(
        result.processed_frame_paths,
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
            "mask_processing_enabled": mask_processing_enabled,
            "mask_processing_stages": mask_processing_stages or [],
            "mask_processing_tuning": mask_processing_tuning or {},
            "inpainter_tuning": inpainter_tuning or {},
            "preprocessing_modules": preprocessing_modules or [],
            "postprocessing_modules": postprocessing_modules or [],
            "processing_tuning": processing_tuning or {},
            "inpainter": inpainter,
            "tracker": tracker,
            "device": device,
            "audio_backend": audio_backend,
        },
        "pipeline": result.metadata,
    }

    if is_cancelled(cancel_token) or getattr(result, "cancelled", False):
        return (
            result.output_path,
            first_frame_preview,
            {
                "cancelled": True,
                **metadata,
            },
            "Pipeline stopped.",
        )

    return (
        result.output_path,
        first_frame_preview,
        metadata,
        "Pipeline finished.",
    )


def load_first_processed_frame(
    processed_frame_paths: list[str],
) -> np.ndarray | None:
    if not processed_frame_paths:
        return None

    first_path = processed_frame_paths[0]

    if not first_path:
        return None

    if not Path(first_path).exists():
        return None

    frame = read_frame(first_path)

    return to_preview_rgb(frame)


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
