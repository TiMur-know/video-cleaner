# control.py

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from dataclasses import dataclass

from controls.audio_control import set_audio_backend
from controls.control_utils import (
    DETECTOR_CHOICES,
    PROPOSAL_DETECTOR_CHOICES,
    REFINER_DETECTOR_CHOICES,
    ProposalDetectorStage,
    RefinerDetectorStage,
    combine_detector_stages,
    env_detectors,
    env_float,
    env_int,
    resolve_detector_mode_summary,
    split_detector_stages,
)
from controls.detection_control import (
    apply_detection_tuning,
    apply_detector_tuning_dict,
    disable_detection,
    enabled_detectors_from_detection,
    get_mode_detection,
    set_detectors,
    set_ocr,
)
from controls.device_control import set_device
from controls.enums import (
    AudioBackend,
    Device,
    Inpainter,
    OCR,
    Preset,
    Tracker,
)
from controls.inpainting_control import (
    disable_inpainting,
    set_inpainter,
)
from controls.io_control import (
    apply_input_output_overrides,
    resolve_run_mode,
)
from controls.mask_processing_control import (
    apply_mask_processing_tuning,
    disable_mask_processing,
)
from controls.postprocessing_control import disable_postprocessing
from controls.preprocessing_control import disable_preprocessing
from controls.preset_control import apply_preset
from controls.tracking_control import set_tracker
from core.config import AppConfig
from utils.env_loader import env_bool, env_choice


@dataclass(slots=True)
class BaseControlOptions:
    preset: Preset = Preset.DEFAULT
    inpainter: Inpainter | None = None
    tracker: Tracker | None = None
    ocr: OCR | None = None
    device: Device | None = None
    audio_backend: AudioBackend | None = None


@dataclass(slots=True)
class DetectorControlOptions(BaseControlOptions):
    proposal_detectors: list[str] | None = None
    refiner_detectors: list[str] | None = None

    def combined_detectors(self) -> list[str]:
        return combine_detector_stages(
            proposal_detectors=self.proposal_detectors,
            refiner_detectors=self.refiner_detectors,
        )

    def has_detector_override(self) -> bool:
        return (
            self.proposal_detectors is not None
            or self.refiner_detectors is not None
        )


@dataclass(slots=True)
class ControlOptions(DetectorControlOptions):
    detection_sensitivity: float = 50
    mask_expand: int = 2
    min_area: int = 25
    fusion_strictness: float = 50

    disable_preprocessing: bool = False
    disable_detection: bool = False
    disable_mask_processing: bool = False
    disable_tracking: bool = False
    disable_inpainting: bool = False
    disable_postprocessing: bool = False

    no_audio: bool = False
    save_intermediate: bool = False
    dry_run: bool = False


def add_control_args(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--preset",
        choices=[preset.value for preset in Preset],
        default=env_choice(
            "WATERMWARK_PRESET",
            [preset.value for preset in Preset],
            Preset.DEFAULT.value,
        ),
        help="Pipeline behavior preset.",
    )

    parser.add_argument(
        "--proposal-detectors",
        nargs="*",
        choices=list(PROPOSAL_DETECTOR_CHOICES),
        default=ProposalDetectorStage.from_env(),
        help="Proposal detectors find possible watermark regions.",
    )

    parser.add_argument(
        "--refiner-detectors",
        nargs="*",
        choices=list(REFINER_DETECTOR_CHOICES),
        default=RefinerDetectorStage.from_env(),
        help="Refiner detectors improve proposal masks.",
    )

    parser.add_argument(
        "--detectors",
        nargs="*",
        choices=list(DETECTOR_CHOICES),
        default=env_detectors(),
        help="Backwards-compatible combined detector list.",
    )

    parser.add_argument(
        "--inpainter",
        choices=[inpainter.value for inpainter in Inpainter],
        default=env_choice(
            "WATERMWARK_INPAINTER",
            [inpainter.value for inpainter in Inpainter],
            Inpainter.AUTO.value,
        ),
        help="Choose inpainter backend.",
    )

    parser.add_argument(
        "--tracker",
        choices=[tracker.value for tracker in Tracker],
        default=env_choice(
            "WATERMWARK_TRACKER",
            [tracker.value for tracker in Tracker],
            Tracker.OPTICAL_FLOW.value,
        ),
        help="Choose video tracker.",
    )

    parser.add_argument(
        "--ocr",
        choices=[ocr.value for ocr in OCR],
        default=env_choice(
            "WATERMWARK_OCR",
            [ocr.value for ocr in OCR],
            OCR.NONE.value,
        ),
        help="Backwards-compatible OCR option.",
    )

    parser.add_argument(
        "--device",
        choices=[device.value for device in Device],
        default=env_choice(
            "WATERMWARK_DEVICE",
            [device.value for device in Device],
            Device.AUTO.value,
        ),
        help="Preferred device for model backends.",
    )

    parser.add_argument(
        "--audio-backend",
        choices=[backend.value for backend in AudioBackend],
        default=env_choice(
            "WATERMWARK_AUDIO_BACKEND",
            [backend.value for backend in AudioBackend],
            AudioBackend.MOVIEPY.value,
        ),
        help="Video audio copy backend.",
    )

    parser.add_argument(
        "--detection-sensitivity",
        type=float,
        default=env_float("WATERMWARK_DETECTION_SENSITIVITY", 50),
        help="Detection sensitivity from 0 to 100.",
    )

    parser.add_argument(
        "--mask-expand",
        type=int,
        default=env_int("WATERMWARK_MASK_EXPAND", 2),
        help="Grow detected mask by this amount.",
    )

    parser.add_argument(
        "--min-area",
        type=int,
        default=env_int("WATERMWARK_MIN_AREA", 25),
        help="Remove detector components smaller than this area.",
    )

    parser.add_argument(
        "--fusion-strictness",
        type=float,
        default=env_float("WATERMWARK_FUSION_STRICTNESS", 50),
        help="Fusion strictness from 0 to 100.",
    )

    parser.add_argument(
        "--no-preprocessing",
        action="store_true",
        default=env_bool("WATERMWARK_NO_PREPROCESSING", False),
        help="Disable preprocessing.",
    )

    parser.add_argument(
        "--no-detection",
        action="store_true",
        default=env_bool("WATERMWARK_NO_DETECTION", False),
        help="Disable detection.",
    )

    parser.add_argument(
        "--no-mask-processing",
        action="store_true",
        default=env_bool("WATERMWARK_NO_MASK_PROCESSING", False),
        help="Disable mask processing between detection and inpainting.",
    )

    parser.add_argument(
        "--no-tracking",
        action="store_true",
        default=env_bool("WATERMWARK_NO_TRACKING", False),
        help="Disable video tracking.",
    )

    parser.add_argument(
        "--no-inpainting",
        action="store_true",
        default=env_bool("WATERMWARK_NO_INPAINTING", False),
        help="Disable inpainting.",
    )

    parser.add_argument(
        "--no-postprocessing",
        action="store_true",
        default=env_bool("WATERMWARK_NO_POSTPROCESSING", False),
        help="Disable postprocessing.",
    )

    parser.add_argument(
        "--no-audio",
        action="store_true",
        default=env_bool("WATERMWARK_NO_AUDIO", False),
        help="Do not copy original audio.",
    )

    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        default=env_bool("WATERMWARK_SAVE_INTERMEDIATE", False),
        help="Save intermediate masks/frames.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=env_bool("WATERMWARK_DRY_RUN", False),
        help="Print resolved config but do not run pipeline.",
    )


def options_from_args(args: Namespace) -> ControlOptions:
    proposal_detectors = args.proposal_detectors
    refiner_detectors = args.refiner_detectors
    combined_detectors = getattr(args, "detectors", None)

    if (
        combined_detectors is not None
        and proposal_detectors is None
        and refiner_detectors is None
    ):
        proposal_detectors, refiner_detectors = split_detector_stages(
            combined_detectors
        )

    return ControlOptions(
        preset=Preset(args.preset),
        proposal_detectors=proposal_detectors,
        refiner_detectors=refiner_detectors,
        inpainter=Inpainter(args.inpainter) if args.inpainter is not None else None,
        tracker=Tracker(args.tracker) if args.tracker is not None else None,
        ocr=OCR(args.ocr) if args.ocr is not None else None,
        device=Device(args.device) if args.device is not None else None,
        audio_backend=AudioBackend(args.audio_backend)
        if args.audio_backend is not None
        else None,
        detection_sensitivity=args.detection_sensitivity,
        mask_expand=args.mask_expand,
        min_area=args.min_area,
        fusion_strictness=args.fusion_strictness,
        disable_preprocessing=args.no_preprocessing,
        disable_detection=args.no_detection,
        disable_mask_processing=args.no_mask_processing,
        disable_tracking=args.no_tracking,
        disable_inpainting=args.no_inpainting,
        disable_postprocessing=args.no_postprocessing,
        no_audio=args.no_audio,
        save_intermediate=args.save_intermediate,
        dry_run=args.dry_run,
    )


def apply_controls(config: AppConfig, args: Namespace) -> ControlOptions:
    options = options_from_args(args)

    apply_input_output_overrides(config, args)
    apply_preset(config, options.preset)

    if options.has_detector_override():
        set_detectors(config, options.combined_detectors())

    elif options.ocr is not None and options.ocr != OCR.NONE:
        set_ocr(config, options.ocr)

    apply_detection_tuning(
        config=config,
        sensitivity=options.detection_sensitivity,
        mask_expand=options.mask_expand,
        min_area=options.min_area,
        fusion_strictness=options.fusion_strictness,
    )

    apply_mask_processing_tuning(
        config=config,
        sensitivity=options.detection_sensitivity,
        mask_expand=options.mask_expand,
        min_area=options.min_area,
        fusion_strictness=options.fusion_strictness,
    )

    if options.inpainter is not None:
        set_inpainter(config, options.inpainter)

    if options.tracker is not None:
        set_tracker(config, options.tracker)

    if options.device is not None:
        set_device(config, options.device)

    if options.audio_backend is not None:
        set_audio_backend(config, options.audio_backend)

    if options.disable_preprocessing:
        disable_preprocessing(config)

    if options.disable_detection:
        disable_detection(config)

    if options.disable_mask_processing:
        disable_mask_processing(config)

    if options.disable_tracking:
        set_tracker(config, Tracker.NONE)

    if options.disable_inpainting:
        disable_inpainting(config)

    if options.disable_postprocessing:
        disable_postprocessing(config)

    if options.no_audio:
        set_audio_backend(config, AudioBackend.NONE)

    if options.save_intermediate:
        config.image.save_mask = True
        config.video.save_processed_frames = True

    return options


def control_summary(
    config: AppConfig,
    mode: str,
    options: ControlOptions,
) -> dict[str, object]:
    selected_detectors = enabled_detectors_summary(config, mode)
    proposal_detectors = enabled_proposal_detectors_summary(config, mode)
    refiner_detectors = enabled_refiner_detectors_summary(config, mode)

    return {
        "mode": mode,
        "preset": options.preset,
        "detectors": selected_detectors,
        "proposal_detectors": proposal_detectors,
        "refiner_detectors": refiner_detectors,
        "detector_mode": resolve_detector_mode_summary(
            proposal_detectors=proposal_detectors,
            refiner_detectors=refiner_detectors,
        ),
        "proposal_fusion_enabled": len(proposal_detectors) > 1,
        "refiner_fusion_enabled": len(refiner_detectors) > 1,
        "fusion_enabled": get_mode_detection(config, mode).enable_fusion,
        "detection_sensitivity": options.detection_sensitivity,
        "mask_expand": options.mask_expand,
        "min_area": options.min_area,
        "fusion_strictness": options.fusion_strictness,
        "dry_run": options.dry_run,
        "image_input": config.image.input_path,
        "image_output": config.image.output_path,
        "video_input": config.video.input_path,
        "video_output": config.video.output_path,
        "image_inpainter": config.image.inpainting.inpainter,
        "video_inpainter": config.video.inpainting.inpainter,
        "video_tracker": config.video.tracking.mode,
        "video_tracking_enabled": config.video.tracking.enabled,
        "copy_audio": config.video.rebuild.copy_audio,
        "audio_backend": config.video.rebuild.audio_backend,
        "mask_processing_disabled": options.disable_mask_processing,
    }


def enabled_detectors_summary(config: AppConfig, mode: str) -> list[str]:
    return enabled_detectors_from_detection(
        get_mode_detection(config, mode)
    )


def enabled_proposal_detectors_summary(
    config: AppConfig,
    mode: str,
) -> list[str]:
    return [
        detector_name
        for detector_name in enabled_detectors_summary(config, mode)
        if detector_name in PROPOSAL_DETECTOR_CHOICES
    ]


def enabled_refiner_detectors_summary(
    config: AppConfig,
    mode: str,
) -> list[str]:
    return [
        detector_name
        for detector_name in enabled_detectors_summary(config, mode)
        if detector_name in REFINER_DETECTOR_CHOICES
    ]


__all__ = [
    "add_control_args",
    "options_from_args",
    "apply_controls",
    "apply_input_output_overrides",
    "resolve_run_mode",
    "control_summary",
    "enabled_detectors_summary",
    "enabled_proposal_detectors_summary",
    "enabled_refiner_detectors_summary",
    "ControlOptions",
    "BaseControlOptions",
    "DetectorControlOptions",
]