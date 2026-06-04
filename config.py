# control.py

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


DEFAULT_CONFIG_PATH = Path("config.json")

from controls.control_utils import (
    DETECTOR_CHOICES,
    PROPOSAL_DETECTOR_CHOICES,
    REFINER_DETECTOR_CHOICES,
    ProposalDetectorStage,
    RefinerDetectorStage,
    clamp_float,
    combine_detector_stages,
    env_detectors,
    env_float,
    env_int,
    normalize_detectors,
    resolve_detector_mode_summary,
    set_if_exists,
    split_detector_stages,
)
from core.config import AppConfig, infer_mode_from_path
from utils.env_loader import env_bool, env_choice


PresetName = Literal["default", "fast", "balanced", "quality"]
InpainterName = Literal["auto", "opencv", "lama", "stable_diffusion", "sdxl", "flux"]
TrackerName = Literal["none", "optical_flow", "kalman", "xmem", "cotracker"]
OCRName = Literal["none", "paddle", "easy", "both"]
DeviceName = Literal["auto", "cpu", "cuda", "mps"]
AudioBackendName = Literal["none", "moviepy", "ffmpeg"]


DETECTOR_CONFIG_FIELDS: dict[str, str] = {
    "opencv": "opencv",
    "grounding_dino": "grounding_dino",
    "yolo": "yolo",
    "fft": "fft",
    "anomaly": "anomaly",
    "paddle_ocr": "paddle_ocr",
    "easy_ocr": "easy_ocr",
    "sam2": "sam2",
    "mobile_sam_2": "mobile_sam_2",
}

DETECTOR_ENABLE_FLAGS: dict[str, str] = {
    detector_name: f"enable_{detector_name}"
    for detector_name in DETECTOR_CONFIG_FIELDS
}


@dataclass(slots=True)
class BaseControlOptions:
    preset: PresetName = "default"
    inpainter: InpainterName | None = None
    tracker: TrackerName | None = None
    ocr: OCRName | None = None
    device: DeviceName | None = None
    audio_backend: AudioBackendName | None = None


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
        return self.proposal_detectors is not None or self.refiner_detectors is not None


@dataclass(slots=True)
class ControlOptions(DetectorControlOptions):
    detection_sensitivity: float = 50
    mask_expand: int = 2
    min_area: int = 25
    fusion_strictness: float = 50

    disable_preprocessing: bool = False
    disable_detection: bool = False
    disable_tracking: bool = False
    disable_inpainting: bool = False
    disable_postprocessing: bool = False

    no_audio: bool = False
    save_intermediate: bool = False
    dry_run: bool = False


def add_control_args(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--preset",
        choices=["default", "fast", "balanced", "quality"],
        default=env_choice(
            "WATERMWARK_PRESET",
            ["default", "fast", "balanced", "quality"],
            "default",
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
        choices=["auto", "opencv", "lama", "stable_diffusion", "sdxl", "flux"],
        default=env_choice(
            "WATERMWARK_INPAINTER",
            ["auto", "opencv", "lama", "stable_diffusion", "sdxl", "flux"],
            "auto",
        ),
        help="Choose inpainter backend.",
    )

    parser.add_argument(
        "--tracker",
        choices=["none", "optical_flow", "kalman", "xmem", "cotracker"],
        default=env_choice(
            "WATERMWARK_TRACKER",
            ["none", "optical_flow", "kalman", "xmem", "cotracker"],
            "optical_flow",
        ),
        help="Choose video tracker.",
    )

    parser.add_argument(
        "--ocr",
        choices=["none", "paddle", "easy", "both"],
        default=env_choice(
            "WATERMWARK_OCR",
            ["none", "paddle", "easy", "both"],
            "none",
        ),
        help="Backwards-compatible OCR option.",
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default=env_choice(
            "WATERMWARK_DEVICE",
            ["auto", "cpu", "cuda", "mps"],
            "auto",
        ),
        help="Preferred device for model backends.",
    )

    parser.add_argument(
        "--audio-backend",
        choices=["none", "moviepy", "ffmpeg"],
        default=env_choice(
            "WATERMWARK_AUDIO_BACKEND",
            ["none", "moviepy", "ffmpeg"],
            "moviepy",
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
        preset=args.preset,
        proposal_detectors=proposal_detectors,
        refiner_detectors=refiner_detectors,
        inpainter=args.inpainter,
        tracker=args.tracker,
        ocr=args.ocr,
        device=args.device,
        audio_backend=args.audio_backend,
        detection_sensitivity=args.detection_sensitivity,
        mask_expand=args.mask_expand,
        min_area=args.min_area,
        fusion_strictness=args.fusion_strictness,
        disable_preprocessing=args.no_preprocessing,
        disable_detection=args.no_detection,
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

    elif options.ocr is not None and options.ocr != "none":
        set_ocr(config, options.ocr)

    apply_detection_tuning(
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

    if options.disable_tracking:
        set_tracker(config, "none")

    if options.disable_inpainting:
        disable_inpainting(config)

    if options.disable_postprocessing:
        disable_postprocessing(config)

    if options.no_audio:
        set_audio_backend(config, "none")

    if options.save_intermediate:
        config.image.save_mask = True
        config.video.save_processed_frames = True

    return options


def apply_input_output_overrides(config: AppConfig, args: Namespace) -> None:
    if args.type is not None:
        config.runtime.mode = args.type

    if args.input is not None:
        mode = args.type or config.runtime.mode

        if mode == "auto":
            mode = infer_mode_from_path(args.input)

        if mode == "image":
            config.image.input_path = args.input

        elif mode == "video":
            config.video.input_path = args.input

    if args.output is not None:
        mode = args.type or config.runtime.mode

        if mode == "auto" and args.input is not None:
            mode = infer_mode_from_path(args.input)

        if mode == "image":
            config.image.output_path = args.output

        elif mode == "video":
            config.video.output_path = args.output


def resolve_run_mode(config: AppConfig, args: Namespace) -> str:
    mode = args.type or config.runtime.mode

    if mode != "auto":
        return mode

    if args.input is not None:
        return infer_mode_from_path(args.input)

    return infer_mode_from_path(config.image.input_path)


def apply_preset(config: AppConfig, preset: PresetName) -> None:
    presets: dict[str, dict[str, Any]] = {
        "fast": {
            "inpainter": "opencv",
            "tracker": "optical_flow",
            "proposal_detectors": ["opencv", "fft", "anomaly"],
            "refiner_detectors": [],
        },
        "balanced": {
            "inpainter": "auto",
            "tracker": "optical_flow",
            "proposal_detectors": ["opencv", "fft", "anomaly", "paddle_ocr"],
            "refiner_detectors": [],
        },
        "quality": {
            "inpainter": "lama",
            "tracker": "xmem",
            "proposal_detectors": [
                "opencv",
                "grounding_dino",
                "fft",
                "anomaly",
                "paddle_ocr",
            ],
            "refiner_detectors": ["sam2"],
        },
    }

    if preset == "default":
        return

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


def set_detectors(
    config: AppConfig,
    detectors: list[str] | tuple[str, ...] | None,
) -> None:
    selected = normalize_detectors(detectors)

    for detection in detection_configs(config):
        disable_detection_config_only(detection)

        for detector_name in selected:
            set_detector_enabled(detection, detector_name, True)

        fusion_enabled = len(selected) > 1
        set_if_exists(detection, "enable_fusion", fusion_enabled)
        set_detector_config_enabled(detection, "fusion", fusion_enabled)


def disable_detection_config_only(detection: Any) -> None:
    for detector_name in DETECTOR_CONFIG_FIELDS:
        set_detector_enabled(detection, detector_name, False)

    set_if_exists(detection, "enable_fusion", False)
    set_detector_config_enabled(detection, "fusion", False)


def set_detector_enabled(
    detection: Any,
    detector_name: str,
    enabled: bool,
) -> None:
    flag_name = DETECTOR_ENABLE_FLAGS.get(detector_name)

    if flag_name is not None:
        set_if_exists(detection, flag_name, enabled)

    set_detector_config_enabled(detection, detector_name, enabled)


def set_detector_config_enabled(
    detection: Any,
    detector_name: str,
    enabled: bool,
) -> None:
    config_name = DETECTOR_CONFIG_FIELDS.get(detector_name, detector_name)
    detector_config = getattr(detection, config_name, None)

    if detector_config is not None:
        set_if_exists(detector_config, "enabled", enabled)


def apply_detection_tuning(
    config: AppConfig,
    sensitivity: int | float = 50,
    mask_expand: int | float = 2,
    min_area: int | float = 25,
    fusion_strictness: int | float = 50,
) -> None:
    sensitivity = clamp_float(sensitivity, 0, 100)
    fusion_strictness = clamp_float(fusion_strictness, 0, 100)

    mask_expand = int(max(0, mask_expand))
    min_area = int(max(0, min_area))

    for detection in detection_configs(config):
        apply_opencv_tuning(detection, sensitivity, mask_expand, min_area)
        apply_sam2_tuning(detection, sensitivity, mask_expand, min_area)
        apply_mobile_sam_2_tuning(detection, sensitivity, mask_expand, min_area)
        apply_fft_tuning(detection, sensitivity, mask_expand, min_area)
        apply_anomaly_tuning(detection, sensitivity, mask_expand, min_area)
        apply_ocr_tuning(detection, mask_expand, min_area)
        apply_fusion_tuning(detection, fusion_strictness)


def apply_detector_tuning_dict(
    config: AppConfig,
    tuning: dict[str, dict[str, Any]] | None,
) -> None:
    if not tuning:
        return

    for detection in detection_configs(config):
        for detector_name, values in tuning.items():
            detector_config = getattr(detection, detector_name, None)

            if detector_config is None:
                continue

            for field_name, value in values.items():
                set_if_exists(detector_config, field_name, value)


def apply_opencv_tuning(
    detection: Any,
    sensitivity: float,
    mask_expand: int,
    min_area: int,
) -> None:
    opencv = getattr(detection, "opencv", None)

    if opencv is None:
        return

    canny_low = int(80 - sensitivity * 0.6)
    canny_high = int(180 - sensitivity * 0.8)
    bright_percentile = 98.0 - sensitivity * 0.10
    dark_percentile = 2.0 + sensitivity * 0.10

    set_if_exists(opencv, "canny_low", max(5, canny_low))
    set_if_exists(opencv, "canny_high", max(20, canny_high))
    set_if_exists(opencv, "bright_percentile", clamp_float(bright_percentile, 70, 99))
    set_if_exists(opencv, "dark_percentile", clamp_float(dark_percentile, 1, 30))
    set_if_exists(opencv, "dilate_iterations", mask_expand)
    set_if_exists(opencv, "min_area", min_area)


def apply_sam2_tuning(
    detection: Any,
    sensitivity: float,
    mask_expand: int,
    min_area: int,
) -> None:
    sam2 = getattr(detection, "sam2", None)

    if sam2 is None:
        return

    set_if_exists(sam2, "prompt_sensitivity", sensitivity)
    set_if_exists(sam2, "dilate_iterations", mask_expand)
    set_if_exists(sam2, "min_area", min_area)
    set_if_exists(sam2, "prompt_min_area", min_area)


def apply_mobile_sam_2_tuning(
    detection: Any,
    sensitivity: float,
    mask_expand: int,
    min_area: int,
) -> None:
    mobile_sam_2 = getattr(detection, "mobile_sam_2", None)

    if mobile_sam_2 is None:
        return

    threshold = clamp_float(0.7 - sensitivity * 0.004, 0.05, 0.95)

    set_if_exists(mobile_sam_2, "mask_threshold", threshold)
    set_if_exists(mobile_sam_2, "min_area", min_area)
    set_if_exists(mobile_sam_2, "prompt_min_area", min_area)
    set_if_exists(mobile_sam_2, "dilate_iterations", mask_expand)


def apply_fft_tuning(
    detection: Any,
    sensitivity: float,
    mask_expand: int,
    min_area: int,
) -> None:
    fft = getattr(detection, "fft", None)

    if fft is None:
        return

    threshold = clamp_float(0.95 - sensitivity * 0.004, 0.40, 0.99)
    percentile = clamp_float(99.0 - sensitivity * 0.15, 80.0, 99.9)

    set_if_exists(fft, "threshold", threshold)
    set_if_exists(fft, "mask_threshold", threshold)
    set_if_exists(fft, "threshold_percentile", percentile)
    set_if_exists(fft, "percentile", percentile)
    set_if_exists(fft, "dilate_iterations", mask_expand)
    set_if_exists(fft, "min_area", min_area)


def apply_anomaly_tuning(
    detection: Any,
    sensitivity: float,
    mask_expand: int,
    min_area: int,
) -> None:
    anomaly = getattr(detection, "anomaly", None)

    if anomaly is None:
        return

    threshold = clamp_float(3.5 - sensitivity * 0.025, 0.5, 5.0)

    set_if_exists(anomaly, "threshold", threshold)
    set_if_exists(anomaly, "z_threshold", threshold)
    set_if_exists(anomaly, "std_threshold", threshold)
    set_if_exists(anomaly, "dilate_iterations", mask_expand)
    set_if_exists(anomaly, "min_area", min_area)


def apply_ocr_tuning(
    detection: Any,
    mask_expand: int,
    min_area: int,
) -> None:
    for detector_name in ["paddle_ocr", "easy_ocr"]:
        ocr = getattr(detection, detector_name, None)

        if ocr is None:
            continue

        set_if_exists(ocr, "dilate_iterations", mask_expand)
        set_if_exists(ocr, "min_area", min_area)


def apply_fusion_tuning(
    detection: Any,
    fusion_strictness: float,
) -> None:
    fusion = getattr(detection, "fusion", None)

    if fusion is None:
        return

    threshold = clamp_float(0.20 + fusion_strictness * 0.006, 0.05, 0.95)
    min_votes = 1 if fusion_strictness < 50 else 2

    set_if_exists(fusion, "threshold", threshold)
    set_if_exists(fusion, "mask_threshold", threshold)
    set_if_exists(fusion, "confidence_threshold", threshold)
    set_if_exists(fusion, "min_votes", min_votes)


def set_ocr(config: AppConfig, name: OCRName) -> None:
    enabled = {
        "paddle_ocr": name in ("paddle", "both"),
        "easy_ocr": name in ("easy", "both"),
    }

    for detection in detection_configs(config):
        for detector_name, is_enabled in enabled.items():
            set_detector_enabled(detection, detector_name, is_enabled)

        selected_count = len(enabled_detectors_from_detection(detection))
        fusion_enabled = selected_count > 1

        set_if_exists(detection, "enable_fusion", fusion_enabled)
        set_detector_config_enabled(detection, "fusion", fusion_enabled)


def set_inpainter(config: AppConfig, name: InpainterName) -> None:
    config.image.inpainting.inpainter = name
    config.video.inpainting.inpainter = name

    for inpainting in [config.image.inpainting, config.video.inpainting]:
        inpainting.enable_opencv = name in ("auto", "opencv")
        inpainting.enable_lama = name in ("auto", "lama")
        inpainting.enable_stable_diffusion = name in ("auto", "stable_diffusion")
        inpainting.enable_sdxl = name in ("auto", "sdxl")
        inpainting.enable_flux = name in ("auto", "flux")


def set_tracker(config: AppConfig, name: TrackerName) -> None:
    config.video.tracking.enabled = name != "none"
    config.video.tracking.mode = name


def set_device(config: AppConfig, device: DeviceName) -> None:
    for detection in detection_configs(config):
        set_nested_attr(detection, "grounding_dino", "device", device)
        set_nested_attr(detection, "sam2", "device", device)
        set_nested_attr(detection, "mobile_sam_2", "device", device)
        set_nested_attr(detection, "easy_ocr", "gpu", device == "cuda")
        set_nested_attr(detection, "paddle_ocr", "use_gpu", device == "cuda")
        set_nested_attr(detection, "yolo", "device", None if device == "auto" else device)

    config.video.tracking.xmem.device = device
    config.video.tracking.cotracker.device = device

    for inpainting in [config.image.inpainting, config.video.inpainting]:
        inpainting.lama.device = device
        inpainting.stable_diffusion.device = device
        inpainting.sdxl.device = device
        inpainting.flux.device = device


def set_audio_backend(config: AppConfig, backend: AudioBackendName) -> None:
    if backend not in ("none", "moviepy", "ffmpeg"):
        raise ValueError(f"Unsupported audio backend: {backend}")

    config.video.rebuild.audio_backend = backend
    config.video.rebuild.copy_audio = backend != "none"


def disable_preprocessing(config: AppConfig) -> None:
    for preprocessing in [config.image.preprocessing, config.video.preprocessing]:
        for module_name in [
            "resize",
            "denoise",
            "gamma",
            "clahe",
            "edge_enhance",
            "fft_enhance",
            "patchify",
        ]:
            set_nested_attr(preprocessing, module_name, "enabled", False)

        preprocessing.enable_main_path = False
        preprocessing.enable_fft_path = False
        preprocessing.enable_patch_path = False


def disable_detection(config: AppConfig) -> None:
    for detection in detection_configs(config):
        disable_detection_config_only(detection)


def disable_inpainting(config: AppConfig) -> None:
    for inpainting in [config.image.inpainting, config.video.inpainting]:
        inpainting.inpainter = "opencv"

        inpainting.enable_opencv = True
        inpainting.enable_lama = False
        inpainting.enable_stable_diffusion = False
        inpainting.enable_sdxl = False
        inpainting.enable_flux = False

        for module_name in ["opencv", "lama", "stable_diffusion", "sdxl", "flux"]:
            set_nested_attr(inpainting, module_name, "enabled", False)


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
            set_nested_attr(postprocessing, module_name, "enabled", False)


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


def enabled_detectors_from_detection(detection: Any) -> list[str]:
    selected: list[str] = []

    for detector_name, flag_name in DETECTOR_ENABLE_FLAGS.items():
        if bool(getattr(detection, flag_name, False)):
            selected.append(detector_name)

    return selected


def get_mode_detection(config: AppConfig, mode: str) -> Any:
    return config.image.detection if mode == "image" else config.video.detection


def detection_configs(config: AppConfig) -> list[Any]:
    return [
        config.image.detection,
        config.video.detection,
    ]


def set_nested_attr(
    parent: Any,
    child_name: str,
    attr_name: str,
    value: Any,
) -> None:
    child = getattr(parent, child_name, None)

    if child is not None:
        set_if_exists(child, attr_name, value)
