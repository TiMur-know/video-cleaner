# controls/detection_control.py

from __future__ import annotations

from typing import Any

from controls.control_utils import (
    clamp_float,
    normalize_detectors,
    set_if_exists,
)
from utils.enums import OCR
from core.config import AppConfig


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


def disable_detection(config: AppConfig) -> None:
    for detection in detection_configs(config):
        disable_detection_config_only(detection)


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
        apply_grounding_dino_tuning(detection, sensitivity, min_area)
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


def apply_grounding_dino_tuning(
    detection: Any,
    sensitivity: float,
    min_area: int,
) -> None:
    grounding_dino = getattr(detection, "grounding_dino", None)

    if grounding_dino is None:
        return

    # Higher sensitivity => lower thresholds.
    box_threshold = clamp_float(0.40 - sensitivity * 0.003, 0.10, 0.60)
    text_threshold = clamp_float(0.35 - sensitivity * 0.0025, 0.10, 0.60)

    set_if_exists(grounding_dino, "box_threshold", box_threshold)
    set_if_exists(grounding_dino, "text_threshold", text_threshold)
    set_if_exists(grounding_dino, "min_area", min_area)


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
        set_if_exists(ocr, "box_padding", mask_expand)
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


def set_ocr(config: AppConfig, name: OCR | str) -> None:
    if isinstance(name, str):
        try:
            name = OCR(name)
        except Exception as exc:
            raise ValueError(f"Unsupported OCR option: {name}") from exc

    enabled = {
        "paddle_ocr": name in (OCR.PADDLE, OCR.BOTH),
        "easy_ocr": name in (OCR.EASY, OCR.BOTH),
    }

    for detection in detection_configs(config):
        for detector_name, is_enabled in enabled.items():
            set_detector_enabled(detection, detector_name, is_enabled)

        selected_count = len(enabled_detectors_from_detection(detection))
        fusion_enabled = selected_count > 1

        set_if_exists(detection, "enable_fusion", fusion_enabled)
        set_detector_config_enabled(detection, "fusion", fusion_enabled)


def enabled_detectors_from_detection(detection: Any) -> list[str]:
    selected: list[str] = []

    for detector_name, flag_name in DETECTOR_ENABLE_FLAGS.items():
        if bool(getattr(detection, flag_name, False)):
            selected.append(detector_name)

    return selected


def get_mode_detection(config: AppConfig, mode: str) -> Any:
    match mode:
        case "image":
            return config.image.detection

        case "video":
            return config.video.detection

        case _:
            raise ValueError(f"Unsupported detection mode: {mode}")


def detection_configs(config: AppConfig) -> list[Any]:
    return [
        config.image.detection,
        config.video.detection,
    ]


def set_device_for_detectors(
    config: AppConfig,
    device_value: str,
    is_cuda: bool,
) -> None:
    for detection in detection_configs(config):
        _set_nested_attr(detection, "grounding_dino", "device", device_value)
        _set_nested_attr(detection, "sam2", "device", device_value)
        _set_nested_attr(detection, "mobile_sam_2", "device", device_value)
        _set_nested_attr(detection, "easy_ocr", "gpu", is_cuda)
        _set_nested_attr(detection, "paddle_ocr", "use_gpu", is_cuda)
        _set_nested_attr(
            detection,
            "yolo",
            "device",
            None if device_value == "auto" else device_value,
        )


def _set_nested_attr(
    parent: Any,
    child_name: str,
    attr_name: str,
    value: Any,
) -> None:
    child = getattr(parent, child_name, None)

    if child is not None:
        set_if_exists(child, attr_name, value)
