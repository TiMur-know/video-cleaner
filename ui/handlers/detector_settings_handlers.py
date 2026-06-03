# ui/handlers/detector_settings_handlers.py

from __future__ import annotations

from typing import Any


DETECTOR_SETTING_ORDER = [
    # Proposal detectors
    "opencv",
    "grounding_dino",
    "yolo",
    "fft",
    "anomaly",
    "paddle_ocr",
    "easy_ocr",

    # Refiner detectors
    "sam2",
    "mobile_sam_2",

    # Final / stage fusion
    "fusion",
]


def update_detector_settings_visibility(
    selected_detectors: list[str] | None,
) -> tuple[Any, ...]:
    """
    Show only settings for selected detectors.

    selected_detectors should be combined before this function is called:

        proposal_detectors + refiner_detectors

    Fusion settings are shown only when 2 or more detectors are selected.
    """

    gr = lazy_import_gradio()

    selected = set(selected_detectors or [])

    updates = []

    for detector_name in DETECTOR_SETTING_ORDER:
        if detector_name == "fusion":
            visible = len(selected) > 1
        else:
            visible = detector_name in selected

        updates.append(gr.update(visible=visible))

    return tuple(updates)


def build_detector_tuning_dict(
    input_names: list[str],
    values: list[Any],
) -> dict[str, dict[str, Any]]:
    """
    Build nested tuning dictionary from UI values.

    Example output:
        {
            "opencv": {
                "mode": "combined",
                "canny_low": 50,
                "canny_high": 150,
            },
            "sam2": {
                "prompt_text": "dark low opacity watermark",
                "use_prompt_text": true,
            },
            "fusion": {
                "threshold": 0.5,
                "min_votes": 2,
            },
        }
    """

    tuning: dict[str, dict[str, Any]] = {}

    for name, value in zip(input_names, values):
        if "." not in name:
            continue

        detector_name, field_name = name.split(".", 1)

        if detector_name not in tuning:
            tuning[detector_name] = {}

        tuning[detector_name][field_name] = clean_value(value)

    return tuning


def build_detector_tuning_dict_from_kwargs(
    **kwargs: Any,
) -> dict[str, dict[str, Any]]:
    """
    Alternative helper if you prefer kwargs.

    Example:
        build_detector_tuning_dict_from_kwargs(
            opencv_canny_low=50,
            sam2_prompt_text="dark watermark",
        )
    """

    tuning: dict[str, dict[str, Any]] = {}

    for key, value in kwargs.items():
        if "_" not in key:
            continue

        detector_name, field_name = split_detector_key(key)

        if detector_name not in tuning:
            tuning[detector_name] = {}

        tuning[detector_name][field_name] = clean_value(value)

    return tuning


def split_detector_key(key: str) -> tuple[str, str]:
    known_prefixes = [
        "mobile_sam_2",
        "grounding_dino",
        "paddle_ocr",
        "easy_ocr",
        "opencv",
        "yolo",
        "fft",
        "anomaly",
        "sam2",
        "fusion",
    ]

    for prefix in known_prefixes:
        marker = f"{prefix}_"

        if key.startswith(marker):
            return prefix, key[len(marker):]

    parts = key.split("_", 1)

    if len(parts) == 1:
        return "unknown", key

    return parts[0], parts[1]


def clean_value(value: Any) -> Any:
    """
    Normalize Gradio values before applying to config.
    """

    if isinstance(value, float):
        if value.is_integer():
            return int(value)

        return value

    return value


def lazy_import_gradio() -> Any:
    try:
        import gradio as gr

        return gr

    except ImportError as exc:
        raise ImportError(
            "Gradio is required for detector settings UI. Install it with:\n"
            "    pip install gradio"
        ) from exc