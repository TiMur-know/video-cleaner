# ui/components/detector_settings.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class DetectorSettingsPanel:
    """
    Container returned by build_detector_settings_panel().

    visibility_outputs:
        Containers whose visibility changes when detector selection changes.

    inputs:
        All detector tuning inputs.

    input_names:
        Names matching values in inputs.
    """

    visibility_outputs: list[Any]
    inputs: list[Any]
    input_names: list[str]


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


def build_detector_settings_panel(
    gr: Any,
    selected_detectors: list[str] | None = None,
) -> DetectorSettingsPanel:
    """
    Build collapsible detector-specific settings.

    selected_detectors:
        Used for initial visibility.

    The UI now has two detector sections:
        proposal detectors
        refiner detectors

    This panel still receives one combined detector list:
        proposal_detectors + refiner_detectors
    """

    gr.Markdown("## Detector settings")

    input_components: list[Any] = []
    input_names: list[str] = []
    visibility_outputs: list[Any] = []

    with gr.Column(
        visible=is_detector_settings_visible("opencv", selected_detectors)
    ) as opencv_box:
        with gr.Accordion("OpenCV proposal detector settings", open=False):
            opencv_mode = gr.Dropdown(
                label="OpenCV mode",
                choices=["auto", "adaptive", "edges", "bright", "dark", "combined"],
                value="combined",
            )

            with gr.Row():
                opencv_canny_low = gr.Slider(
                    label="Canny low",
                    minimum=0,
                    maximum=255,
                    step=1,
                    value=50,
                )

                opencv_canny_high = gr.Slider(
                    label="Canny high",
                    minimum=0,
                    maximum=255,
                    step=1,
                    value=150,
                )

            with gr.Row():
                opencv_bright_percentile = gr.Slider(
                    label="Bright percentile",
                    minimum=70,
                    maximum=99.9,
                    step=0.1,
                    value=92,
                )

                opencv_dark_percentile = gr.Slider(
                    label="Dark percentile",
                    minimum=0.1,
                    maximum=30,
                    step=0.1,
                    value=8,
                )

            with gr.Row():
                opencv_min_area = gr.Number(
                    label="Min area",
                    value=25,
                    precision=0,
                )

                opencv_dilate_iterations = gr.Slider(
                    label="Dilate iterations",
                    minimum=0,
                    maximum=12,
                    step=1,
                    value=2,
                )

    visibility_outputs.append(opencv_box)

    add_inputs(
        input_components,
        input_names,
        {
            "opencv.mode": opencv_mode,
            "opencv.canny_low": opencv_canny_low,
            "opencv.canny_high": opencv_canny_high,
            "opencv.bright_percentile": opencv_bright_percentile,
            "opencv.dark_percentile": opencv_dark_percentile,
            "opencv.min_area": opencv_min_area,
            "opencv.dilate_iterations": opencv_dilate_iterations,
        },
    )

    with gr.Column(
        visible=is_detector_settings_visible("grounding_dino", selected_detectors)
    ) as grounding_dino_box:
        with gr.Accordion("GroundingDINO proposal detector settings", open=False):
            grounding_dino_model_id = gr.Textbox(
                label="Model ID",
                value="IDEA-Research/grounding-dino-base",
            )

            grounding_dino_prompt = gr.Textbox(
                label="Prompt",
                value=(
                    "watermark . logo . text watermark . transparent watermark . "
                    "faint watermark . low opacity watermark"
                ),
                lines=3,
                placeholder=(
                    "Example: watermark . logo . transparent watermark\n"
                    "Example: dark text watermark . white logo watermark"
                ),
            )

            with gr.Row():
                grounding_dino_box_threshold = gr.Slider(
                    label="Box threshold",
                    minimum=0,
                    maximum=1,
                    step=0.01,
                    value=0.25,
                )

                grounding_dino_text_threshold = gr.Slider(
                    label="Text threshold",
                    minimum=0,
                    maximum=1,
                    step=0.01,
                    value=0.20,
                )

            with gr.Row():
                grounding_dino_min_area = gr.Number(
                    label="Min area",
                    value=25,
                    precision=0,
                )

                grounding_dino_max_boxes = gr.Number(
                    label="Max boxes",
                    value=20,
                    precision=0,
                )

            with gr.Row():
                grounding_dino_strict_loading = gr.Checkbox(
                    label="Strict loading",
                    value=False,
                )

                grounding_dino_fallback_to_empty = gr.Checkbox(
                    label="Fallback to empty mask on error",
                    value=True,
                )

    visibility_outputs.append(grounding_dino_box)

    add_inputs(
        input_components,
        input_names,
        {
            "grounding_dino.model_id": grounding_dino_model_id,
            "grounding_dino.prompt": grounding_dino_prompt,
            "grounding_dino.box_threshold": grounding_dino_box_threshold,
            "grounding_dino.text_threshold": grounding_dino_text_threshold,
            "grounding_dino.min_area": grounding_dino_min_area,
            "grounding_dino.max_boxes": grounding_dino_max_boxes,
            "grounding_dino.strict_loading": grounding_dino_strict_loading,
            "grounding_dino.fallback_to_empty": grounding_dino_fallback_to_empty,
        },
    )

    with gr.Column(
        visible=is_detector_settings_visible("yolo", selected_detectors)
    ) as yolo_box:
        with gr.Accordion("YOLO proposal detector settings", open=False):
            with gr.Row():
                yolo_confidence = gr.Slider(
                    label="Confidence",
                    minimum=0,
                    maximum=1,
                    step=0.01,
                    value=0.25,
                )

                yolo_iou = gr.Slider(
                    label="IoU",
                    minimum=0,
                    maximum=1,
                    step=0.01,
                    value=0.45,
                )

            yolo_image_size = gr.Number(
                label="Image size",
                value=640,
                precision=0,
            )

    visibility_outputs.append(yolo_box)

    add_inputs(
        input_components,
        input_names,
        {
            "yolo.confidence": yolo_confidence,
            "yolo.confidence_threshold": yolo_confidence,
            "yolo.conf": yolo_confidence,
            "yolo.iou": yolo_iou,
            "yolo.iou_threshold": yolo_iou,
            "yolo.image_size": yolo_image_size,
            "yolo.imgsz": yolo_image_size,
        },
    )

    with gr.Column(
        visible=is_detector_settings_visible("fft", selected_detectors)
    ) as fft_box:
        with gr.Accordion("FFT proposal detector settings", open=False):
            with gr.Row():
                fft_threshold_percentile = gr.Slider(
                    label="Threshold percentile",
                    minimum=80,
                    maximum=99.9,
                    step=0.1,
                    value=95,
                )

                fft_min_area = gr.Number(
                    label="Min area",
                    value=25,
                    precision=0,
                )

            fft_dilate_iterations = gr.Slider(
                label="Dilate iterations",
                minimum=0,
                maximum=12,
                step=1,
                value=2,
            )

    visibility_outputs.append(fft_box)

    add_inputs(
        input_components,
        input_names,
        {
            "fft.threshold_percentile": fft_threshold_percentile,
            "fft.percentile": fft_threshold_percentile,
            "fft.min_area": fft_min_area,
            "fft.dilate_iterations": fft_dilate_iterations,
        },
    )

    with gr.Column(
        visible=is_detector_settings_visible("anomaly", selected_detectors)
    ) as anomaly_box:
        with gr.Accordion("Anomaly proposal detector settings", open=False):
            with gr.Row():
                anomaly_threshold = gr.Slider(
                    label="Threshold",
                    minimum=0.1,
                    maximum=8,
                    step=0.1,
                    value=2.5,
                )

                anomaly_min_area = gr.Number(
                    label="Min area",
                    value=25,
                    precision=0,
                )

            anomaly_dilate_iterations = gr.Slider(
                label="Dilate iterations",
                minimum=0,
                maximum=12,
                step=1,
                value=2,
            )

    visibility_outputs.append(anomaly_box)

    add_inputs(
        input_components,
        input_names,
        {
            "anomaly.threshold": anomaly_threshold,
            "anomaly.z_threshold": anomaly_threshold,
            "anomaly.std_threshold": anomaly_threshold,
            "anomaly.min_area": anomaly_min_area,
            "anomaly.dilate_iterations": anomaly_dilate_iterations,
        },
    )

    with gr.Column(
        visible=is_detector_settings_visible("paddle_ocr", selected_detectors)
    ) as paddle_ocr_box:
        with gr.Accordion("PaddleOCR proposal detector settings", open=False):
            with gr.Row():
                paddle_confidence_threshold = gr.Slider(
                    label="Confidence threshold",
                    minimum=0,
                    maximum=1,
                    step=0.01,
                    value=0.4,
                )

                paddle_min_area = gr.Number(
                    label="Min area",
                    value=25,
                    precision=0,
                )

            paddle_dilate_iterations = gr.Slider(
                label="Dilate iterations",
                minimum=0,
                maximum=12,
                step=1,
                value=2,
            )

    visibility_outputs.append(paddle_ocr_box)

    add_inputs(
        input_components,
        input_names,
        {
            "paddle_ocr.confidence_threshold": paddle_confidence_threshold,
            "paddle_ocr.confidence": paddle_confidence_threshold,
            "paddle_ocr.min_area": paddle_min_area,
            "paddle_ocr.dilate_iterations": paddle_dilate_iterations,
        },
    )

    with gr.Column(
        visible=is_detector_settings_visible("easy_ocr", selected_detectors)
    ) as easy_ocr_box:
        with gr.Accordion("EasyOCR proposal detector settings", open=False):
            with gr.Row():
                easy_confidence_threshold = gr.Slider(
                    label="Confidence threshold",
                    minimum=0,
                    maximum=1,
                    step=0.01,
                    value=0.4,
                )

                easy_min_area = gr.Number(
                    label="Min area",
                    value=25,
                    precision=0,
                )

            easy_dilate_iterations = gr.Slider(
                label="Dilate iterations",
                minimum=0,
                maximum=12,
                step=1,
                value=2,
            )

    visibility_outputs.append(easy_ocr_box)

    add_inputs(
        input_components,
        input_names,
        {
            "easy_ocr.confidence_threshold": easy_confidence_threshold,
            "easy_ocr.confidence": easy_confidence_threshold,
            "easy_ocr.min_area": easy_min_area,
            "easy_ocr.dilate_iterations": easy_dilate_iterations,
        },
    )

    with gr.Column(
        visible=is_detector_settings_visible("sam2", selected_detectors)
    ) as sam2_box:
        with gr.Accordion("SAM2 refiner detector settings", open=False):
            sam2_checkpoint_path = gr.Textbox(
                label="Checkpoint path",
                value="models/sam2/sam2_b.pt",
            )

            sam2_model_config_path = gr.Textbox(
                label="Model config path",
                value="configs/sam2/sam2_hiera_b+.yaml",
            )

            with gr.Row():
                sam2_min_area = gr.Number(
                    label="Min area",
                    value=50,
                    precision=0,
                )

                sam2_mask_threshold = gr.Slider(
                    label="Mask threshold",
                    minimum=0,
                    maximum=1,
                    step=0.01,
                    value=0.5,
                )

            with gr.Row():
                sam2_dilate_iterations = gr.Slider(
                    label="Dilate iterations",
                    minimum=0,
                    maximum=12,
                    step=1,
                    value=1,
                )

                sam2_morph_kernel_size = gr.Slider(
                    label="Morph kernel size",
                    minimum=3,
                    maximum=21,
                    step=2,
                    value=5,
                )

            with gr.Row():
                sam2_require_prompts = gr.Checkbox(
                    label="Require prompts",
                    value=True,
                )

                sam2_strict_loading = gr.Checkbox(
                    label="Strict loading",
                    value=False,
                )

            sam2_use_auto_watermark_prompts = gr.Checkbox(
                label="Auto-generate watermark prompts",
                value=True,
            )

            sam2_use_prompt_text = gr.Checkbox(
                label="Use editable prompt text",
                value=True,
            )

            sam2_prompt_text = gr.Textbox(
                label="SAM2 watermark prompt",
                value=(
                    "dark watermark, light watermark, low opacity watermark, "
                    "transparent watermark, text watermark, logo"
                ),
                lines=3,
                placeholder=(
                    "Example: dark low opacity text watermark logo\n"
                    "Example: transparent white watermark\n"
                    "Example: black text watermark"
                ),
            )

            with gr.Row():
                sam2_prompt_dark_watermark = gr.Checkbox(
                    label="Look for dark watermark",
                    value=True,
                )

                sam2_prompt_light_watermark = gr.Checkbox(
                    label="Look for light watermark",
                    value=True,
                )

            with gr.Row():
                sam2_prompt_low_opacity_watermark = gr.Checkbox(
                    label="Look for low-opacity watermark",
                    value=True,
                )

                sam2_prompt_text_watermark = gr.Checkbox(
                    label="Look for text-like watermark",
                    value=True,
                )

            sam2_prompt_sensitivity = gr.Slider(
                label="Prompt sensitivity",
                minimum=0,
                maximum=100,
                step=1,
                value=50,
            )

            with gr.Row():
                sam2_prompt_min_area = gr.Number(
                    label="Prompt min area",
                    value=25,
                    precision=0,
                )

                sam2_prompt_max_boxes = gr.Number(
                    label="Max prompt boxes",
                    value=20,
                    precision=0,
                )

            sam2_prompt_max_area_ratio = gr.Slider(
                label="Prompt max area ratio",
                minimum=0.01,
                maximum=1.0,
                step=0.01,
                value=0.35,
            )

            with gr.Row():
                sam2_dark_percentile = gr.Slider(
                    label="Dark percentile",
                    minimum=1,
                    maximum=35,
                    step=0.1,
                    value=8,
                )

                sam2_light_percentile = gr.Slider(
                    label="Light percentile",
                    minimum=65,
                    maximum=99,
                    step=0.1,
                    value=92,
                )

            with gr.Row():
                sam2_low_opacity_edge_low = gr.Slider(
                    label="Low opacity edge low",
                    minimum=1,
                    maximum=255,
                    step=1,
                    value=30,
                )

                sam2_low_opacity_edge_high = gr.Slider(
                    label="Low opacity edge high",
                    minimum=1,
                    maximum=255,
                    step=1,
                    value=100,
                )

    visibility_outputs.append(sam2_box)

    add_inputs(
        input_components,
        input_names,
        {
            "sam2.checkpoint_path": sam2_checkpoint_path,
            "sam2.model_config_path": sam2_model_config_path,
            "sam2.min_area": sam2_min_area,
            "sam2.mask_threshold": sam2_mask_threshold,
            "sam2.dilate_iterations": sam2_dilate_iterations,
            "sam2.morph_kernel_size": sam2_morph_kernel_size,
            "sam2.require_prompts": sam2_require_prompts,
            "sam2.strict_loading": sam2_strict_loading,
            "sam2.use_auto_watermark_prompts": sam2_use_auto_watermark_prompts,
            "sam2.use_prompt_text": sam2_use_prompt_text,
            "sam2.prompt_text": sam2_prompt_text,
            "sam2.prompt_dark_watermark": sam2_prompt_dark_watermark,
            "sam2.prompt_light_watermark": sam2_prompt_light_watermark,
            "sam2.prompt_low_opacity_watermark": sam2_prompt_low_opacity_watermark,
            "sam2.prompt_text_watermark": sam2_prompt_text_watermark,
            "sam2.prompt_sensitivity": sam2_prompt_sensitivity,
            "sam2.prompt_min_area": sam2_prompt_min_area,
            "sam2.prompt_max_boxes": sam2_prompt_max_boxes,
            "sam2.prompt_max_area_ratio": sam2_prompt_max_area_ratio,
            "sam2.dark_percentile": sam2_dark_percentile,
            "sam2.light_percentile": sam2_light_percentile,
            "sam2.low_opacity_edge_low": sam2_low_opacity_edge_low,
            "sam2.low_opacity_edge_high": sam2_low_opacity_edge_high,
        },
    )

    with gr.Column(
        visible=is_detector_settings_visible("mobile_sam_2", selected_detectors)
    ) as mobile_sam_2_box:
        with gr.Accordion("MobileSAM2 refiner detector settings", open=False):
            with gr.Row():
                mobile_sam_2_mask_threshold = gr.Slider(
                    label="Mask threshold",
                    minimum=0,
                    maximum=1,
                    step=0.01,
                    value=0.5,
                )

                mobile_sam_2_min_area = gr.Number(
                    label="Min area",
                    value=25,
                    precision=0,
                )

            mobile_sam_2_require_prompts = gr.Checkbox(
                label="Require prompts",
                value=True,
            )

    visibility_outputs.append(mobile_sam_2_box)

    add_inputs(
        input_components,
        input_names,
        {
            "mobile_sam_2.mask_threshold": mobile_sam_2_mask_threshold,
            "mobile_sam_2.threshold": mobile_sam_2_mask_threshold,
            "mobile_sam_2.min_area": mobile_sam_2_min_area,
            "mobile_sam_2.require_prompts": mobile_sam_2_require_prompts,
        },
    )

    with gr.Column(
        visible=is_detector_settings_visible("fusion", selected_detectors)
    ) as fusion_box:
        with gr.Accordion("Fusion settings", open=False):
            with gr.Row():
                fusion_threshold = gr.Slider(
                    label="Fusion threshold",
                    minimum=0,
                    maximum=1,
                    step=0.01,
                    value=0.5,
                )

                fusion_min_votes = gr.Number(
                    label="Min votes",
                    value=2,
                    precision=0,
                )

    visibility_outputs.append(fusion_box)

    add_inputs(
        input_components,
        input_names,
        {
            "fusion.threshold": fusion_threshold,
            "fusion.mask_threshold": fusion_threshold,
            "fusion.confidence_threshold": fusion_threshold,
            "fusion.min_votes": fusion_min_votes,
        },
    )

    return DetectorSettingsPanel(
        visibility_outputs=visibility_outputs,
        inputs=input_components,
        input_names=input_names,
    )


def is_detector_settings_visible(
    detector_name: str,
    selected_detectors: list[str] | None,
) -> bool:
    selected = set(selected_detectors or [])

    if detector_name == "fusion":
        return len(selected) > 1

    return detector_name in selected


def add_inputs(
    input_components: list[Any],
    input_names: list[str],
    mapping: dict[str, Any],
) -> None:
    for name, component in mapping.items():
        input_names.append(name)
        input_components.append(component)