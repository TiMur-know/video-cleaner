# ui/components/image_edit.py

from __future__ import annotations

from typing import Any

from controls.control_utils import (
    PROPOSAL_DETECTOR_CHOICES,
    REFINER_DETECTOR_CHOICES,
)
from ui.components.detector_settings import build_detector_settings_panel
from ui.components.processing_settings import (
    build_postprocessing_settings_panel,
    build_preprocessing_settings_panel,
    build_processing_tuning_dict,
    merge_processing_settings,
)
from ui.handlers.detector_settings_handlers import (
    build_detector_tuning_dict,
    update_detector_settings_visibility,
)
from ui.handlers.image_handlers import run_image_from_ui


DEFAULT_IMAGE_PROPOSAL_DETECTORS = [
    "opencv",
    "fft",
    "anomaly",
]

DEFAULT_IMAGE_REFINER_DETECTORS: list[str] = []


def build_image_tab(gr: Any, config_path: str | None) -> None:
    default_selected_detectors = [
        *DEFAULT_IMAGE_PROPOSAL_DETECTORS,
        *DEFAULT_IMAGE_REFINER_DETECTORS,
    ]

    with gr.Row():
        image_input = gr.File(
            label="Image input",
            file_types=["image"],
        )

        output_path = gr.Textbox(
            label="Output image path",
            value="data/outputs/output.png",
        )

        mask_output_path = gr.Textbox(
            label="Output mask path",
            value="data/masks/mask.png",
        )

    with gr.Row():
        preset = gr.Dropdown(
            label="Preset",
            choices=["default", "fast", "balanced", "quality"],
            value="balanced",
        )

        inpainter = gr.Dropdown(
            label="Inpainter",
            choices=["auto", "opencv", "lama", "sdxl", "flux"],
            value="auto",
        )

        device = gr.Dropdown(
            label="Device",
            choices=["auto", "cpu", "cuda", "mps"],
            value="auto",
        )

    preprocessing_settings = build_preprocessing_settings_panel(gr)

    gr.Markdown("## Detection")

    proposal_detectors = gr.CheckboxGroup(
        label="Proposal detectors: find possible watermark regions",
        choices=list(PROPOSAL_DETECTOR_CHOICES),
        value=DEFAULT_IMAGE_PROPOSAL_DETECTORS,
    )

    refiner_detectors = gr.CheckboxGroup(
        label="Refiner detectors: improve masks from proposal results",
        choices=list(REFINER_DETECTOR_CHOICES),
        value=DEFAULT_IMAGE_REFINER_DETECTORS,
    )

    gr.Markdown("## General detection tuning")

    with gr.Row():
        detection_sensitivity = gr.Slider(
            label="Detection sensitivity",
            minimum=0,
            maximum=100,
            step=1,
            value=50,
        )

        mask_expand = gr.Slider(
            label="Mask expand",
            minimum=0,
            maximum=12,
            step=1,
            value=2,
        )

    with gr.Row():
        min_area = gr.Number(
            label="Min area",
            value=25,
            precision=0,
        )

        fusion_strictness = gr.Slider(
            label="Fusion strictness",
            minimum=0,
            maximum=100,
            step=1,
            value=50,
        )

    detector_settings = build_detector_settings_panel(
        gr,
        selected_detectors=default_selected_detectors,
    )

    proposal_detectors.change(
        fn=lambda proposal_values, refiner_values: update_detector_settings_visibility(
            [
                *(proposal_values or []),
                *(refiner_values or []),
            ]
        ),
        inputs=[
            proposal_detectors,
            refiner_detectors,
        ],
        outputs=detector_settings.visibility_outputs,
    )

    refiner_detectors.change(
        fn=lambda proposal_values, refiner_values: update_detector_settings_visibility(
            [
                *(proposal_values or []),
                *(refiner_values or []),
            ]
        ),
        inputs=[
            proposal_detectors,
            refiner_detectors,
        ],
        outputs=detector_settings.visibility_outputs,
    )

    postprocessing_settings = build_postprocessing_settings_panel(gr)
    processing_settings = merge_processing_settings(
        preprocessing_settings,
        postprocessing_settings,
    )

    run_button = gr.Button(
        "Run image pipeline",
        variant="primary",
    )

    with gr.Row():
        original_preview = gr.Image(
            label="Original image",
            type="numpy",
        )

        preprocessed_preview = gr.Image(
            label="After preprocessing",
            type="numpy",
        )

        mask_preview = gr.Image(
            label="Detected mask",
            type="numpy",
        )

    with gr.Row():
        inpainted_preview = gr.Image(
            label="After inpainting",
            type="numpy",
        )

        final_preview = gr.Image(
            label="After postprocessing",
            type="numpy",
        )

    metadata_output = gr.JSON(label="Pipeline metadata")

    run_button.click(
        fn=lambda image_file,
        out_path,
        mask_path,
        preset_value,
        proposal_detector_values,
        refiner_detector_values,
        inpainter_value,
        device_value,
        sensitivity_value,
        mask_expand_value,
        min_area_value,
        fusion_strictness_value,
        preprocessing_module_values,
        postprocessing_module_values,
        *detector_setting_values: run_image_from_ui(
            config_path=config_path,
            image_file=image_file,
            output_path=out_path,
            mask_output_path=mask_path,
            preset=preset_value,
            proposal_detectors=proposal_detector_values,
            refiner_detectors=refiner_detector_values,
            inpainter=inpainter_value,
            device=device_value,
            detection_sensitivity=sensitivity_value,
            mask_expand=mask_expand_value,
            min_area=min_area_value,
            fusion_strictness=fusion_strictness_value,
            detector_tuning=build_detector_tuning_dict(
                detector_settings.input_names,
                list(detector_setting_values[: len(detector_settings.inputs)]),
            ),
            preprocessing_modules=preprocessing_module_values,
            postprocessing_modules=postprocessing_module_values,
            processing_tuning=build_processing_tuning_dict(
                processing_settings.input_names,
                list(detector_setting_values[len(detector_settings.inputs) :]),
            ),
        ),
        inputs=[
            image_input,
            output_path,
            mask_output_path,
            preset,
            proposal_detectors,
            refiner_detectors,
            inpainter,
            device,
            detection_sensitivity,
            mask_expand,
            min_area,
            fusion_strictness,
            processing_settings.preprocessing_modules,
            processing_settings.postprocessing_modules,
            *detector_settings.inputs,
            *processing_settings.inputs,
        ],
        outputs=[
            original_preview,
            preprocessed_preview,
            mask_preview,
            inpainted_preview,
            final_preview,
            metadata_output,
        ],
    )
