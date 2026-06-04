# ui/components/processing_settings.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ui.control_defaults import (
    POSTPROCESSING_CHOICES,
    POSTPROCESSING_DEFAULTS,
    PREPROCESSING_CHOICES,
    PREPROCESSING_DEFAULTS,
)


@dataclass(slots=True)
class ProcessingSettingsPanel:
    preprocessing_modules: Any | None
    postprocessing_modules: Any | None
    inputs: list[Any]
    input_names: list[str]


def build_preprocessing_settings_panel(gr: Any) -> ProcessingSettingsPanel:
    gr.Markdown("## Preprocessing")

    preprocessing_modules = gr.CheckboxGroup(
        label="Preprocessing stages",
        choices=PREPROCESSING_CHOICES,
        value=PREPROCESSING_DEFAULTS["modules"],
    )

    input_components: list[Any] = []
    input_names: list[str] = []

    with gr.Accordion("Preprocessing settings", open=False):
        with gr.Row():
            resize_width = gr.Number(
                label="Resize width, 0 = original",
                value=PREPROCESSING_DEFAULTS["resize_width"],
                precision=0,
            )

            resize_height = gr.Number(
                label="Resize height, 0 = original",
                value=PREPROCESSING_DEFAULTS["resize_height"],
                precision=0,
            )

        with gr.Row():
            resize_keep_aspect = gr.Checkbox(
                label="Keep aspect ratio",
                value=PREPROCESSING_DEFAULTS["resize_keep_aspect"],
            )

            resize_only_downscale = gr.Checkbox(
                label="Only downscale",
                value=PREPROCESSING_DEFAULTS["resize_only_downscale"],
            )

        with gr.Row():
            denoise_strength = gr.Slider(
                label="Denoise strength",
                minimum=0,
                maximum=150,
                step=1,
                value=PREPROCESSING_DEFAULTS["denoise_strength"],
            )

            denoise_kernel_size = gr.Slider(
                label="Denoise kernel size",
                minimum=1,
                maximum=21,
                step=2,
                value=PREPROCESSING_DEFAULTS["denoise_kernel_size"],
            )

        with gr.Row():
            gamma = gr.Slider(
                label="Gamma",
                minimum=0.2,
                maximum=3.0,
                step=0.05,
                value=PREPROCESSING_DEFAULTS["gamma"],
            )

            gain = gr.Slider(
                label="Gain",
                minimum=0.2,
                maximum=3.0,
                step=0.05,
                value=PREPROCESSING_DEFAULTS["gain"],
            )

        with gr.Row():
            clahe_clip_limit = gr.Slider(
                label="CLAHE clip limit",
                minimum=0.1,
                maximum=8.0,
                step=0.1,
                value=PREPROCESSING_DEFAULTS["clahe_clip_limit"],
            )

            clahe_tile_size = gr.Slider(
                label="CLAHE tile size",
                minimum=2,
                maximum=32,
                step=1,
                value=PREPROCESSING_DEFAULTS["clahe_tile_size"],
            )

        with gr.Row():
            edge_strength = gr.Slider(
                label="Edge strength",
                minimum=0,
                maximum=4.0,
                step=0.05,
                value=PREPROCESSING_DEFAULTS["edge_strength"],
            )

            fft_strength = gr.Slider(
                label="FFT strength",
                minimum=0,
                maximum=1.0,
                step=0.01,
                value=PREPROCESSING_DEFAULTS["fft_strength"],
            )

        fft_high_pass_radius = gr.Slider(
            label="FFT high-pass radius",
            minimum=1,
            maximum=120,
            step=1,
            value=PREPROCESSING_DEFAULTS["fft_high_pass_radius"],
        )

    add_inputs(
        input_components,
        input_names,
        {
            "resize_width": resize_width,
            "resize_height": resize_height,
            "resize_keep_aspect": resize_keep_aspect,
            "resize_only_downscale": resize_only_downscale,
            "denoise_strength": denoise_strength,
            "denoise_kernel_size": denoise_kernel_size,
            "gamma": gamma,
            "gain": gain,
            "clahe_clip_limit": clahe_clip_limit,
            "clahe_tile_size": clahe_tile_size,
            "edge_strength": edge_strength,
            "fft_strength": fft_strength,
            "fft_high_pass_radius": fft_high_pass_radius,
        },
    )

    return ProcessingSettingsPanel(
        preprocessing_modules=preprocessing_modules,
        postprocessing_modules=None,
        inputs=input_components,
        input_names=input_names,
    )


def build_postprocessing_settings_panel(gr: Any) -> ProcessingSettingsPanel:
    gr.Markdown("## Postprocessing")

    postprocessing_modules = gr.CheckboxGroup(
        label="Postprocessing stages",
        choices=POSTPROCESSING_CHOICES,
        value=POSTPROCESSING_DEFAULTS["modules"],
    )

    input_components: list[Any] = []
    input_names: list[str] = []

    with gr.Accordion("Postprocessing settings", open=False):
        with gr.Row():
            color_match_max_shift = gr.Slider(
                label="Color match max shift",
                minimum=0,
                maximum=100,
                step=1,
                value=POSTPROCESSING_DEFAULTS["color_match_max_shift"],
            )

            color_match_max_scale = gr.Slider(
                label="Color match max scale",
                minimum=1.0,
                maximum=4.0,
                step=0.05,
                value=POSTPROCESSING_DEFAULTS["color_match_max_scale"],
            )

        with gr.Row():
            seam_strength = gr.Slider(
                label="Seam blend strength",
                minimum=0,
                maximum=1.0,
                step=0.01,
                value=POSTPROCESSING_DEFAULTS["seam_strength"],
            )

            seam_blur_kernel_size = gr.Slider(
                label="Seam blur size",
                minimum=3,
                maximum=61,
                step=2,
                value=POSTPROCESSING_DEFAULTS["seam_blur_kernel_size"],
            )

        seam_dilate_iterations = gr.Slider(
            label="Seam dilate iterations",
            minimum=0,
            maximum=12,
            step=1,
            value=POSTPROCESSING_DEFAULTS["seam_dilate_iterations"],
        )

        with gr.Row():
            artifact_kernel_size = gr.Slider(
                label="Artifact kernel size",
                minimum=1,
                maximum=15,
                step=2,
                value=POSTPROCESSING_DEFAULTS["artifact_kernel_size"],
            )

            artifact_dilate_iterations = gr.Slider(
                label="Artifact mask dilate",
                minimum=0,
                maximum=8,
                step=1,
                value=POSTPROCESSING_DEFAULTS["artifact_dilate_iterations"],
            )

        with gr.Row():
            sharpen_strength = gr.Slider(
                label="Sharpen strength",
                minimum=0,
                maximum=2.0,
                step=0.05,
                value=POSTPROCESSING_DEFAULTS["sharpen_strength"],
            )

            sharpen_blur_kernel_size = gr.Slider(
                label="Sharpen blur size",
                minimum=3,
                maximum=21,
                step=2,
                value=POSTPROCESSING_DEFAULTS["sharpen_blur_kernel_size"],
            )

        with gr.Row():
            temporal_alpha = gr.Slider(
                label="Temporal smoothing alpha",
                minimum=0,
                maximum=1.0,
                step=0.01,
                value=POSTPROCESSING_DEFAULTS["temporal_alpha"],
            )

            temporal_window_size = gr.Slider(
                label="Temporal window size",
                minimum=1,
                maximum=15,
                step=2,
                value=POSTPROCESSING_DEFAULTS["temporal_window_size"],
            )

    add_inputs(
        input_components,
        input_names,
        {
            "color_match_max_shift": color_match_max_shift,
            "color_match_max_scale": color_match_max_scale,
            "seam_strength": seam_strength,
            "seam_blur_kernel_size": seam_blur_kernel_size,
            "seam_dilate_iterations": seam_dilate_iterations,
            "artifact_kernel_size": artifact_kernel_size,
            "artifact_dilate_iterations": artifact_dilate_iterations,
            "sharpen_strength": sharpen_strength,
            "sharpen_blur_kernel_size": sharpen_blur_kernel_size,
            "temporal_alpha": temporal_alpha,
            "temporal_window_size": temporal_window_size,
        },
    )

    return ProcessingSettingsPanel(
        preprocessing_modules=None,
        postprocessing_modules=postprocessing_modules,
        inputs=input_components,
        input_names=input_names,
    )


def build_processing_settings_panel(gr: Any) -> ProcessingSettingsPanel:
    preprocessing_settings = build_preprocessing_settings_panel(gr)
    postprocessing_settings = build_postprocessing_settings_panel(gr)

    return ProcessingSettingsPanel(
        preprocessing_modules=preprocessing_settings.preprocessing_modules,
        postprocessing_modules=postprocessing_settings.postprocessing_modules,
        inputs=[
            *preprocessing_settings.inputs,
            *postprocessing_settings.inputs,
        ],
        input_names=[
            *preprocessing_settings.input_names,
            *postprocessing_settings.input_names,
        ],
    )


def merge_processing_settings(
    *settings: ProcessingSettingsPanel,
) -> ProcessingSettingsPanel:
    preprocessing_modules = None
    postprocessing_modules = None
    inputs: list[Any] = []
    input_names: list[str] = []

    for setting in settings:
        if setting.preprocessing_modules is not None:
            preprocessing_modules = setting.preprocessing_modules

        if setting.postprocessing_modules is not None:
            postprocessing_modules = setting.postprocessing_modules

        inputs.extend(setting.inputs)
        input_names.extend(setting.input_names)

    return ProcessingSettingsPanel(
        preprocessing_modules=preprocessing_modules,
        postprocessing_modules=postprocessing_modules,
        inputs=inputs,
        input_names=input_names,
    )


def build_processing_tuning_dict(
    input_names: list[str],
    values: list[Any],
) -> dict[str, Any]:
    return {
        name: value
        for name, value in zip(input_names, values)
    }


def add_inputs(
    input_components: list[Any],
    input_names: list[str],
    mapping: dict[str, Any],
) -> None:
    for name, component in mapping.items():
        input_names.append(name)
        input_components.append(component)
