# ui/components/mask_processing_settings.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ui.control_defaults import (
    MASK_PROCESSING_DEFAULTS,
    MASK_PROCESSING_STAGE_CHOICES,
)


@dataclass(slots=True)
class MaskProcessingSettingsPanel:
    enabled: Any
    stages: Any
    inputs: list[Any]
    input_names: list[str]


def build_mask_processing_settings_panel(gr: Any) -> MaskProcessingSettingsPanel:
    gr.Markdown("## Mask processing")

    enabled = gr.Checkbox(
        label="Enable mask processing",
        value=MASK_PROCESSING_DEFAULTS["enabled"],
    )

    stages = gr.CheckboxGroup(
        label="Mask processing stages",
        choices=MASK_PROCESSING_STAGE_CHOICES,
        value=MASK_PROCESSING_DEFAULTS["stages"],
    )

    input_components: list[Any] = []
    input_names: list[str] = []

    with gr.Accordion("Mask processing settings", open=False):
        with gr.Row():
            use_candidate_selection = gr.Checkbox(
                label="Compare candidate masks",
                value=MASK_PROCESSING_DEFAULTS["use_candidate_selection"],
            )

            candidate_min_area = gr.Number(
                label="Candidate min area",
                value=MASK_PROCESSING_DEFAULTS["candidate_min_area"],
                precision=0,
            )

            candidate_max_area_ratio = gr.Slider(
                label="Candidate max area ratio",
                minimum=0.01,
                maximum=1.0,
                step=0.01,
                value=MASK_PROCESSING_DEFAULTS["candidate_max_area_ratio"],
            )

        with gr.Row():
            include_detection_masks = gr.Checkbox(
                label="Include detector masks in fusion",
                value=MASK_PROCESSING_DEFAULTS["include_detection_masks"],
            )

            include_support_masks = gr.Checkbox(
                label="Include support masks",
                value=MASK_PROCESSING_DEFAULTS["include_support_masks"],
            )

            fallback_to_detection_mask = gr.Checkbox(
                label="Fallback to detector mask",
                value=MASK_PROCESSING_DEFAULTS["fallback_to_detection_mask"],
            )

        with gr.Row():
            box_filter_method = gr.Dropdown(
                label="Box filter method",
                choices=["residual", "adaptive", "combined"],
                value=MASK_PROCESSING_DEFAULTS["box_filter_method"],
            )

            box_filter_support_mode = gr.Dropdown(
                label="Box support mode",
                choices=["none", "intersect", "union"],
                value=MASK_PROCESSING_DEFAULTS["box_filter_support_mode"],
            )

        with gr.Row():
            box_filter_diff_threshold = gr.Slider(
                label="Box diff threshold",
                minimum=0.01,
                maximum=0.12,
                step=0.005,
                value=MASK_PROCESSING_DEFAULTS["box_filter_diff_threshold"],
            )

            box_filter_adaptive_c = gr.Slider(
                label="Box adaptive C",
                minimum=-8,
                maximum=1,
                step=0.1,
                value=MASK_PROCESSING_DEFAULTS["box_filter_adaptive_c"],
            )

        with gr.Row():
            box_filter_box_padding = gr.Slider(
                label="Box padding",
                minimum=0,
                maximum=32,
                step=1,
                value=MASK_PROCESSING_DEFAULTS["box_filter_box_padding"],
            )

            box_filter_min_component_area = gr.Number(
                label="Box min component area",
                value=MASK_PROCESSING_DEFAULTS["box_filter_min_component_area"],
                precision=0,
            )

            box_filter_dilate_iterations = gr.Slider(
                label="Box dilate iterations",
                minimum=0,
                maximum=12,
                step=1,
                value=MASK_PROCESSING_DEFAULTS["box_filter_dilate_iterations"],
            )

        with gr.Row():
            fusion_method = gr.Dropdown(
                label="Mask fusion method",
                choices=["union", "intersection", "vote", "weighted"],
                value=MASK_PROCESSING_DEFAULTS["fusion_method"],
            )

            fusion_min_votes = gr.Number(
                label="Fusion min votes",
                value=MASK_PROCESSING_DEFAULTS["fusion_min_votes"],
                precision=0,
            )

            fusion_weighted_threshold = gr.Slider(
                label="Fusion weighted threshold",
                minimum=0.05,
                maximum=0.95,
                step=0.01,
                value=MASK_PROCESSING_DEFAULTS["fusion_weighted_threshold"],
            )

        with gr.Row():
            cleanup_min_area = gr.Number(
                label="Cleanup min area",
                value=MASK_PROCESSING_DEFAULTS["cleanup_min_area"],
                precision=0,
            )

            cleanup_kernel_size = gr.Slider(
                label="Cleanup kernel size",
                minimum=1,
                maximum=15,
                step=2,
                value=MASK_PROCESSING_DEFAULTS["cleanup_kernel_size"],
            )

        with gr.Row():
            cleanup_dilate_iterations = gr.Slider(
                label="Cleanup dilate iterations",
                minimum=0,
                maximum=12,
                step=1,
                value=MASK_PROCESSING_DEFAULTS["cleanup_dilate_iterations"],
            )

            cleanup_dilate_kernel_size = gr.Slider(
                label="Cleanup dilate kernel",
                minimum=1,
                maximum=15,
                step=2,
                value=MASK_PROCESSING_DEFAULTS["cleanup_dilate_kernel_size"],
            )

            cleanup_fill_holes = gr.Checkbox(
                label="Fill mask holes",
                value=MASK_PROCESSING_DEFAULTS["cleanup_fill_holes"],
            )

    add_inputs(
        input_components,
        input_names,
        {
            "use_candidate_selection": use_candidate_selection,
            "candidate_min_area": candidate_min_area,
            "candidate_max_area_ratio": candidate_max_area_ratio,
            "include_detection_masks": include_detection_masks,
            "include_support_masks": include_support_masks,
            "fallback_to_detection_mask": fallback_to_detection_mask,
            "box_filter_method": box_filter_method,
            "box_filter_support_mode": box_filter_support_mode,
            "box_filter_diff_threshold": box_filter_diff_threshold,
            "box_filter_adaptive_c": box_filter_adaptive_c,
            "box_filter_box_padding": box_filter_box_padding,
            "box_filter_min_component_area": box_filter_min_component_area,
            "box_filter_dilate_iterations": box_filter_dilate_iterations,
            "fusion_method": fusion_method,
            "fusion_min_votes": fusion_min_votes,
            "fusion_weighted_threshold": fusion_weighted_threshold,
            "cleanup_min_area": cleanup_min_area,
            "cleanup_kernel_size": cleanup_kernel_size,
            "cleanup_dilate_iterations": cleanup_dilate_iterations,
            "cleanup_dilate_kernel_size": cleanup_dilate_kernel_size,
            "cleanup_fill_holes": cleanup_fill_holes,
        },
    )

    return MaskProcessingSettingsPanel(
        enabled=enabled,
        stages=stages,
        inputs=input_components,
        input_names=input_names,
    )


def build_mask_processing_tuning_dict(
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
