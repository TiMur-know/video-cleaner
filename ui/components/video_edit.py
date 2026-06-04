# ui/components/video_edit.py

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
from ui.handlers.frame_handlers import (
    extract_video_frames_for_slider,
    reset_cancel_token,
    select_frame_from_store,
    stop_cancel_token,
)
from ui.handlers.video_handlers import run_video_from_ui


DEFAULT_VIDEO_PROPOSAL_DETECTORS = [
    "opencv",
    "fft",
    "anomaly",
]

DEFAULT_VIDEO_REFINER_DETECTORS: list[str] = []


def build_video_tab(gr: Any, config_path: str | None) -> None:
    default_selected_detectors = [
        *DEFAULT_VIDEO_PROPOSAL_DETECTORS,
        *DEFAULT_VIDEO_REFINER_DETECTORS,
    ]

    extraction_cancel_state = gr.State({"stop": False})
    pipeline_cancel_state = gr.State({"stop": False})

    frame_store_state = gr.State([])
    frame_labels_state = gr.State([])

    with gr.Row():
        video_input = gr.File(
            label="Video input",
            file_types=["video"],
        )

        output_path = gr.Textbox(
            label="Output video path",
            value="data/outputs/output.mp4",
        )

    with gr.Row():
        preset = gr.Dropdown(
            label="Preset",
            choices=["default", "fast", "balanced", "quality"],
            value="fast",
        )

        inpainter = gr.Dropdown(
            label="Inpainter",
            choices=["auto", "opencv", "lama", "sdxl", "flux"],
            value="opencv",
        )

        tracker = gr.Dropdown(
            label="Tracker",
            choices=["none", "optical_flow", "kalman", "xmem", "cotracker"],
            value="optical_flow",
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
        value=DEFAULT_VIDEO_PROPOSAL_DETECTORS,
    )

    refiner_detectors = gr.CheckboxGroup(
        label="Refiner detectors: improve masks from proposal results",
        choices=list(REFINER_DETECTOR_CHOICES),
        value=DEFAULT_VIDEO_REFINER_DETECTORS,
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

    audio_backend = gr.Dropdown(
        label="Audio backend",
        choices=["none", "moviepy", "ffmpeg"],
        value="moviepy",
    )

    gr.Markdown("## Frame explorer")

    with gr.Row():
        frame_step = gr.Number(
            label="Frame step",
            value=1,
            precision=0,
        )

        max_gallery_frames = gr.Number(
            label="Max frames, 0 = all",
            value=0,
            precision=0,
        )

    with gr.Row():
        extract_frames_button = gr.Button(
            "Extract frames",
            variant="secondary",
        )

        stop_extract_button = gr.Button(
            "Stop frame extraction",
            variant="stop",
        )

    frame_count_output = gr.Textbox(
        label="Frame count / status",
        value="No video loaded.",
    )

    video_metadata_output = gr.JSON(
        label="Video metadata",
    )

    frame_slider = gr.Slider(
        label="Frame number",
        minimum=0,
        maximum=1,
        step=1,
        value=0,
        interactive=True,
    )

    selected_frame_preview = gr.Image(
        label="Selected frame",
        type="numpy",
    )

    selected_frame_label = gr.Textbox(
        label="Selected frame info",
        value="No frame selected.",
    )

    extract_frames_button.click(
        fn=reset_cancel_token,
        inputs=[],
        outputs=[extraction_cancel_state],
    ).then(
        fn=lambda video_file, step_value, max_value, cancel_token: extract_video_frames_for_slider(
            video_file=video_file,
            frame_step=step_value,
            max_gallery_frames=max_value,
            cancel_token=cancel_token,
        ),
        inputs=[
            video_input,
            frame_step,
            max_gallery_frames,
            extraction_cancel_state,
        ],
        outputs=[
            frame_count_output,
            frame_store_state,
            frame_labels_state,
            frame_slider,
            selected_frame_preview,
            selected_frame_label,
            video_metadata_output,
        ],
    )

    stop_extract_button.click(
        fn=stop_cancel_token,
        inputs=[extraction_cancel_state],
        outputs=[extraction_cancel_state],
    ).then(
        fn=lambda: "Stopping frame extraction...",
        inputs=[],
        outputs=[frame_count_output],
    )

    frame_slider.change(
        fn=select_frame_from_store,
        inputs=[
            frame_slider,
            frame_store_state,
            frame_labels_state,
        ],
        outputs=[
            selected_frame_preview,
            selected_frame_label,
        ],
    )

    gr.Markdown("## Pipeline output")

    with gr.Row():
        run_button = gr.Button(
            "Run video pipeline",
            variant="primary",
        )

        stop_pipeline_button = gr.Button(
            "Stop pipeline",
            variant="stop",
        )

    with gr.Row():
        output_video = gr.Video(label="Output video")

        first_processed_frame = gr.Image(
            label="First processed frame",
            type="numpy",
        )

    metadata_output = gr.JSON(label="Pipeline metadata")

    pipeline_status_output = gr.Textbox(
        label="Pipeline status",
        value="Pipeline not started.",
    )

    run_button.click(
        fn=reset_cancel_token,
        inputs=[],
        outputs=[pipeline_cancel_state],
    ).then(
        fn=lambda video_file,
        out_path,
        preset_value,
        proposal_detector_values,
        refiner_detector_values,
        inpainter_value,
        tracker_value,
        device_value,
        audio_backend_value,
        sensitivity_value,
        mask_expand_value,
        min_area_value,
        fusion_strictness_value,
        preprocessing_module_values,
        postprocessing_module_values,
        cancel_token,
        *detector_setting_values: run_video_from_ui(
            config_path=config_path,
            video_file=video_file,
            output_path=out_path,
            preset=preset_value,
            proposal_detectors=proposal_detector_values,
            refiner_detectors=refiner_detector_values,
            inpainter=inpainter_value,
            tracker=tracker_value,
            device=device_value,
            audio_backend=audio_backend_value,
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
            cancel_token=cancel_token,
        ),
        inputs=[
            video_input,
            output_path,
            preset,
            proposal_detectors,
            refiner_detectors,
            inpainter,
            tracker,
            device,
            audio_backend,
            detection_sensitivity,
            mask_expand,
            min_area,
            fusion_strictness,
            processing_settings.preprocessing_modules,
            processing_settings.postprocessing_modules,
            pipeline_cancel_state,
            *detector_settings.inputs,
            *processing_settings.inputs,
        ],
        outputs=[
            output_video,
            first_processed_frame,
            metadata_output,
            pipeline_status_output,
        ],
    )

    stop_pipeline_button.click(
        fn=stop_cancel_token,
        inputs=[pipeline_cancel_state],
        outputs=[pipeline_cancel_state],
    ).then(
        fn=lambda: (
            "Stop requested. Pipeline will stop when the current stage checks "
            "cancel_token."
        ),
        inputs=[],
        outputs=[pipeline_status_output],
    )
