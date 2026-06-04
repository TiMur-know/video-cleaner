# ui/visual_app.py

from __future__ import annotations

from typing import Any

from core.ui_help_text import apply_ui_help_text
from ui.components.config_panel import build_config_tab
from ui.components.image_edit import build_image_tab
from ui.components.package_panel import build_packages_tab
from ui.components.video_edit import build_video_tab


def launch_visual_app(
    config_path: str | None = None,
    server_name: str = "127.0.0.1",
    server_port: int = 7860,
) -> None:
    """
    Launch clickable local browser UI.

    Install:
        pip install gradio
    """

    gr = lazy_import_gradio()
    apply_ui_help_text(gr)

    with gr.Blocks(title="watermwark") as app:
        gr.Markdown("# watermwark")
        gr.Markdown(
            "Choose image or video, select detector/inpainter/tracker options, "
            "preview frames, masks, intermediate outputs, and run the pipeline."
        )

        with gr.Tab("Image"):
            build_image_tab(gr, config_path)

        with gr.Tab("Video"):
            build_video_tab(gr, config_path)

        with gr.Tab("Packages"):
            build_packages_tab(gr)

        with gr.Tab("Config"):
            build_config_tab(gr, config_path)

    app.launch(
        server_name=server_name,
        server_port=server_port,
        show_error=True,
    )


def lazy_import_gradio() -> Any:
    try:
        import gradio as gr

        return gr

    except ImportError as exc:
        raise ImportError(
            "Gradio is required for visual UI. Install it with:\n"
            "    pip install gradio"
        ) from exc
