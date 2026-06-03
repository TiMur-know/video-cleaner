# ui/components/package_panel.py

from __future__ import annotations

from typing import Any

from ui.handlers.package_handlers import (
    check_visual_packages_text,
    install_visual_packages_text,
)


def build_packages_tab(gr: Any) -> None:
    gr.Markdown("## Package manager")
    gr.Markdown(
        "Check or install packages needed for the visual UI. "
        "Model-specific packages may still be requested later depending on "
        "your selected inpainter, OCR, tracker, or device."
    )

    with gr.Row():
        check_button = gr.Button(
            "Check visual UI packages",
            variant="secondary",
        )

        install_button = gr.Button(
            "Install missing visual UI packages",
            variant="primary",
        )

    output = gr.Textbox(
        label="Package status",
        lines=14,
    )

    check_button.click(
        fn=check_visual_packages_text,
        inputs=[],
        outputs=[output],
    )

    install_button.click(
        fn=install_visual_packages_text,
        inputs=[],
        outputs=[output],
    )