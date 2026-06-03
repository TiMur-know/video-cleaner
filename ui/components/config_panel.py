# ui/components/config_panel.py

from __future__ import annotations

from typing import Any

from core.config import load_config, to_dict


def build_config_tab(gr: Any, config_path: str | None) -> None:
    gr.Markdown("## Config preview")
    gr.Markdown(
        "Shows the resolved config loaded from config.json or defaults."
    )

    refresh_button = gr.Button(
        "Show resolved config",
        variant="secondary",
    )

    config_output = gr.JSON(label="Resolved config")

    refresh_button.click(
        fn=lambda: to_dict(load_config(config_path)),
        inputs=[],
        outputs=[config_output],
    )