# ui/components/package_panel.py

from __future__ import annotations

from typing import Any

from ui.handlers.package_handlers import (
    check_environment_status,
    installed_models_text,
    installed_packages_text,
    install_selected_models_text,
    install_selected_packages_text,
    model_size_summary_text,
    model_action_choices,
    package_action_choices,
    refresh_package_window_status,
    uninstall_selected_models_text,
    uninstall_selected_packages_text,
)


def build_packages_tab(gr: Any) -> None:
    gr.Markdown("## Package manager")
    gr.Markdown(
        "Check installed runtime packages and model files, then install or "
        "uninstall selected items."
    )

    refresh_button = gr.Button(
        "Refresh package and model status",
        variant="secondary",
    )

    status_output = gr.JSON(
        label="Package and model status",
        value=check_environment_status(),
    )

    with gr.Row():
        installed_packages_output = gr.Textbox(
            label="Installed packages",
            value=installed_packages_text(),
            lines=10,
            interactive=False,
        )

        installed_models_output = gr.Textbox(
            label="Installed models and checkpoints",
            value=installed_models_text(),
            lines=10,
            interactive=False,
        )

    model_size_output = gr.Textbox(
        label="Model size estimates",
        value=model_size_summary_text(),
        lines=10,
        interactive=False,
    )

    package_selection = gr.CheckboxGroup(
        label="Runtime packages",
        choices=package_action_choices(),
        value=[],
    )

    with gr.Row():
        install_packages_button = gr.Button(
            "Install selected packages",
            variant="primary",
        )

        uninstall_packages_button = gr.Button(
            "Uninstall selected packages",
            variant="stop",
        )

    model_selection = gr.CheckboxGroup(
        label="Models and checkpoints",
        choices=model_action_choices(),
        value=[],
    )

    with gr.Row():
        install_models_button = gr.Button(
            "Install selected models",
            variant="primary",
        )

        uninstall_models_button = gr.Button(
            "Uninstall selected models",
            variant="stop",
        )

    action_output = gr.Textbox(
        label="Package action output",
        lines=14,
    )

    refresh_button.click(
        fn=refresh_package_window_status,
        inputs=[],
        outputs=[
            status_output,
            installed_packages_output,
            installed_models_output,
            model_size_output,
        ],
    )

    install_packages_button.click(
        fn=install_selected_packages_text,
        inputs=[package_selection],
        outputs=[action_output],
    ).then(
        fn=refresh_package_window_status,
        inputs=[],
        outputs=[
            status_output,
            installed_packages_output,
            installed_models_output,
            model_size_output,
        ],
    )

    uninstall_packages_button.click(
        fn=uninstall_selected_packages_text,
        inputs=[package_selection],
        outputs=[action_output],
    ).then(
        fn=refresh_package_window_status,
        inputs=[],
        outputs=[
            status_output,
            installed_packages_output,
            installed_models_output,
            model_size_output,
        ],
    )

    install_models_button.click(
        fn=install_selected_models_text,
        inputs=[model_selection],
        outputs=[action_output],
    ).then(
        fn=refresh_package_window_status,
        inputs=[],
        outputs=[
            status_output,
            installed_packages_output,
            installed_models_output,
            model_size_output,
        ],
    )

    uninstall_models_button.click(
        fn=uninstall_selected_models_text,
        inputs=[model_selection],
        outputs=[action_output],
    ).then(
        fn=refresh_package_window_status,
        inputs=[],
        outputs=[
            status_output,
            installed_packages_output,
            installed_models_output,
            model_size_output,
        ],
    )
