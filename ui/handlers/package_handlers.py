# ui/handlers/package_handlers.py

from __future__ import annotations

from utils.package_manager import (
    format_missing_packages,
    format_installed_model_names,
    format_installed_package_names,
    format_model_size_summary,
    full_environment_status,
    install_models_by_name,
    install_missing_for_visual_ui,
    install_packages_by_name,
    model_choices,
    missing_for_visual_ui,
    package_choices,
    uninstall_models_by_name,
    uninstall_packages_by_name,
)


def check_visual_packages_text() -> str:
    """
    Return visual UI package status as text for Gradio textbox.
    """

    missing = missing_for_visual_ui()

    if not missing:
        return "All visual UI packages are installed."

    return "Missing packages:\n" + format_missing_packages(missing)


def install_visual_packages_text() -> str:
    """
    Install missing packages used by the visual UI.
    """

    return install_missing_for_visual_ui()


def package_action_choices() -> list[str]:
    return package_choices()


def model_action_choices() -> list[str]:
    return model_choices()


def check_environment_status() -> dict[str, object]:
    return full_environment_status()


def installed_packages_text() -> str:
    return format_installed_package_names()


def installed_models_text() -> str:
    return format_installed_model_names()


def model_size_summary_text() -> str:
    return format_model_size_summary()


def refresh_package_window_status() -> tuple[dict[str, object], str, str, str]:
    return (
        check_environment_status(),
        installed_packages_text(),
        installed_models_text(),
        model_size_summary_text(),
    )


def install_selected_packages_text(package_names: list[str] | None) -> str:
    return install_packages_by_name(package_names)


def uninstall_selected_packages_text(package_names: list[str] | None) -> str:
    return uninstall_packages_by_name(package_names)


def install_selected_models_text(model_names: list[str] | None) -> str:
    return install_models_by_name(model_names)


def uninstall_selected_models_text(model_names: list[str] | None) -> str:
    return uninstall_models_by_name(model_names)
