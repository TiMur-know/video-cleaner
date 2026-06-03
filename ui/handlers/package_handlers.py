# ui/handlers/package_handlers.py

from __future__ import annotations

from utils.package_manager import (
    format_missing_packages,
    install_missing_for_visual_ui,
    missing_for_visual_ui,
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