# controls/tracking_control.py

from __future__ import annotations

from controls.enums import Tracker
from core.config import AppConfig


def set_tracker(config: AppConfig, name: Tracker | str) -> None:
    if isinstance(name, str):
        try:
            name = Tracker(name)
        except Exception as exc:
            raise ValueError(f"Unsupported tracker: {name}") from exc

    config.video.tracking.enabled = name != Tracker.NONE
    config.video.tracking.mode = name.value


def set_device_for_tracking(
    config: AppConfig,
    device_value: str,
) -> None:
    config.video.tracking.xmem.device = device_value
    config.video.tracking.cotracker.device = device_value