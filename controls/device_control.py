# controls/device_control.py

from __future__ import annotations

from controls.detection_control import set_device_for_detectors
from controls.enums import Device
from controls.inpainting_control import set_device_for_inpainting
from controls.tracking_control import set_device_for_tracking
from core.config import AppConfig


def set_device(config: AppConfig, device: Device | str) -> None:
    if isinstance(device, str):
        try:
            device = Device(device)
        except Exception as exc:
            raise ValueError(f"Unsupported device: {device}") from exc

    device_value = device.value
    is_cuda = device == Device.CUDA

    set_device_for_detectors(
        config=config,
        device_value=device_value,
        is_cuda=is_cuda,
    )

    set_device_for_tracking(
        config=config,
        device_value=device_value,
    )

    set_device_for_inpainting(
        config=config,
        device_value=device_value,
    )