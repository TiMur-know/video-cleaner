# controls/audio_control.py

from __future__ import annotations

from utils.enums import AudioBackend
from core.config import AppConfig


def set_audio_backend(
    config: AppConfig,
    backend: AudioBackend | str,
) -> None:
    if isinstance(backend, str):
        try:
            backend = AudioBackend(backend)
        except Exception as exc:
            raise ValueError(f"Unsupported audio backend: {backend}") from exc

    config.video.rebuild.audio_backend = backend.value
    config.video.rebuild.copy_audio = backend != AudioBackend.NONE
