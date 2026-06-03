# controls/enums.py

from __future__ import annotations

from enum import StrEnum


class Preset(StrEnum):
    DEFAULT = "default"
    FAST = "fast"
    BALANCED = "balanced"
    QUALITY = "quality"


class Inpainter(StrEnum):
    AUTO = "auto"
    OPENCV = "opencv"
    LAMA = "lama"
    SDXL = "sdxl"
    FLUX = "flux"


class Tracker(StrEnum):
    NONE = "none"
    OPTICAL_FLOW = "optical_flow"
    KALMAN = "kalman"
    XMEM = "xmem"
    COTRACKER = "cotracker"


class OCR(StrEnum):
    NONE = "none"
    PADDLE = "paddle"
    EASY = "easy"
    BOTH = "both"


class Device(StrEnum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


class AudioBackend(StrEnum):
    NONE = "none"
    MOVIEPY = "moviepy"
    FFMPEG = "ffmpeg"