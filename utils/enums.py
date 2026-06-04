# utils/enums.py

from __future__ import annotations

from enum import StrEnum


class AppEnum(StrEnum):
    @classmethod
    def values(cls) -> list[str]:
        return [item.value for item in cls]


class Preset(AppEnum):
    DEFAULT = "default"
    FAST = "fast"
    BALANCED = "balanced"
    QUALITY = "quality"


class Inpainter(AppEnum):
    AUTO = "auto"
    OPENCV = "opencv"
    LAMA = "lama"
    STABLE_DIFFUSION = "stable_diffusion"
    SDXL = "sdxl"
    FLUX = "flux"


class Tracker(AppEnum):
    NONE = "none"
    OPTICAL_FLOW = "optical_flow"
    KALMAN = "kalman"
    XMEM = "xmem"
    COTRACKER = "cotracker"


class OCR(AppEnum):
    NONE = "none"
    PADDLE = "paddle"
    EASY = "easy"
    BOTH = "both"


class Device(AppEnum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


DeviceType = Device


class AudioBackend(AppEnum):
    NONE = "none"
    MOVIEPY = "moviepy"
    FFMPEG = "ffmpeg"


class InputType(AppEnum):
    IMAGE = "image"
    VIDEO = "video"


class PipelineType(AppEnum):
    IMAGE = "image"
    VIDEO = "video"
    DETECTION = "detection"
    INPAINTING = "inpainting"
    PREPROCESSING = "preprocessing"
    POSTPROCESSING = "postprocessing"


class ColorOrder(AppEnum):
    BGR = "bgr"
    RGB = "rgb"


class MaskFormat(AppEnum):
    UINT8 = "uint8"
    BOOL = "bool"
    FLOAT = "float"


class DetectorName(AppEnum):
    OPENCV = "opencv"
    GROUNDING_DINO = "grounding_dino"
    YOLO = "yolo"
    SAM2 = "sam2"
    MOBILE_SAM_2 = "mobile_sam_2"
    PADDLE_OCR = "paddle_ocr"
    EASY_OCR = "easy_ocr"
    FFT = "fft"
    ANOMALY = "anomaly"
    FUSION = "fusion"


class TrackerName(AppEnum):
    OPTICAL_FLOW = "optical_flow"
    KALMAN = "kalman"
    COTRACKER = "cotracker"
    XMEM = "xmem"


class InpainterName(AppEnum):
    OPENCV = "opencv"
    LAMA = "lama"
    STABLE_DIFFUSION = "stable_diffusion"
    FLUX = "flux"
    SDXL = "sdxl"


class LogLevel(AppEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


__all__ = [
    "AudioBackend",
    "AppEnum",
    "ColorOrder",
    "DetectorName",
    "Device",
    "DeviceType",
    "Inpainter",
    "InpainterName",
    "InputType",
    "LogLevel",
    "MaskFormat",
    "OCR",
    "PipelineType",
    "Preset",
    "Tracker",
    "TrackerName",
]
