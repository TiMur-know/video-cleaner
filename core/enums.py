# core/enums.py

from __future__ import annotations

from enum import Enum


class InputType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"


class PipelineType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    DETECTION = "detection"
    INPAINTING = "inpainting"
    PREPROCESSING = "preprocessing"
    POSTPROCESSING = "postprocessing"


class DeviceType(str, Enum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


class ColorOrder(str, Enum):
    BGR = "bgr"
    RGB = "rgb"


class MaskFormat(str, Enum):
    UINT8 = "uint8"
    BOOL = "bool"
    FLOAT = "float"


class DetectorName(str, Enum):
    YOLO = "yolo"
    SAM2 = "sam2"
    PADDLE_OCR = "paddle_ocr"
    EASY_OCR = "easy_ocr"
    FFT = "fft"
    ANOMALY = "anomaly"
    FUSION = "fusion"


class TrackerName(str, Enum):
    OPTICAL_FLOW = "optical_flow"
    KALMAN = "kalman"
    COTRACKER = "cotracker"
    XMEM = "xmem"


class InpainterName(str, Enum):
    OPENCV = "opencv"
    LAMA = "lama"
    FLUX = "flux"
    SDXL = "sdxl"


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"