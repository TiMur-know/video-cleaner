# core/types.py

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from core.enums import InputType


BBox = tuple[int, int, int, int]


@dataclass(slots=True)
class MediaInfo:
    path: str
    input_type: InputType
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    frame_count: int | None = None
    duration_sec: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Detection:
    bbox: BBox
    confidence: float
    mask: np.ndarray
    label: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DetectorResult:
    mask: np.ndarray
    detections: list[Detection] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FrameData:
    index: int
    image: np.ndarray
    timestamp_sec: float = 0.0
    path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MaskData:
    mask: np.ndarray
    source: str
    confidence: float = 1.0
    frame_index: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PipelineContext:
    input_path: str
    output_path: str
    input_type: InputType
    frame_index: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PipelineResult:
    output_path: str | None = None
    image: np.ndarray | None = None
    frames: list[np.ndarray] | None = None
    mask: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def ensure_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()