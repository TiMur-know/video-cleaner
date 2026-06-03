# core/frame_extract.py

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from utils.video_io_utils import ensure_dir, lazy_import, save_frame, save_json


@dataclass(slots=True)
class ExtractedFrame:
    index: int
    timestamp_sec: float
    image: np.ndarray | None = None
    path: str | None = None


@dataclass(slots=True)
class FrameExtractionResult:
    frames: list[ExtractedFrame]
    fps: float
    width: int
    height: int
    total_frames: int
    extracted_count: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FrameExtractionConfig:
    enabled: bool = True
    output_dir: str = "data/temp/frames"

    save_frames: bool = True
    keep_in_memory: bool = False

    image_extension: str = "png"
    every_nth_frame: int = 1

    start_frame: int = 0
    end_frame: int | None = None
    max_frames: int | None = None

    jpeg_quality: int = 95
    png_compression: int = 3

    save_metadata: bool = True


class FrameExtractor:
    name = "frame_extractor"

    def __init__(self, config: FrameExtractionConfig | None = None) -> None:
        self.config = config or FrameExtractionConfig()

    def extract(
        self,
        video_path: str,
        context: dict[str, Any] | None = None,
    ) -> FrameExtractionResult:
        if not self.config.enabled:
            return FrameExtractionResult([], 0.0, 0, 0, 0, 0, {"enabled": False})

        cv2 = lazy_import("cv2")
        video_path_obj = Path(video_path)

        if not video_path_obj.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        if self.config.every_nth_frame <= 0:
            raise ValueError("every_nth_frame must be greater than 0")

        output_dir = ensure_dir(self.config.output_dir)

        cap = cv2.VideoCapture(str(video_path_obj))

        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frames: list[ExtractedFrame] = []
        frame_index = self.config.start_frame
        extracted = 0

        if self.config.start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.config.start_frame)

        try:
            while True:
                if self._should_stop(frame_index, extracted):
                    break

                ok, frame = cap.read()

                if not ok:
                    break

                if self._should_extract(frame_index):
                    frames.append(
                        self._make_frame_record(
                            frame=frame,
                            frame_index=frame_index,
                            fps=fps,
                            output_dir=output_dir,
                        )
                    )
                    extracted += 1

                frame_index += 1

        finally:
            cap.release()

        metadata = {
            "source_video": str(video_path_obj),
            "fps": fps,
            "width": width,
            "height": height,
            "total_frames": total_frames,
            "extracted_count": extracted,
            "every_nth_frame": self.config.every_nth_frame,
            "start_frame": self.config.start_frame,
            "end_frame": self.config.end_frame,
            "save_frames": self.config.save_frames,
            "keep_in_memory": self.config.keep_in_memory,
            "output_dir": str(output_dir),
        }

        if self.config.save_metadata:
            save_json(output_dir / "metadata.json", metadata)

        return FrameExtractionResult(
            frames=frames,
            fps=fps,
            width=width,
            height=height,
            total_frames=total_frames,
            extracted_count=extracted,
            metadata=metadata,
        )

    def _should_stop(self, frame_index: int, extracted: int) -> bool:
        if self.config.end_frame is not None and frame_index > self.config.end_frame:
            return True

        if self.config.max_frames is not None and extracted >= self.config.max_frames:
            return True

        return False

    def _should_extract(self, frame_index: int) -> bool:
        return (
            frame_index - self.config.start_frame
        ) % self.config.every_nth_frame == 0

    def _make_frame_record(
        self,
        frame: np.ndarray,
        frame_index: int,
        fps: float,
        output_dir: Path,
    ) -> ExtractedFrame:
        path = None

        if self.config.save_frames:
            ext = self.config.image_extension.lower().lstrip(".")
            path = save_frame(
                frame,
                output_dir / f"frame_{frame_index:08d}.{ext}",
                jpeg_quality=self.config.jpeg_quality,
                png_compression=self.config.png_compression,
            )

        return ExtractedFrame(
            index=frame_index,
            timestamp_sec=frame_index / fps if fps > 0 else 0.0,
            image=frame if self.config.keep_in_memory else None,
            path=path,
        )


def extract_frames(
    video_path: str,
    output_dir: str = "data/temp/frames",
    keep_in_memory: bool = False,
) -> FrameExtractionResult:
    return FrameExtractor(
        FrameExtractionConfig(
            output_dir=output_dir,
            keep_in_memory=keep_in_memory,
        )
    ).extract(video_path)