# utils/video_rebuilder.py

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from utils.video_io_utils import (
    ensure_dir,
    get_fps_from_context,
    lazy_import,
    mux_audio,
    prepare_video_frame,
    read_frame,
)


AudioBackend = Literal["none", "ffmpeg", "moviepy"]


@dataclass(slots=True)
class VideoRebuildConfig:
    enabled: bool = True

    output_path: str = "data/outputs/output.mp4"
    fps: float | None = None

    width: int | None = None
    height: int | None = None

    codec: str = "mp4v"
    resize_to_output: bool = True

    copy_audio: bool = False
    audio_backend: AudioBackend = "moviepy"
    source_video_path: str | None = None

    temp_video_path: str = "data/temp/rebuilt_no_audio.mp4"
    cleanup_temp: bool = True


@dataclass(slots=True)
class VideoRebuildResult:
    output_path: str
    frame_count: int
    fps: float
    width: int
    height: int
    metadata: dict[str, Any] = field(default_factory=dict)


class VideoRebuilder:
    name = "video_rebuilder"

    def __init__(self, config: VideoRebuildConfig | None = None) -> None:
        self.config = config or VideoRebuildConfig()

    def rebuild(
        self,
        frames: list[np.ndarray] | list[str],
        context: dict[str, Any] | None = None,
    ) -> VideoRebuildResult:
        if not self.config.enabled:
            raise RuntimeError("VideoRebuilder is disabled.")

        if not frames:
            raise ValueError("No frames provided for video rebuild.")

        first_frame = read_frame(frames[0])

        width = self.config.width or first_frame.shape[1]
        height = self.config.height or first_frame.shape[0]
        fps = self.config.fps or get_fps_from_context(context)

        if fps is None or fps <= 0:
            raise ValueError("fps must be provided in config or context.")

        final_path = Path(self.config.output_path)

        silent_path = (
            Path(self.config.temp_video_path)
            if self.config.copy_audio and self.config.audio_backend != "none"
            else final_path
        )

        ensure_dir(silent_path.parent)

        writer = self._create_writer(
            path=silent_path,
            fps=fps,
            width=width,
            height=height,
        )

        frame_count = 0

        try:
            for item in frames:
                frame = prepare_video_frame(
                    read_frame(item),
                    width=width,
                    height=height,
                    resize=self.config.resize_to_output,
                )

                writer.write(frame)
                frame_count += 1

        finally:
            writer.release()

        output_path = silent_path

        if self.config.copy_audio and self.config.audio_backend != "none":
            if self.config.source_video_path is None:
                raise ValueError("source_video_path is required when copy_audio=True.")

            output_path = mux_audio(
                silent_video_path=silent_path,
                source_video_path=self.config.source_video_path,
                final_video_path=final_path,
                backend=self.config.audio_backend,
                cleanup=self.config.cleanup_temp,
            )

        metadata = {
            "output_path": str(output_path),
            "frame_count": frame_count,
            "fps": fps,
            "width": width,
            "height": height,
            "codec": self.config.codec,
            "copy_audio": self.config.copy_audio,
            "audio_backend": self.config.audio_backend,
        }

        return VideoRebuildResult(
            output_path=str(output_path),
            frame_count=frame_count,
            fps=float(fps),
            width=width,
            height=height,
            metadata=metadata,
        )

    def _create_writer(
        self,
        path: Path,
        fps: float,
        width: int,
        height: int,
    ) -> Any:
        cv2 = lazy_import("cv2")

        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*self.config.codec),
            fps,
            (width, height),
        )

        if not writer.isOpened():
            raise RuntimeError(f"Could not create video writer for: {path}")

        return writer


def rebuild_video(
    frames: list[np.ndarray] | list[str],
    output_path: str,
    fps: float,
) -> VideoRebuildResult:
    return VideoRebuilder(
        VideoRebuildConfig(
            output_path=output_path,
            fps=fps,
        )
    ).rebuild(frames)