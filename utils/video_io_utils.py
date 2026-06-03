# utils/video_io_utils.py

from __future__ import annotations

import importlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Literal

import numpy as np


AudioBackend = Literal["none", "ffmpeg", "moviepy"]


_CACHE: dict[str, Any] = {}


def lazy_import(name: str) -> Any:
    if name not in _CACHE:
        try:
            _CACHE[name] = importlib.import_module(name)
        except ImportError as exc:
            raise ImportError(f"Missing optional dependency '{name}'.") from exc

    return _CACHE[name]


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def read_frame(frame: np.ndarray | str) -> np.ndarray:
    if isinstance(frame, np.ndarray):
        return frame

    cv2 = lazy_import("cv2")

    frame_path = Path(frame)

    if not frame_path.exists():
        raise FileNotFoundError(f"Frame not found: {frame_path}")

    image = cv2.imread(str(frame_path))

    if image is None:
        raise RuntimeError(f"Could not read frame: {frame_path}")

    return image


def save_frame(
    frame: np.ndarray,
    path: str | Path,
    jpeg_quality: int = 95,
    png_compression: int = 3,
) -> str:
    cv2 = lazy_import("cv2")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ext = path.suffix.lower().lstrip(".")
    params: list[int] = []

    if ext in {"jpg", "jpeg"}:
        params = [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]

    elif ext == "png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, int(png_compression)]

    if not cv2.imwrite(str(path), frame, params):
        raise RuntimeError(f"Failed to save frame: {path}")

    return str(path)


def prepare_video_frame(
    frame: np.ndarray,
    width: int,
    height: int,
    resize: bool = True,
) -> np.ndarray:
    cv2 = lazy_import("cv2")

    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    if frame.ndim == 3 and frame.shape[-1] == 4:
        frame = frame[..., :3]

    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(f"Unsupported frame shape: {frame.shape}")

    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    h, w = frame.shape[:2]

    if (w, h) != (width, height):
        if not resize:
            raise ValueError(
                f"Frame size {(w, h)} does not match output size {(width, height)}"
            )

        frame = cv2.resize(
            frame,
            dsize=(width, height),
            interpolation=cv2.INTER_LINEAR,
        )

    return frame


def get_fps_from_context(context: dict[str, Any] | None) -> float | None:
    if not context:
        return None

    if "fps" in context:
        return float(context["fps"])

    metadata = context.get("metadata")

    if isinstance(metadata, dict) and "fps" in metadata:
        return float(metadata["fps"])

    return None


def mux_audio(
    silent_video_path: str | Path,
    source_video_path: str | Path,
    final_video_path: str | Path,
    backend: AudioBackend = "moviepy",
    cleanup: bool = True,
) -> Path:
    """
    Copy/mux audio from source video into rebuilt video.

    backend:
        none:
            Do not copy audio. Returns silent_video_path.

        ffmpeg:
            Uses command-line ffmpeg.

        moviepy:
            Uses MoviePy API.
    """

    if backend == "none":
        return Path(silent_video_path)

    if backend == "ffmpeg":
        return mux_audio_ffmpeg(
            silent_video_path=silent_video_path,
            source_video_path=source_video_path,
            final_video_path=final_video_path,
            cleanup=cleanup,
        )

    if backend == "moviepy":
        return mux_audio_moviepy(
            silent_video_path=silent_video_path,
            source_video_path=source_video_path,
            final_video_path=final_video_path,
            cleanup=cleanup,
        )

    raise ValueError(f"Unsupported audio backend: {backend}")


def mux_audio_ffmpeg(
    silent_video_path: str | Path,
    source_video_path: str | Path,
    final_video_path: str | Path,
    cleanup: bool = True,
) -> Path:
    silent_video_path = Path(silent_video_path)
    source_video_path = Path(source_video_path)
    final_video_path = Path(final_video_path)

    if not source_video_path.exists():
        raise FileNotFoundError(f"Source video not found: {source_video_path}")

    ffmpeg = shutil.which("ffmpeg")

    if ffmpeg is None:
        raise RuntimeError("ffmpeg was not found.")

    final_video_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        ffmpeg,
        "-y",
        "-i",
        str(silent_video_path),
        "-i",
        str(source_video_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(final_video_path),
    ]

    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio mux failed:\n{result.stderr}")

    if cleanup and silent_video_path.exists():
        silent_video_path.unlink()

    return final_video_path


def mux_audio_moviepy(
    silent_video_path: str | Path,
    source_video_path: str | Path,
    final_video_path: str | Path,
    cleanup: bool = True,
) -> Path:
    """
    Copy audio using MoviePy.

    Install:
        pip install moviepy

    Note:
        MoviePy may still rely on video codecs available in your environment,
        but your project code no longer calls ffmpeg directly.
    """

    silent_video_path = Path(silent_video_path)
    source_video_path = Path(source_video_path)
    final_video_path = Path(final_video_path)

    if not silent_video_path.exists():
        raise FileNotFoundError(f"Silent video not found: {silent_video_path}")

    if not source_video_path.exists():
        raise FileNotFoundError(f"Source video not found: {source_video_path}")

    final_video_path.parent.mkdir(parents=True, exist_ok=True)

    VideoFileClip = import_moviepy_video_file_clip()

    source_clip = None
    silent_clip = None
    final_clip = None

    try:
        source_clip = VideoFileClip(str(source_video_path))
        silent_clip = VideoFileClip(str(silent_video_path))

        if source_clip.audio is None:
            # No audio to copy. Just move/copy the silent video path.
            if silent_video_path != final_video_path:
                final_video_path.write_bytes(silent_video_path.read_bytes())

            return final_video_path

        if hasattr(silent_clip, "with_audio"):
            final_clip = silent_clip.with_audio(source_clip.audio)
        else:
            final_clip = silent_clip.set_audio(source_clip.audio)

        final_clip.write_videofile(
            str(final_video_path),
            codec="libx264",
            audio_codec="aac",
            temp_audiofile=str(final_video_path.with_suffix(".temp-audio.m4a")),
            remove_temp=True,
            logger=None,
        )

    finally:
        for clip in [final_clip, silent_clip, source_clip]:
            if clip is not None and hasattr(clip, "close"):
                clip.close()

    if cleanup and silent_video_path.exists():
        silent_video_path.unlink()

    return final_video_path


def import_moviepy_video_file_clip() -> Any:
    """
    Supports both common MoviePy import styles.
    """

    try:
        moviepy = lazy_import("moviepy")
        if hasattr(moviepy, "VideoFileClip"):
            return moviepy.VideoFileClip
    except ImportError:
        pass

    try:
        moviepy_editor = lazy_import("moviepy.editor")
        if hasattr(moviepy_editor, "VideoFileClip"):
            return moviepy_editor.VideoFileClip
    except ImportError as exc:
        raise ImportError(
            "MoviePy is required for audio_backend='moviepy'. "
            "Install it with: pip install moviepy"
        ) from exc

    raise ImportError(
        "Could not import MoviePy VideoFileClip. "
        "Try: pip install moviepy"
    )