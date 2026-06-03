# ui/handlers/frame_handlers.py

from __future__ import annotations

from typing import Any

import numpy as np

from ui.components.common import (
    get_file_path,
    to_preview_rgb,
)
from utils.video_io_utils import lazy_import


def reset_cancel_token() -> dict[str, bool]:
    """
    Reset cancel state before starting a task.
    """

    return {"stop": False}


def stop_cancel_token(
    cancel_token: dict[str, bool] | None,
) -> dict[str, bool]:
    """
    Set cancel flag.

    Gradio state passes dictionaries between events,
    so always return the updated token.
    """

    if cancel_token is None:
        cancel_token = {}

    cancel_token["stop"] = True

    return cancel_token


def is_cancelled(
    cancel_token: dict[str, bool] | None,
) -> bool:
    if not cancel_token:
        return False

    return bool(cancel_token.get("stop", False))


def extract_video_frames_for_slider(
    video_file: Any,
    frame_step: int | float | None = 1,
    max_gallery_frames: int | float | None = 0,
    cancel_token: dict[str, bool] | None = None,
) -> tuple[
    str,
    list[np.ndarray],
    list[str],
    Any,
    np.ndarray | None,
    str,
    dict[str, Any],
]:
    """
    Extract video frames for slider preview.

    frame_step:
        1 = every frame
        5 = every 5th frame

    max_gallery_frames:
        0 = no limit
        N = stop after N extracted frames
    """

    gr = lazy_import_gradio()

    if video_file is None:
        raise ValueError("Choose a video file first.")

    video_path = get_file_path(video_file)

    step = int(frame_step or 1)
    step = max(1, step)

    limit = int(max_gallery_frames or 0)
    limit = max(0, limit)

    cv2 = lazy_import("cv2")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    frame_store: list[np.ndarray] = []
    frame_labels: list[str] = []

    frame_index = 0
    extracted_count = 0
    stopped = False

    try:
        while True:
            if is_cancelled(cancel_token):
                stopped = True
                break

            success, frame = cap.read()

            if not success:
                break

            if frame_index % step == 0:
                preview = to_preview_rgb(frame)

                if preview is not None:
                    frame_store.append(preview)
                    frame_labels.append(f"Frame {frame_index}")
                    extracted_count += 1

                if limit > 0 and extracted_count >= limit:
                    break

            frame_index += 1

    finally:
        cap.release()

    metadata = {
        "video_path": video_path,
        "total_frames": total_frames,
        "extracted_frames": extracted_count,
        "frame_step": step,
        "gallery_limit": limit,
        "fps": fps,
        "width": width,
        "height": height,
        "duration_seconds": total_frames / fps if fps > 0 else None,
        "stopped": stopped,
    }

    if stopped:
        status = (
            f"Stopped. Total frames: {total_frames} | "
            f"Extracted: {extracted_count} | "
            f"FPS: {fps:.3f} | "
            f"Size: {width}x{height}"
        )
    else:
        status = (
            f"Done. Total frames: {total_frames} | "
            f"Extracted: {extracted_count} | "
            f"FPS: {fps:.3f} | "
            f"Size: {width}x{height}"
        )

    if not frame_store:
        return (
            status,
            frame_store,
            frame_labels,
            gr.update(
                minimum=0,
                maximum=1,
                step=1,
                value=0,
                interactive=False,
            ),
            None,
            "No frame extracted.",
            metadata,
        )

    return (
        status,
        frame_store,
        frame_labels,
        gr.update(
            minimum=0,
            maximum=max(1, len(frame_store) - 1),
            step=1,
            value=0,
            interactive=True,
        ),
        frame_store[0],
        frame_labels[0],
        metadata,
    )


def select_frame_from_store(
    slider_index: int | float,
    frame_store: list[np.ndarray] | None,
    frame_labels: list[str] | None,
) -> tuple[np.ndarray | None, str]:
    """
    Show selected frame from slider index.
    """

    if not frame_store:
        return None, "No frames extracted."

    index = int(slider_index)
    index = max(0, min(index, len(frame_store) - 1))

    label = f"Frame {index}"

    if frame_labels and index < len(frame_labels):
        label = frame_labels[index]

    return frame_store[index], f"{label} | Slider index {index}"


def lazy_import_gradio() -> Any:
    try:
        import gradio as gr

        return gr

    except ImportError as exc:
        raise ImportError(
            "Gradio is required for visual UI. Install it with:\n"
            "    pip install gradio"
        ) from exc