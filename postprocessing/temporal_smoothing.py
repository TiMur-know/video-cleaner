# postprocessing/temporal_smoothing.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from postprocessing.utils import blend_masked, validate_frames


TemporalSmoothingMode = Literal["ema", "window_average"]


@dataclass(slots=True)
class TemporalSmoothingConfig:
    enabled: bool = True

    mode: TemporalSmoothingMode = "ema"

    # EMA value.
    alpha: float = 0.7

    # Used for window average.
    window_size: int = 3

    # If True and masks are provided, smooth only repaired regions.
    masked_only: bool = True


class TemporalSmoothingPostprocessor:
    """
    Smooth frame-to-frame flicker after video inpainting.

    Input:
        frames:
            list of repaired frames

        masks:
            optional list of repaired-region masks

    Output:
        temporally-smoothed frames
    """

    name = "temporal_smoothing"

    def __init__(self, config: TemporalSmoothingConfig | None = None) -> None:
        self.config = config or TemporalSmoothingConfig()

    def __call__(
        self,
        frames: list[np.ndarray],
        masks: list[np.ndarray] | None = None,
        context: dict[str, Any] | None = None,
    ) -> list[np.ndarray]:
        return self.apply(frames, masks=masks, context=context)

    def apply(
        self,
        frames: list[np.ndarray],
        masks: list[np.ndarray] | None = None,
        context: dict[str, Any] | None = None,
    ) -> list[np.ndarray]:
        if not self.config.enabled:
            return frames

        if not frames:
            return []

        validate_frames(frames, self.name)

        if masks is not None and len(masks) != len(frames):
            raise ValueError("masks length must match frames length")

        if self.config.mode == "ema":
            return self._ema(frames, masks)

        if self.config.mode == "window_average":
            return self._window_average(frames, masks)

        raise ValueError(f"Unsupported temporal smoothing mode: {self.config.mode}")

    def _ema(
        self,
        frames: list[np.ndarray],
        masks: list[np.ndarray] | None,
    ) -> list[np.ndarray]:
        alpha = float(np.clip(self.config.alpha, 0.0, 1.0))

        output: list[np.ndarray] = []

        running = frames[0].astype(np.float32)
        output.append(frames[0])

        for idx in range(1, len(frames)):
            current = frames[idx].astype(np.float32)

            running = alpha * running + (1.0 - alpha) * current

            smoothed = np.clip(running, 0, 255).astype(frames[idx].dtype)

            if masks is not None and self.config.masked_only:
                smoothed = blend_masked(
                    original=frames[idx],
                    processed=smoothed,
                    mask=masks[idx],
                )

            output.append(smoothed)

        return output

    def _window_average(
        self,
        frames: list[np.ndarray],
        masks: list[np.ndarray] | None,
    ) -> list[np.ndarray]:
        window_size = max(1, self.config.window_size)

        if window_size % 2 == 0:
            window_size += 1

        radius = window_size // 2

        output: list[np.ndarray] = []

        for idx in range(len(frames)):
            start = max(0, idx - radius)
            end = min(len(frames), idx + radius + 1)

            stack = np.stack(
                [frame.astype(np.float32) for frame in frames[start:end]],
                axis=0,
            )

            averaged = np.mean(stack, axis=0)
            averaged = np.clip(averaged, 0, 255).astype(frames[idx].dtype)

            if masks is not None and self.config.masked_only:
                averaged = blend_masked(
                    original=frames[idx],
                    processed=averaged,
                    mask=masks[idx],
                )

            output.append(averaged)

        return output


def temporal_smooth(
    frames: list[np.ndarray],
    masks: list[np.ndarray] | None = None,
) -> list[np.ndarray]:
    processor = TemporalSmoothingPostprocessor()
    return processor(frames, masks=masks)