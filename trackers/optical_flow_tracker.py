# trackers/optical_flow_tracker.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import importlib
import numpy as np


_MODULE_CACHE: dict[str, Any] = {}


def lazy_import(module_name: str) -> Any:
    if module_name not in _MODULE_CACHE:
        try:
            _MODULE_CACHE[module_name] = importlib.import_module(module_name)
        except ImportError as exc:
            raise ImportError(
                f"Missing optional dependency '{module_name}'. "
                f"Install it before using OpticalFlowTracker."
            ) from exc

    return _MODULE_CACHE[module_name]


@dataclass(slots=True)
class OpticalFlowTrackerConfig:
    enabled: bool = True

    pyr_scale: float = 0.5
    levels: int = 3
    winsize: int = 21
    iterations: int = 3
    poly_n: int = 5
    poly_sigma: float = 1.2

    mask_threshold: int = 127

    # Clean tracked mask after warping.
    morph_kernel_size: int = 3
    apply_morphology: bool = True


class OpticalFlowTracker:
    """
    Dense optical-flow mask tracker.

    Purpose:
        Propagate a watermark mask from previous frame to current frame.

    Input:
        previous frame
        current frame
        previous mask

    Output:
        estimated current mask

    Best for:
        - small camera motion
        - slowly moving watermark
        - stable overlays
    """

    name = "optical_flow"

    def __init__(self, config: OpticalFlowTrackerConfig | None = None) -> None:
        self.config = config or OpticalFlowTrackerConfig()

    def track(
        self,
        previous_frame: np.ndarray,
        current_frame: np.ndarray,
        previous_mask: np.ndarray,
    ) -> np.ndarray:
        if not self.config.enabled:
            return previous_mask

        self._validate_inputs(previous_frame, current_frame, previous_mask)

        cv2 = lazy_import("cv2")

        previous_gray = self._to_gray(previous_frame)
        current_gray = self._to_gray(current_frame)

        flow = cv2.calcOpticalFlowFarneback(
            previous_gray,
            current_gray,
            None,
            pyr_scale=self.config.pyr_scale,
            levels=self.config.levels,
            winsize=self.config.winsize,
            iterations=self.config.iterations,
            poly_n=self.config.poly_n,
            poly_sigma=self.config.poly_sigma,
            flags=0,
        )

        tracked_mask = self._warp_mask(previous_mask, flow)

        if self.config.apply_morphology:
            tracked_mask = self._clean_mask(tracked_mask)

        return tracked_mask

    def _warp_mask(self, mask: np.ndarray, flow: np.ndarray) -> np.ndarray:
        """
        Warp previous mask into current frame using dense optical flow.
        """
        cv2 = lazy_import("cv2")

        mask_uint8 = self._to_mask_uint8(mask)

        height, width = mask_uint8.shape[:2]

        grid_x, grid_y = np.meshgrid(
            np.arange(width, dtype=np.float32),
            np.arange(height, dtype=np.float32),
        )

        flow_x = flow[..., 0]
        flow_y = flow[..., 1]

        map_x = grid_x - flow_x
        map_y = grid_y - flow_y

        warped = cv2.remap(
            mask_uint8,
            map_x,
            map_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        _, binary = cv2.threshold(
            warped,
            self.config.mask_threshold,
            255,
            cv2.THRESH_BINARY,
        )

        return binary

    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        cv2 = lazy_import("cv2")

        kernel_size = max(1, self.config.morph_kernel_size)

        kernel = np.ones(
            (kernel_size, kernel_size),
            dtype=np.uint8,
        )

        cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

        return cleaned

    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        cv2 = lazy_import("cv2")

        if frame.ndim == 2:
            return frame

        if frame.ndim == 3 and frame.shape[-1] == 1:
            return frame[..., 0]

        if frame.ndim == 3 and frame.shape[-1] == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if frame.ndim == 3 and frame.shape[-1] == 4:
            return cv2.cvtColor(frame[..., :3], cv2.COLOR_BGR2GRAY)

        raise ValueError(f"Unsupported frame shape: {frame.shape}")

    @staticmethod
    def _to_mask_uint8(mask: np.ndarray) -> np.ndarray:
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask[..., 0]

        if mask.ndim != 2:
            raise ValueError(f"Mask must be HxW or HxWx1, got {mask.shape}")

        if mask.dtype == np.bool_:
            return mask.astype(np.uint8) * 255

        if mask.dtype == np.uint8:
            if mask.max() <= 1:
                return mask * 255
            return mask

        return (mask > 0).astype(np.uint8) * 255

    @staticmethod
    def _validate_inputs(
        previous_frame: np.ndarray,
        current_frame: np.ndarray,
        previous_mask: np.ndarray,
    ) -> None:
        if previous_frame is None:
            raise ValueError("previous_frame is None")

        if current_frame is None:
            raise ValueError("current_frame is None")

        if previous_mask is None:
            raise ValueError("previous_mask is None")

        if previous_frame.shape[:2] != current_frame.shape[:2]:
            raise ValueError(
                "previous_frame and current_frame must have the same height/width"
            )

        if previous_frame.shape[:2] != previous_mask.shape[:2]:
            raise ValueError(
                "previous_frame and previous_mask must have the same height/width"
            )