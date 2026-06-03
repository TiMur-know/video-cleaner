# trackers/kalman_tracker.py

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
                f"Install it before using KalmanTracker."
            ) from exc

    return _MODULE_CACHE[module_name]


BBox = tuple[int, int, int, int]


@dataclass(slots=True)
class KalmanTrackerConfig:
    enabled: bool = True

    # State:
    # cx, cy, w, h, vx, vy, vw, vh
    process_noise: float = 1e-2
    measurement_noise: float = 1e-1

    # If no detection is available, keep predicting for this many frames.
    max_missing: int = 10

    # Threshold for binary mask conversion.
    mask_threshold: int = 127

    # Clean output mask.
    apply_morphology: bool = True
    morph_kernel_size: int = 3


class KalmanTracker:
    """
    Kalman-based watermark mask tracker.

    Purpose:
        Smooth and predict watermark position across frames.

    Best for:
        - stable overlay watermarks
        - logos/text that move slowly
        - filling short detection gaps

    Input:
        detected mask on current frame, if available

    Output:
        smoothed/predicted mask
    """

    name = "kalman"

    def __init__(self, config: KalmanTrackerConfig | None = None) -> None:
        self.config = config or KalmanTrackerConfig()

        self.kalman: Any | None = None
        self.initialized = False
        self.missing_count = 0

        self.last_mask: np.ndarray | None = None
        self.last_bbox: BBox | None = None
        self.last_output_shape: tuple[int, int] | None = None

    def initialize(self, mask: np.ndarray) -> None:
        """
        Initialize tracker from first detected mask.
        """
        mask_uint8 = self._to_mask_uint8(mask)
        bbox = mask_to_bbox(mask_uint8)

        if bbox is None:
            raise ValueError("Cannot initialize KalmanTracker with empty mask.")

        self.last_mask = mask_uint8
        self.last_bbox = bbox
        self.last_output_shape = mask_uint8.shape[:2]

        self.kalman = self._create_kalman()
        measurement = bbox_to_measurement(bbox)

        self.kalman.statePre[:4, 0] = measurement[:, 0]
        self.kalman.statePost[:4, 0] = measurement[:, 0]

        self.initialized = True
        self.missing_count = 0

    def track(
        self,
        detected_mask: np.ndarray | None = None,
        output_shape: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """
        Track one frame.

        detected_mask:
            Current detection mask. Can be None if detector failed.

        output_shape:
            Required only if detected_mask is None before shape is known.

        Returns:
            Binary uint8 mask.
        """
        if not self.config.enabled:
            if detected_mask is None:
                if self.last_mask is None:
                    raise RuntimeError("No mask available.")
                return self.last_mask

            return self._to_mask_uint8(detected_mask)

        if not self.initialized:
            if detected_mask is None:
                raise RuntimeError(
                    "KalmanTracker must be initialized with a mask first."
                )

            self.initialize(detected_mask)
            return self._to_mask_uint8(detected_mask)

        assert self.kalman is not None

        prediction = self.kalman.predict()
        predicted_bbox = state_to_bbox(prediction)

        if detected_mask is not None:
            mask_uint8 = self._to_mask_uint8(detected_mask)
            measured_bbox = mask_to_bbox(mask_uint8)

            if measured_bbox is not None:
                measurement = bbox_to_measurement(measured_bbox)
                corrected = self.kalman.correct(measurement)

                smoothed_bbox = state_to_bbox(corrected)

                output = self._warp_last_mask_to_bbox(
                    source_mask=mask_uint8,
                    source_bbox=measured_bbox,
                    target_bbox=smoothed_bbox,
                    output_shape=mask_uint8.shape[:2],
                )

                self.last_mask = output
                self.last_bbox = smoothed_bbox
                self.last_output_shape = mask_uint8.shape[:2]
                self.missing_count = 0

                return self._clean_mask(output)

        self.missing_count += 1

        if self.missing_count > self.config.max_missing:
            if output_shape is None and self.last_output_shape is None:
                raise RuntimeError("output_shape is required after track is lost.")

            shape = output_shape or self.last_output_shape
            assert shape is not None

            return np.zeros(shape, dtype=np.uint8)

        if self.last_mask is None or self.last_bbox is None:
            if output_shape is None:
                raise RuntimeError("output_shape is required when no last mask exists.")

            return np.zeros(output_shape, dtype=np.uint8)

        shape = output_shape or self.last_output_shape

        if shape is None:
            raise RuntimeError("output_shape is unknown.")

        output = self._warp_last_mask_to_bbox(
            source_mask=self.last_mask,
            source_bbox=self.last_bbox,
            target_bbox=predicted_bbox,
            output_shape=shape,
        )

        self.last_mask = output
        self.last_bbox = predicted_bbox
        self.last_output_shape = shape

        return self._clean_mask(output)

    def reset(self) -> None:
        self.kalman = None
        self.initialized = False
        self.missing_count = 0
        self.last_mask = None
        self.last_bbox = None
        self.last_output_shape = None

    def _create_kalman(self) -> Any:
        cv2 = lazy_import("cv2")

        kalman = cv2.KalmanFilter(8, 4)

        # State transition matrix.
        # cx, cy, w, h, vx, vy, vw, vh
        kalman.transitionMatrix = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float32,
        )

        # Measurement matrix.
        # We measure cx, cy, w, h.
        kalman.measurementMatrix = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
            ],
            dtype=np.float32,
        )

        kalman.processNoiseCov = (
            np.eye(8, dtype=np.float32) * self.config.process_noise
        )
        kalman.measurementNoiseCov = (
            np.eye(4, dtype=np.float32) * self.config.measurement_noise
        )
        kalman.errorCovPost = np.eye(8, dtype=np.float32)

        return kalman

    def _warp_last_mask_to_bbox(
        self,
        source_mask: np.ndarray,
        source_bbox: BBox,
        target_bbox: BBox,
        output_shape: tuple[int, int],
    ) -> np.ndarray:
        """
        Move/scale source mask from source_bbox to target_bbox.
        """
        cv2 = lazy_import("cv2")

        source_mask = self._to_mask_uint8(source_mask)

        sx1, sy1, sx2, sy2 = source_bbox
        tx1, ty1, tx2, ty2 = target_bbox

        source_crop = source_mask[sy1:sy2, sx1:sx2]

        if source_crop.size == 0:
            return np.zeros(output_shape, dtype=np.uint8)

        target_width = max(1, tx2 - tx1)
        target_height = max(1, ty2 - ty1)

        resized_crop = cv2.resize(
            source_crop,
            dsize=(target_width, target_height),
            interpolation=cv2.INTER_NEAREST,
        )

        output = np.zeros(output_shape, dtype=np.uint8)

        height, width = output_shape

        paste_x1 = max(0, tx1)
        paste_y1 = max(0, ty1)
        paste_x2 = min(width, tx2)
        paste_y2 = min(height, ty2)

        crop_x1 = max(0, -tx1)
        crop_y1 = max(0, -ty1)
        crop_x2 = crop_x1 + max(0, paste_x2 - paste_x1)
        crop_y2 = crop_y1 + max(0, paste_y2 - paste_y1)

        if paste_x2 <= paste_x1 or paste_y2 <= paste_y1:
            return output

        output[paste_y1:paste_y2, paste_x1:paste_x2] = resized_crop[
            crop_y1:crop_y2,
            crop_x1:crop_x2,
        ]

        return output

    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        if not self.config.apply_morphology:
            return mask

        cv2 = lazy_import("cv2")

        kernel_size = max(1, self.config.morph_kernel_size)
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

        cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

        return cleaned

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

        if mask.max() <= 1.0:
            return (mask > 0.5).astype(np.uint8) * 255

        return (mask > 127).astype(np.uint8) * 255


def mask_to_bbox(mask: np.ndarray) -> BBox | None:
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]

    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        return None

    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1

    return x1, y1, x2, y2


def bbox_to_measurement(bbox: BBox) -> np.ndarray:
    x1, y1, x2, y2 = bbox

    width = max(1, x2 - x1)
    height = max(1, y2 - y1)

    cx = x1 + width / 2.0
    cy = y1 + height / 2.0

    return np.array([[cx], [cy], [width], [height]], dtype=np.float32)


def state_to_bbox(state: np.ndarray) -> BBox:
    cx = float(state[0, 0])
    cy = float(state[1, 0])
    width = max(1.0, float(state[2, 0]))
    height = max(1.0, float(state[3, 0]))

    x1 = round(cx - width / 2.0)
    y1 = round(cy - height / 2.0)
    x2 = round(cx + width / 2.0)
    y2 = round(cy + height / 2.0)

    return int(x1), int(y1), int(x2), int(y2)