# detectors/opencv_detector.py

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np


OpenCVDetectorMode = Literal[
    "auto",
    "adaptive",
    "edges",
    "bright",
    "dark",
    "combined",
]


@dataclass(slots=True)
class OpenCVDetectorConfig:
    enabled: bool = True

    mode: OpenCVDetectorMode = "combined"

    # Resize for faster detection. None means original size.
    max_side: int | None = 1280

    # Threshold settings
    adaptive_block_size: int = 31
    adaptive_c: int = 7

    # Edge settings
    canny_low: int = 50
    canny_high: int = 150

    # Bright/dark watermark settings
    bright_percentile: float = 92.0
    dark_percentile: float = 8.0

    # Mask cleanup
    blur_kernel: int = 3
    morph_kernel: int = 5
    dilate_iterations: int = 2
    erode_iterations: int = 1

    # Remove tiny areas
    min_area: int = 25

    # Final binary mask values
    mask_value: int = 255


@dataclass(slots=True)
class OpenCVDetectorResult:
    mask: np.ndarray
    boxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class OpenCVDetector:
    """
    Classical OpenCV watermark detector.

    Good for:
        - simple transparent text/logos
        - high contrast watermark regions
        - edge-like watermark shapes
        - quick fallback detector

    It does not load any AI model.
    """

    name = "opencv_detector"

    def __init__(
        self,
        config: OpenCVDetectorConfig | None = None,
    ) -> None:
        self.config = config or OpenCVDetectorConfig()

    def detect(
        self,
        image: np.ndarray,
        context: dict[str, Any] | None = None,
    ) -> OpenCVDetectorResult:
        context = context or {}

        if image is None:
            raise ValueError("image is required")

        if not self.config.enabled:
            return self._empty_result(image, context)

        original_shape = image.shape[:2]

        work_image, scale = self._resize_for_detection(image)
        gray = self._to_gray(work_image)

        masks: list[np.ndarray] = []

        mode = self.config.mode

        if mode in {"auto", "combined", "adaptive"}:
            masks.append(self._adaptive_mask(gray))

        if mode in {"auto", "combined", "edges"}:
            masks.append(self._edge_mask(gray))

        if mode in {"auto", "combined", "bright"}:
            masks.append(self._bright_mask(gray))

        if mode in {"auto", "combined", "dark"}:
            masks.append(self._dark_mask(gray))

        if not masks:
            return self._empty_result(image, context)

        mask = self._combine_masks(masks)
        mask = self._cleanup_mask(mask)
        mask = self._filter_small_components(mask)

        if scale != 1.0:
            mask = self._resize_mask(mask, original_shape)

        boxes = self._mask_to_boxes(mask)

        metadata = {
            "detector": self.name,
            "enabled": True,
            "mode": self.config.mode,
            "input_shape": image.shape,
            "mask_shape": mask.shape,
            "mask_area": int(np.count_nonzero(mask)),
            "box_count": len(boxes),
            "scale": scale,
            "context": context,
        }

        return OpenCVDetectorResult(
            mask=mask,
            boxes=boxes,
            scores=[1.0 for _ in boxes],
            labels=["opencv_watermark" for _ in boxes],
            metadata=metadata,
        )

    def __call__(
        self,
        image: np.ndarray,
        context: dict[str, Any] | None = None,
    ) -> OpenCVDetectorResult:
        return self.detect(image, context=context)

    def _empty_result(
        self,
        image: np.ndarray,
        context: dict[str, Any] | None = None,
    ) -> OpenCVDetectorResult:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)

        return OpenCVDetectorResult(
            mask=mask,
            boxes=[],
            scores=[],
            labels=[],
            metadata={
                "detector": self.name,
                "enabled": False,
                "mask_area": 0,
                "context": context or {},
            },
        )

    def _resize_for_detection(
        self,
        image: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        if self.config.max_side is None:
            return image, 1.0

        height, width = image.shape[:2]
        max_side = max(height, width)

        if max_side <= self.config.max_side:
            return image, 1.0

        scale = self.config.max_side / float(max_side)
        new_width = max(1, int(width * scale))
        new_height = max(1, int(height * scale))

        cv2 = lazy_import_cv2()

        resized = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA,
        )

        return resized, scale

    def _to_gray(self, image: np.ndarray) -> np.ndarray:
        cv2 = lazy_import_cv2()

        if image.ndim == 2:
            gray = image

        elif image.ndim == 3 and image.shape[-1] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        elif image.ndim == 3 and image.shape[-1] == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        if gray.dtype != np.uint8:
            gray = np.clip(gray, 0, 255).astype(np.uint8)

        return gray

    def _adaptive_mask(self, gray: np.ndarray) -> np.ndarray:
        cv2 = lazy_import_cv2()

        block_size = self._odd_at_least(
            self.config.adaptive_block_size,
            minimum=3,
        )

        blurred = self._maybe_blur(gray)

        mask = cv2.adaptiveThreshold(
            blurred,
            self.config.mask_value,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            self.config.adaptive_c,
        )

        # Adaptive threshold often returns large white areas.
        # Keep only local details by subtracting from a smoothed version.
        inv_mask = cv2.bitwise_not(mask)

        return inv_mask

    def _edge_mask(self, gray: np.ndarray) -> np.ndarray:
        cv2 = lazy_import_cv2()

        blurred = self._maybe_blur(gray)

        edges = cv2.Canny(
            blurred,
            self.config.canny_low,
            self.config.canny_high,
        )

        return edges

    def _bright_mask(self, gray: np.ndarray) -> np.ndarray:
        threshold = np.percentile(gray, self.config.bright_percentile)
        mask = np.where(gray >= threshold, self.config.mask_value, 0)

        return mask.astype(np.uint8)

    def _dark_mask(self, gray: np.ndarray) -> np.ndarray:
        threshold = np.percentile(gray, self.config.dark_percentile)
        mask = np.where(gray <= threshold, self.config.mask_value, 0)

        return mask.astype(np.uint8)

    def _combine_masks(
        self,
        masks: list[np.ndarray],
    ) -> np.ndarray:
        combined = np.zeros_like(masks[0], dtype=np.uint8)

        for mask in masks:
            combined = np.maximum(combined, mask.astype(np.uint8))

        return combined

    def _cleanup_mask(
        self,
        mask: np.ndarray,
    ) -> np.ndarray:
        cv2 = lazy_import_cv2()

        kernel_size = self._odd_at_least(
            self.config.morph_kernel,
            minimum=3,
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )

        cleaned = mask

        if self.config.dilate_iterations > 0:
            cleaned = cv2.dilate(
                cleaned,
                kernel,
                iterations=self.config.dilate_iterations,
            )

        if self.config.erode_iterations > 0:
            cleaned = cv2.erode(
                cleaned,
                kernel,
                iterations=self.config.erode_iterations,
            )

        cleaned = cv2.morphologyEx(
            cleaned,
            cv2.MORPH_CLOSE,
            kernel,
        )

        _, cleaned = cv2.threshold(
            cleaned,
            1,
            self.config.mask_value,
            cv2.THRESH_BINARY,
        )

        return cleaned

    def _filter_small_components(
        self,
        mask: np.ndarray,
    ) -> np.ndarray:
        cv2 = lazy_import_cv2()

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )

        output = np.zeros_like(mask, dtype=np.uint8)

        for label_index in range(1, num_labels):
            area = int(stats[label_index, cv2.CC_STAT_AREA])

            if area < self.config.min_area:
                continue

            output[labels == label_index] = self.config.mask_value

        return output

    def _mask_to_boxes(
        self,
        mask: np.ndarray,
    ) -> list[tuple[int, int, int, int]]:
        cv2 = lazy_import_cv2()

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        boxes: list[tuple[int, int, int, int]] = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            if w * h < self.config.min_area:
                continue

            boxes.append((int(x), int(y), int(x + w), int(y + h)))

        return boxes

    def _resize_mask(
        self,
        mask: np.ndarray,
        original_shape: tuple[int, int],
    ) -> np.ndarray:
        cv2 = lazy_import_cv2()

        height, width = original_shape

        resized = cv2.resize(
            mask,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )

        _, resized = cv2.threshold(
            resized,
            1,
            self.config.mask_value,
            cv2.THRESH_BINARY,
        )

        return resized.astype(np.uint8)

    def _maybe_blur(
        self,
        gray: np.ndarray,
    ) -> np.ndarray:
        if self.config.blur_kernel <= 1:
            return gray

        cv2 = lazy_import_cv2()

        kernel = self._odd_at_least(
            self.config.blur_kernel,
            minimum=3,
        )

        return cv2.GaussianBlur(
            gray,
            (kernel, kernel),
            0,
        )

    @staticmethod
    def _odd_at_least(
        value: int,
        minimum: int,
    ) -> int:
        value = max(int(value), minimum)

        if value % 2 == 0:
            value += 1

        return value


def lazy_import_cv2() -> Any:
    try:
        return importlib.import_module("cv2")

    except ImportError as exc:
        raise ImportError(
            "OpenCV is required for OpenCVDetector. "
            "Install it with: pip install opencv-python"
        ) from exc