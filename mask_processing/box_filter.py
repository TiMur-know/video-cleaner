# mask_refinement/box_filter.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from detectors.utils import (
    BBox,
    clean_binary_mask,
    empty_mask,
    lazy_import,
    pad_bbox,
    to_gray_float,
    to_mask_uint8,
    validate_image,
)

from mask_processing.cleanup import remove_small_components


SupportMode = Literal["none", "intersect", "union"]
FilterMethod = Literal["residual", "adaptive", "combined"]


@dataclass(slots=True)
class WatermarkBoxFilterConfig:
    enabled: bool = True

    input_color_order: str = "bgr"

    method: FilterMethod = "combined"

    # For residual filtering.
    # Bigger blur = smoother background estimate.
    blur_kernel_size: int = 31
    diff_threshold: float = 0.055

    # For adaptive filtering.
    adaptive_block_size: int = 35
    adaptive_c: float = -3.0

    # Remove tree/noise fragments.
    min_component_area: int = 25

    # Morphology cleanup.
    morph_kernel_size: int = 3
    close: bool = True
    open_: bool = True
    dilate_iterations: int = 1

    # Expand each rough detector box.
    box_padding: int = 0

    # Optional OCR/MobileSAM/OpenCV support mask behavior.
    support_mode: SupportMode = "none"

    log_events: bool = True


@dataclass(slots=True)
class WatermarkBoxFilterResult:
    mask: np.ndarray
    boxes: list[BBox] = field(default_factory=list)
    detections: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class WatermarkBoxFilter:
    """
    Filter actual watermark pixels inside rough boxes.

    Use after:
        GroundingDINO boxes
        OCR boxes
        OpenCV rough candidates

    This is useful because GroundingDINO may find a large box around the watermark,
    but you usually do not want to inpaint the whole box.
    """

    name = "watermark_box_filter"

    def __init__(self, config: WatermarkBoxFilterConfig | None = None) -> None:
        self.config = config or WatermarkBoxFilterConfig()

    def apply(
        self,
        image: np.ndarray,
        boxes: list[BBox],
        support_mask: np.ndarray | None = None,
        context: dict[str, Any] | None = None,
    ) -> WatermarkBoxFilterResult:
        context = context or {}

        self._log(
            "input",
            {
                "image_shape": None if image is None else image.shape,
                "box_count": len(boxes),
                "boxes_preview": boxes[:5],
                "support_mask_shape": None if support_mask is None else support_mask.shape,
                "support_mode": self.config.support_mode,
                "method": self.config.method,
                "enabled": self.config.enabled,
                "context_keys": list(context.keys()),
            },
        )

        validate_image(image, self.name)

        if not self.config.enabled:
            result = WatermarkBoxFilterResult(
                mask=empty_mask(image),
                boxes=[],
                detections=[],
                metadata={
                    "stage": self.name,
                    "enabled": False,
                    "reason": "disabled",
                    "mask_area": 0,
                },
            )
            self._log_output(result)
            return result

        if not boxes:
            result = WatermarkBoxFilterResult(
                mask=empty_mask(image),
                boxes=[],
                detections=[],
                metadata={
                    "stage": self.name,
                    "enabled": True,
                    "reason": "no_boxes",
                    "mask_area": 0,
                },
            )
            self._log_output(result)
            return result

        height, width = image.shape[:2]

        gray = to_gray_float(
            image,
            input_color_order=self.config.input_color_order,
        )

        full_mask = np.zeros((height, width), dtype=np.uint8)
        detections: list[dict[str, Any]] = []

        support_uint8 = None
        if support_mask is not None:
            support_uint8 = to_mask_uint8(support_mask)

            if support_uint8.shape[:2] != (height, width):
                raise ValueError(
                    f"support_mask shape {support_uint8.shape[:2]} does not match "
                    f"image shape {(height, width)}"
                )

        for box in boxes:
            padded_box = pad_bbox(
                bbox=box,
                image_shape=(height, width),
                padding=max(0, self.config.box_padding),
            )

            x1, y1, x2, y2 = padded_box

            if x2 <= x1 or y2 <= y1:
                continue

            crop_gray = gray[y1:y2, x1:x2]

            if crop_gray.size == 0:
                continue

            local_mask = self._filter_crop(crop_gray)

            local_mask = clean_binary_mask(
                local_mask,
                kernel_size=self.config.morph_kernel_size,
                close=self.config.close,
                open_=self.config.open_,
                dilate_iterations=self.config.dilate_iterations,
            )

            local_mask = remove_small_components(
                local_mask,
                min_area=self.config.min_component_area,
            )

            if support_uint8 is not None and self.config.support_mode != "none":
                local_support = support_uint8[y1:y2, x1:x2]
                local_mask = self._combine_with_support(
                    local_mask=local_mask,
                    local_support=local_support,
                )

            area = int(np.count_nonzero(local_mask))

            if area <= 0:
                continue

            full_mask[y1:y2, x1:x2] = np.maximum(
                full_mask[y1:y2, x1:x2],
                local_mask,
            )

            detections.append(
                {
                    "box": padded_box,
                    "mask_area": area,
                }
            )

        result = WatermarkBoxFilterResult(
            mask=full_mask,
            boxes=[item["box"] for item in detections],
            detections=detections,
            metadata={
                "stage": self.name,
                "enabled": True,
                "method": self.config.method,
                "input_box_count": len(boxes),
                "box_count": len(detections),
                "mask_area": int(np.count_nonzero(full_mask)),
                "support_mode": self.config.support_mode,
                "diff_threshold": self.config.diff_threshold,
            },
        )

        self._log_output(result)
        return result

    def __call__(
        self,
        image: np.ndarray,
        boxes: list[BBox],
        support_mask: np.ndarray | None = None,
        context: dict[str, Any] | None = None,
    ) -> WatermarkBoxFilterResult:
        return self.apply(
            image=image,
            boxes=boxes,
            support_mask=support_mask,
            context=context,
        )

    def _filter_crop(self, crop_gray: np.ndarray) -> np.ndarray:
        if self.config.method == "residual":
            return self._residual_filter(crop_gray)

        if self.config.method == "adaptive":
            return self._adaptive_filter(crop_gray)

        if self.config.method == "combined":
            residual = self._residual_filter(crop_gray)
            adaptive = self._adaptive_filter(crop_gray)
            return np.maximum(residual, adaptive)

        raise ValueError(f"Unsupported filter method: {self.config.method}")

    def _residual_filter(self, crop_gray: np.ndarray) -> np.ndarray:
        """
        Detect pixels that differ from a smooth local background estimate.
        Useful for transparent text/logos.
        """
        cv2 = lazy_import("cv2")

        k = max(3, int(self.config.blur_kernel_size))

        if k % 2 == 0:
            k += 1

        background = cv2.GaussianBlur(
            crop_gray,
            ksize=(k, k),
            sigmaX=0,
        )

        diff = np.abs(crop_gray - background)

        return (diff > self.config.diff_threshold).astype(np.uint8) * 255

    def _adaptive_filter(self, crop_gray: np.ndarray) -> np.ndarray:
        """
        Local thresholding for text-like marks, including rotated text.
        """
        cv2 = lazy_import("cv2")

        crop_uint8 = np.clip(crop_gray * 255.0, 0, 255).astype(np.uint8)

        block_size = max(3, int(self.config.adaptive_block_size))

        if block_size % 2 == 0:
            block_size += 1

        # Dark text on bright background.
        dark = cv2.adaptiveThreshold(
            crop_uint8,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size,
            self.config.adaptive_c,
        )

        # Bright text on dark background.
        bright = cv2.adaptiveThreshold(
            crop_uint8,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            self.config.adaptive_c,
        )

        return np.maximum(dark, bright)

    def _combine_with_support(
        self,
        local_mask: np.ndarray,
        local_support: np.ndarray,
    ) -> np.ndarray:
        local_mask = to_mask_uint8(local_mask)
        local_support = to_mask_uint8(local_support)

        if self.config.support_mode == "intersect":
            return np.where(
                (local_mask > 0) & (local_support > 0),
                255,
                0,
            ).astype(np.uint8)

        if self.config.support_mode == "union":
            return np.where(
                (local_mask > 0) | (local_support > 0),
                255,
                0,
            ).astype(np.uint8)

        if self.config.support_mode == "none":
            return local_mask

        raise ValueError(f"Unsupported support_mode: {self.config.support_mode}")

    def _log(self, event: str, payload: dict[str, Any]) -> None:
        if not self.config.log_events:
            return

        print(f"{self.name} {event}:", payload)

    def _log_output(self, result: WatermarkBoxFilterResult) -> None:
        self._log(
            "output",
            {
                "enabled": result.metadata.get("enabled"),
                "reason": result.metadata.get("reason"),
                "input_box_count": result.metadata.get("input_box_count"),
                "box_count": result.metadata.get("box_count"),
                "mask_shape": result.mask.shape,
                "mask_area": result.metadata.get("mask_area"),
                "method": result.metadata.get("method"),
                "support_mode": result.metadata.get("support_mode"),
            },
        )