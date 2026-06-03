# mask_refinement/cleanup.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from detectors.utils import (
    BBox,
    clean_binary_mask,
    dilate_mask,
    empty_mask,
    lazy_import,
    mask_to_bbox,
    resize_mask_nearest,
    to_mask_uint8,
    validate_image,
)


@dataclass(slots=True)
class MaskCleanupConfig:
    enabled: bool = True

    # Remove tiny noise components.
    min_area: int = 25

    # Morphology cleanup.
    kernel_size: int = 3
    close: bool = True
    open_: bool = True

    # Expand mask slightly for safer inpainting.
    dilate_iterations: int = 1
    dilate_kernel_size: int = 5

    # Fill holes inside detected mask regions.
    fill_holes: bool = True

    # If provided, force mask to this image shape.
    resize_to_image: bool = True

    log_events: bool = True


@dataclass(slots=True)
class MaskCleanupResult:
    mask: np.ndarray
    boxes: list[BBox] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class MaskCleaner:
    """
    Clean a rough binary mask before inpainting.

    Typical use:
        rough detector mask
            -> remove tiny noise
            -> close gaps
            -> fill holes
            -> dilate a little
            -> final inpaint mask
    """

    name = "mask_cleanup"

    def __init__(self, config: MaskCleanupConfig | None = None) -> None:
        self.config = config or MaskCleanupConfig()

    def apply(
        self,
        mask: np.ndarray,
        image: np.ndarray | None = None,
        context: dict[str, Any] | None = None,
    ) -> MaskCleanupResult:
        context = context or {}

        self._log(
            "input",
            {
                "mask_shape": None if mask is None else mask.shape,
                "image_shape": None if image is None else image.shape,
                "enabled": self.config.enabled,
                "context_keys": list(context.keys()),
            },
        )

        if mask is None:
            if image is None:
                raise ValueError("MaskCleaner requires mask or image.")

            result = MaskCleanupResult(
                mask=empty_mask(image),
                boxes=[],
                metadata={
                    "stage": self.name,
                    "enabled": self.config.enabled,
                    "empty": True,
                    "reason": "mask_none",
                    "mask_area": 0,
                },
            )
            self._log_output(result)
            return result

        if image is not None:
            validate_image(image, self.name)
            image_shape = image.shape[:2]
        else:
            image_shape = mask.shape[:2]

        mask_uint8 = to_mask_uint8(mask)

        if self.config.resize_to_image and mask_uint8.shape[:2] != image_shape:
            mask_uint8 = resize_mask_nearest(mask_uint8, image_shape)

        if not self.config.enabled:
            boxes = self._mask_to_boxes(mask_uint8)
            result = MaskCleanupResult(
                mask=mask_uint8,
                boxes=boxes,
                metadata={
                    "stage": self.name,
                    "enabled": False,
                    "mask_area": int(np.count_nonzero(mask_uint8)),
                    "box_count": len(boxes),
                },
            )
            self._log_output(result)
            return result

        cleaned = clean_binary_mask(
            mask_uint8,
            kernel_size=self.config.kernel_size,
            close=self.config.close,
            open_=self.config.open_,
            dilate_iterations=0,
        )

        cleaned = remove_small_components(
            cleaned,
            min_area=self.config.min_area,
        )

        if self.config.fill_holes:
            cleaned = fill_mask_holes(cleaned)

        if self.config.dilate_iterations > 0:
            cleaned = dilate_mask(
                cleaned,
                iterations=self.config.dilate_iterations,
                kernel_size=self.config.dilate_kernel_size,
            )

        boxes = self._mask_to_boxes(cleaned)

        result = MaskCleanupResult(
            mask=cleaned,
            boxes=boxes,
            metadata={
                "stage": self.name,
                "enabled": True,
                "input_mask_area": int(np.count_nonzero(mask_uint8)),
                "mask_area": int(np.count_nonzero(cleaned)),
                "box_count": len(boxes),
                "min_area": self.config.min_area,
                "dilate_iterations": self.config.dilate_iterations,
            },
        )

        self._log_output(result)
        return result

    def __call__(
        self,
        mask: np.ndarray,
        image: np.ndarray | None = None,
        context: dict[str, Any] | None = None,
    ) -> MaskCleanupResult:
        return self.apply(mask=mask, image=image, context=context)

    def _mask_to_boxes(self, mask: np.ndarray) -> list[BBox]:
        bbox = mask_to_bbox(mask)
        return [bbox] if bbox is not None else []

    def _log(self, event: str, payload: dict[str, Any]) -> None:
        if not self.config.log_events:
            return

        print(f"{self.name} {event}:", payload)

    def _log_output(self, result: MaskCleanupResult) -> None:
        self._log(
            "output",
            {
                "mask_shape": result.mask.shape,
                "mask_area": result.metadata.get("mask_area"),
                "box_count": len(result.boxes),
                "enabled": result.metadata.get("enabled"),
                "reason": result.metadata.get("reason"),
            },
        )


def remove_small_components(
    mask: np.ndarray,
    min_area: int = 25,
) -> np.ndarray:
    """
    Remove connected components smaller than min_area.
    """
    mask_uint8 = to_mask_uint8(mask)

    if min_area <= 0:
        return mask_uint8

    cv2 = lazy_import("cv2")

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_uint8,
        connectivity=8,
    )

    output = np.zeros_like(mask_uint8, dtype=np.uint8)

    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])

        if area >= min_area:
            output[labels == label_idx] = 255

    return output


def fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    """
    Fill holes inside binary mask regions.
    """
    mask_uint8 = to_mask_uint8(mask)

    if not np.any(mask_uint8):
        return mask_uint8

    cv2 = lazy_import("cv2")

    height, width = mask_uint8.shape[:2]

    flood = mask_uint8.copy()
    flood_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)

    cv2.floodFill(
        flood,
        flood_mask,
        seedPoint=(0, 0),
        newVal=255,
    )

    holes = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(mask_uint8, holes)

    return to_mask_uint8(filled)


def mask_area(mask: np.ndarray) -> int:
    return int(np.count_nonzero(to_mask_uint8(mask)))


def mask_is_empty(mask: np.ndarray | None) -> bool:
    if mask is None:
        return True

    return mask_area(mask) == 0