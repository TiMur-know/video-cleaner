# mask_refinement/fusion.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from detectors.utils import (
    BBox,
    empty_mask,
    mask_to_bbox,
    resize_mask_nearest,
    to_mask_uint8,
    validate_image,
)


FusionMethod = Literal["union", "intersection", "vote", "weighted"]


@dataclass(slots=True)
class MaskFusionConfig:
    enabled: bool = True

    method: FusionMethod = "union"

    # Used for vote fusion.
    min_votes: int = 1

    # Used for weighted fusion.
    weighted_threshold: float = 0.5

    # Resize all masks to image shape.
    resize_to_image: bool = True

    log_events: bool = True


@dataclass(slots=True)
class MaskFusionResult:
    mask: np.ndarray
    boxes: list[BBox] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class MaskFusion:
    """
    Merge multiple masks into one final mask.

    Examples:
        GroundingDINO mask
        + EasyOCR text mask
        + OpenCV candidate mask
        + WatermarkBoxFilter mask
        -> fused mask
    """

    name = "mask_fusion"

    def __init__(self, config: MaskFusionConfig | None = None) -> None:
        self.config = config or MaskFusionConfig()

    def apply(
        self,
        masks: list[np.ndarray],
        image: np.ndarray | None = None,
        weights: list[float] | None = None,
        context: dict[str, Any] | None = None,
    ) -> MaskFusionResult:
        context = context or {}

        image_shape = self._resolve_image_shape(
            image=image,
            masks=masks,
        )

        self._log(
            "input",
            {
                "mask_count": len(masks),
                "image_shape": None if image is None else image.shape,
                "resolved_image_shape": image_shape,
                "method": self.config.method,
                "enabled": self.config.enabled,
                "context_keys": list(context.keys()),
            },
        )

        if not masks:
            result = MaskFusionResult(
                mask=empty_mask(image_shape),
                boxes=[],
                metadata={
                    "stage": self.name,
                    "enabled": self.config.enabled,
                    "empty": True,
                    "reason": "no_masks",
                    "mask_area": 0,
                },
            )
            self._log_output(result)
            return result

        normalized = [
            self._normalize_mask(mask, image_shape)
            for mask in masks
            if mask is not None
        ]

        if not normalized:
            result = MaskFusionResult(
                mask=empty_mask(image_shape),
                boxes=[],
                metadata={
                    "stage": self.name,
                    "enabled": self.config.enabled,
                    "empty": True,
                    "reason": "all_masks_none",
                    "mask_area": 0,
                },
            )
            self._log_output(result)
            return result

        if not self.config.enabled:
            fused = normalized[0]

        elif self.config.method == "union":
            fused = fuse_union(normalized)

        elif self.config.method == "intersection":
            fused = fuse_intersection(normalized)

        elif self.config.method == "vote":
            fused = fuse_vote(
                normalized,
                min_votes=self.config.min_votes,
            )

        elif self.config.method == "weighted":
            fused = fuse_weighted(
                normalized,
                weights=weights,
                threshold=self.config.weighted_threshold,
            )

        else:
            raise ValueError(f"Unsupported fusion method: {self.config.method}")

        boxes = self._mask_to_boxes(fused)

        result = MaskFusionResult(
            mask=fused,
            boxes=boxes,
            metadata={
                "stage": self.name,
                "enabled": self.config.enabled,
                "method": self.config.method,
                "input_mask_count": len(normalized),
                "mask_area": int(np.count_nonzero(fused)),
                "box_count": len(boxes),
            },
        )

        self._log_output(result)
        return result

    def __call__(
        self,
        masks: list[np.ndarray],
        image: np.ndarray | None = None,
        weights: list[float] | None = None,
        context: dict[str, Any] | None = None,
    ) -> MaskFusionResult:
        return self.apply(
            masks=masks,
            image=image,
            weights=weights,
            context=context,
        )

    def _resolve_image_shape(
        self,
        image: np.ndarray | None,
        masks: list[np.ndarray],
    ) -> tuple[int, int]:
        if image is not None:
            validate_image(image, self.name)
            return image.shape[:2]

        for mask in masks:
            if mask is not None:
                return mask.shape[:2]

        raise ValueError("MaskFusion requires image or at least one mask.")

    def _normalize_mask(
        self,
        mask: np.ndarray,
        image_shape: tuple[int, int],
    ) -> np.ndarray:
        mask_uint8 = to_mask_uint8(mask)

        if self.config.resize_to_image and mask_uint8.shape[:2] != image_shape:
            mask_uint8 = resize_mask_nearest(mask_uint8, image_shape)

        return to_mask_uint8(mask_uint8)

    def _mask_to_boxes(self, mask: np.ndarray) -> list[BBox]:
        bbox = mask_to_bbox(mask)
        return [bbox] if bbox is not None else []

    def _log(self, event: str, payload: dict[str, Any]) -> None:
        if not self.config.log_events:
            return

        print(f"{self.name} {event}:", payload)

    def _log_output(self, result: MaskFusionResult) -> None:
        self._log(
            "output",
            {
                "mask_shape": result.mask.shape,
                "mask_area": result.metadata.get("mask_area"),
                "box_count": len(result.boxes),
                "method": result.metadata.get("method"),
                "reason": result.metadata.get("reason"),
            },
        )


def fuse_union(masks: list[np.ndarray]) -> np.ndarray:
    if not masks:
        raise ValueError("fuse_union requires at least one mask.")

    output = np.zeros_like(to_mask_uint8(masks[0]), dtype=np.uint8)

    for mask in masks:
        output = np.maximum(output, to_mask_uint8(mask))

    return output


def fuse_intersection(masks: list[np.ndarray]) -> np.ndarray:
    if not masks:
        raise ValueError("fuse_intersection requires at least one mask.")

    output = to_mask_uint8(masks[0]) > 0

    for mask in masks[1:]:
        output &= to_mask_uint8(mask) > 0

    return output.astype(np.uint8) * 255


def fuse_vote(
    masks: list[np.ndarray],
    min_votes: int = 1,
) -> np.ndarray:
    if not masks:
        raise ValueError("fuse_vote requires at least one mask.")

    min_votes = max(1, int(min_votes))

    stack = np.stack(
        [(to_mask_uint8(mask) > 0).astype(np.uint8) for mask in masks],
        axis=0,
    )

    votes = np.sum(stack, axis=0)

    return (votes >= min_votes).astype(np.uint8) * 255


def fuse_weighted(
    masks: list[np.ndarray],
    weights: list[float] | None = None,
    threshold: float = 0.5,
) -> np.ndarray:
    if not masks:
        raise ValueError("fuse_weighted requires at least one mask.")

    if weights is None:
        weights = [1.0 for _ in masks]

    if len(weights) != len(masks):
        raise ValueError(
            f"weights length must match masks length. "
            f"Got {len(weights)} weights and {len(masks)} masks."
        )

    weights_array = np.asarray(weights, dtype=np.float32)

    if float(weights_array.sum()) <= 0:
        raise ValueError("weights must sum to a positive value.")

    normalized_weights = weights_array / weights_array.sum()

    score = np.zeros_like(to_mask_uint8(masks[0]), dtype=np.float32)

    for mask, weight in zip(masks, normalized_weights):
        score += (to_mask_uint8(mask) > 0).astype(np.float32) * float(weight)

    return (score >= threshold).astype(np.uint8) * 255