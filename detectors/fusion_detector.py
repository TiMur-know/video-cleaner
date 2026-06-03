# detectors/fusion_detector.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from detectors.utils import (
    empty_mask,
    resize_mask_nearest,
    to_mask_uint8,
)


@dataclass(slots=True)
class FusionDetectorConfig:
    enabled: bool = True

    # Ratio threshold used when fusing several masks.
    # Example:
    #   threshold=0.5 means at least 50% of valid masks must agree.
    threshold: float = 0.5

    # Alias fields used by UI/control tuning.
    mask_threshold: float = 0.5
    confidence_threshold: float = 0.5

    # Minimum detector votes required.
    # If 2 valid masks exist and min_votes=2, both must agree.
    min_votes: int = 1

    # Ignore detector results with empty masks.
    # Important: prevents empty SAM2 masks from deleting valid OpenCV masks.
    ignore_empty_masks: bool = True

    # If only one valid mask remains after filtering, return that mask directly.
    return_single_valid_mask: bool = True

    # If True, all non-empty masks are combined with max/union when voting fails.
    fallback_to_union: bool = False


@dataclass(slots=True)
class FusionDetectorResult:
    mask: np.ndarray
    detections: list[Any] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class FusionDetector:
    """
    Fuse detector masks into one final mask.

    Important behavior:
        - Empty masks are ignored.
        - If SAM2 returns an empty mask, it will not erase OpenCV/FFT/anomaly.
        - If only one valid mask remains, that mask is returned directly.
    """

    name = "fusion"

    def __init__(
        self,
        config: FusionDetectorConfig | None = None,
    ) -> None:
        self.config = config or FusionDetectorConfig()

    def detect(
        self,
        inputs: list[Any] | tuple[Any, ...] | dict[str, Any],
        image_shape: tuple[int, int],
        context: dict[str, Any] | None = None,
    ) -> FusionDetectorResult:
        context = context or {}

        if not self.config.enabled:
            return FusionDetectorResult(
                mask=empty_mask_from_shape(image_shape),
                metadata={
                    "detector": self.name,
                    "enabled": False,
                    "reason": "fusion disabled",
                },
            )

        named_inputs = normalize_inputs(inputs)

        valid_masks: list[np.ndarray] = []
        valid_names: list[str] = []
        ignored_empty_names: list[str] = []
        ignored_invalid_names: list[str] = []

        for name, item in named_inputs:
            mask = self._extract_mask(item)

            if mask is None:
                ignored_invalid_names.append(name)
                continue

            mask = self._normalize_mask(
                mask=mask,
                image_shape=image_shape,
            )

            mask_area = int(np.count_nonzero(mask))

            if self.config.ignore_empty_masks and mask_area == 0:
                ignored_empty_names.append(name)
                continue

            valid_masks.append(mask)
            valid_names.append(name)

        if not valid_masks:
            return FusionDetectorResult(
                mask=empty_mask_from_shape(image_shape),
                metadata={
                    "detector": self.name,
                    "num_inputs": len(named_inputs),
                    "num_valid_masks": 0,
                    "valid_sources": [],
                    "ignored_empty_sources": ignored_empty_names,
                    "ignored_invalid_sources": ignored_invalid_names,
                    "reason": "no valid non-empty masks",
                    "mask_area": 0,
                },
            )

        if len(valid_masks) == 1 and self.config.return_single_valid_mask:
            mask = valid_masks[0]

            return FusionDetectorResult(
                mask=mask,
                metadata={
                    "detector": self.name,
                    "num_inputs": len(named_inputs),
                    "num_valid_masks": 1,
                    "valid_sources": valid_names,
                    "ignored_empty_sources": ignored_empty_names,
                    "ignored_invalid_sources": ignored_invalid_names,
                    "mode": "single_valid_mask",
                    "mask_area": int(np.count_nonzero(mask)),
                },
            )

        fused_mask = self._vote_masks(valid_masks)

        if (
            self.config.fallback_to_union
            and int(np.count_nonzero(fused_mask)) == 0
        ):
            fused_mask = self._union_masks(valid_masks)
            mode = "fallback_union"
        else:
            mode = "vote"

        detections = self._collect_detections(named_inputs)

        return FusionDetectorResult(
            mask=fused_mask,
            detections=detections,
            metadata={
                "detector": self.name,
                "num_inputs": len(named_inputs),
                "num_valid_masks": len(valid_masks),
                "valid_sources": valid_names,
                "ignored_empty_sources": ignored_empty_names,
                "ignored_invalid_sources": ignored_invalid_names,
                "mode": mode,
                "threshold": self._effective_threshold(),
                "min_votes": self._effective_min_votes(len(valid_masks)),
                "mask_area": int(np.count_nonzero(fused_mask)),
            },
        )

    def _extract_mask(
        self,
        item: Any,
    ) -> np.ndarray | None:
        if item is None:
            return None

        if isinstance(item, np.ndarray):
            return item

        if isinstance(item, dict):
            mask = item.get("mask")
            return mask if isinstance(mask, np.ndarray) else None

        mask = getattr(item, "mask", None)

        if isinstance(mask, np.ndarray):
            return mask

        return None

    def _normalize_mask(
        self,
        mask: np.ndarray,
        image_shape: tuple[int, int],
    ) -> np.ndarray:
        if mask.ndim == 3:
            mask = mask[..., 0]

        if mask.shape[:2] != image_shape:
            mask = resize_mask_nearest(mask, image_shape)

        mask = to_mask_uint8(mask)

        if mask.ndim == 3:
            mask = mask[..., 0]

        if mask.dtype != np.uint8:
            mask = (mask > 0).astype(np.uint8) * 255

        if mask.size > 0 and mask.max() <= 1:
            mask = mask * 255

        return np.where(mask > 0, 255, 0).astype(np.uint8)

    def _vote_masks(
        self,
        masks: list[np.ndarray],
    ) -> np.ndarray:
        binary_masks = [
            (mask > 0).astype(np.uint8)
            for mask in masks
        ]

        stack = np.stack(binary_masks, axis=0)
        votes = np.sum(stack, axis=0)

        required_votes = self._effective_min_votes(len(masks))

        fused = np.where(votes >= required_votes, 255, 0).astype(np.uint8)

        return fused

    def _union_masks(
        self,
        masks: list[np.ndarray],
    ) -> np.ndarray:
        full_mask = np.zeros_like(masks[0], dtype=np.uint8)

        for mask in masks:
            full_mask = np.maximum(full_mask, mask)

        return np.where(full_mask > 0, 255, 0).astype(np.uint8)

    def _effective_threshold(self) -> float:
        threshold = self.config.threshold

        if hasattr(self.config, "mask_threshold"):
            threshold = self.config.mask_threshold

        if hasattr(self.config, "confidence_threshold"):
            threshold = self.config.confidence_threshold

        return float(np.clip(threshold, 0.0, 1.0))

    def _effective_min_votes(
        self,
        num_masks: int,
    ) -> int:
        threshold = self._effective_threshold()

        threshold_votes = int(np.ceil(num_masks * threshold))
        configured_votes = int(max(1, self.config.min_votes))

        required_votes = max(threshold_votes, configured_votes)

        return int(np.clip(required_votes, 1, num_masks))

    def _collect_detections(
        self,
        named_inputs: list[tuple[str, Any]],
    ) -> list[Any]:
        detections: list[Any] = []

        for name, item in named_inputs:
            if name == self.name:
                continue

            item_detections = getattr(item, "detections", [])

            if item_detections:
                detections.extend(item_detections)

        return detections


def normalize_inputs(
    inputs: list[Any] | tuple[Any, ...] | dict[str, Any],
) -> list[tuple[str, Any]]:
    if isinstance(inputs, dict):
        return [
            (str(name), item)
            for name, item in inputs.items()
        ]

    return [
        (infer_input_name(item, index), item)
        for index, item in enumerate(inputs)
    ]


def infer_input_name(
    item: Any,
    index: int,
) -> str:
    if item is None:
        return f"input_{index}"

    metadata = getattr(item, "metadata", None)

    if isinstance(metadata, dict):
        detector_name = metadata.get("detector")

        if detector_name:
            return str(detector_name)

    detector_name = getattr(item, "name", None)

    if detector_name:
        return str(detector_name)

    return f"input_{index}"


def empty_mask_from_shape(
    image_shape: tuple[int, int],
) -> np.ndarray:
    return np.zeros(image_shape, dtype=np.uint8)