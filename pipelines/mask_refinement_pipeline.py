# pipelines/mask_refinement_pipeline.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from detectors.utils import (
    BBox,
    empty_mask,
    to_mask_uint8,
    validate_image,
)

from mask_processing.box_filter import (
    WatermarkBoxFilter,
    WatermarkBoxFilterConfig,
)

from mask_processing.cleanup import (
    MaskCleaner,
    MaskCleanupConfig,
)

from mask_processing.fusion import (
    MaskFusion,
    MaskFusionConfig,
)


@dataclass(slots=True)
class MaskRefinementPipelineConfig:
    enabled: bool = True

    # Main stages.
    use_box_filter: bool = True
    use_fusion: bool = True
    use_cleanup: bool = True

    # Important:
    # GroundingDINO mask is usually a full rectangle box mask.
    # Do not include it by default, otherwise you inpaint whole boxes.
    include_detection_masks: bool = False

    # Include OCR / OpenCV / MobileSAM support masks in fusion.
    include_support_masks: bool = True

    # If no refined mask exists, fallback to rough detector mask.
    fallback_to_detection_mask: bool = True

    box_filter: WatermarkBoxFilterConfig = field(
        default_factory=WatermarkBoxFilterConfig
    )

    fusion: MaskFusionConfig = field(
        default_factory=MaskFusionConfig
    )

    cleanup: MaskCleanupConfig = field(
        default_factory=MaskCleanupConfig
    )

    log_events: bool = True


@dataclass(slots=True)
class MaskRefinementPipelineResult:
    mask: np.ndarray
    boxes: list[BBox] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class MaskRefinementPipeline:
    """
    Refine detector output into a final mask for inpainting.

    Input examples:
        - GroundingDINO boxes
        - OCR masks
        - OpenCV rough masks
        - MobileSAM masks

    Output:
        - final binary mask for inpainting
    """

    name = "mask_refinement_pipeline"

    def __init__(
        self,
        config: MaskRefinementPipelineConfig | None = None,
        box_filter: WatermarkBoxFilter | None = None,
        fusion: MaskFusion | None = None,
        cleaner: MaskCleaner | None = None,
    ) -> None:
        self.config = config or MaskRefinementPipelineConfig()

        self.box_filter = box_filter or WatermarkBoxFilter(self.config.box_filter)
        self.fusion = fusion or MaskFusion(self.config.fusion)
        self.cleaner = cleaner or MaskCleaner(self.config.cleanup)

    def run_image(
        self,
        image: np.ndarray,
        rough_mask: np.ndarray | None = None,
        boxes: list[BBox] | None = None,
        support_masks: list[np.ndarray] | None = None,
        detection_results: list[Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> MaskRefinementPipelineResult:
        context = context or {}
        support_masks = support_masks or []
        detection_results = detection_results or []

        validate_image(image, self.name)

        extracted_boxes = self._extract_boxes(
            boxes=boxes,
            detection_results=detection_results,
        )

        detection_masks = self._extract_masks(
            rough_mask=rough_mask,
            detection_results=detection_results,
        )

        self._log(
            "input",
            {
                "image_shape": image.shape,
                "enabled": self.config.enabled,
                "box_count": len(extracted_boxes),
                "detection_mask_count": len(detection_masks),
                "support_mask_count": len(support_masks),
                "context_keys": list(context.keys()),
            },
        )

        if not self.config.enabled:
            result = MaskRefinementPipelineResult(
                mask=self._first_available_mask_or_empty(
                    image=image,
                    masks=detection_masks + support_masks,
                ),
                boxes=extracted_boxes,
                metadata={
                    "stage": self.name,
                    "enabled": False,
                    "reason": "disabled",
                },
            )
            self._log_output(result)
            return result

        masks_for_fusion: list[np.ndarray] = []

        support_mask = None
        if support_masks and self.config.include_support_masks:
            support_fusion = self.fusion.apply(
                masks=support_masks,
                image=image,
                context={**context, "stage": "support_mask_fusion"},
            )
            support_mask = support_fusion.mask
            masks_for_fusion.append(support_mask)

        if (
            self.config.use_box_filter
            and extracted_boxes
        ):
            box_filter_result = self.box_filter.apply(
                image=image,
                boxes=extracted_boxes,
                support_mask=support_mask,
                context={**context, "stage": "box_filter"},
            )

            if np.count_nonzero(box_filter_result.mask) > 0:
                masks_for_fusion.append(box_filter_result.mask)

        if self.config.include_detection_masks:
            masks_for_fusion.extend(detection_masks)

        if not masks_for_fusion and self.config.fallback_to_detection_mask:
            masks_for_fusion.extend(detection_masks)

        if not masks_for_fusion and support_masks:
            masks_for_fusion.extend(support_masks)

        if not masks_for_fusion:
            result = MaskRefinementPipelineResult(
                mask=empty_mask(image),
                boxes=[],
                metadata={
                    "stage": self.name,
                    "enabled": True,
                    "empty": True,
                    "reason": "no_masks_after_refinement",
                    "input_box_count": len(extracted_boxes),
                    "mask_area": 0,
                },
            )
            self._log_output(result)
            return result

        if self.config.use_fusion:
            fusion_result = self.fusion.apply(
                masks=masks_for_fusion,
                image=image,
                context={**context, "stage": "final_fusion"},
            )
            refined_mask = fusion_result.mask
        else:
            refined_mask = masks_for_fusion[0]

        if self.config.use_cleanup:
            cleanup_result = self.cleaner.apply(
                mask=refined_mask,
                image=image,
                context={**context, "stage": "cleanup"},
            )

            final_mask = cleanup_result.mask
            final_boxes = cleanup_result.boxes
        else:
            final_mask = to_mask_uint8(refined_mask)
            final_boxes = extracted_boxes

        result = MaskRefinementPipelineResult(
            mask=final_mask,
            boxes=final_boxes,
            metadata={
                "stage": self.name,
                "enabled": True,
                "input_box_count": len(extracted_boxes),
                "detection_mask_count": len(detection_masks),
                "support_mask_count": len(support_masks),
                "fusion_mask_count": len(masks_for_fusion),
                "box_count": len(final_boxes),
                "mask_area": int(np.count_nonzero(final_mask)),
                "include_detection_masks": self.config.include_detection_masks,
                "fallback_to_detection_mask": self.config.fallback_to_detection_mask,
            },
        )

        self._log_output(result)
        return result

    def __call__(
        self,
        image: np.ndarray,
        rough_mask: np.ndarray | None = None,
        boxes: list[BBox] | None = None,
        support_masks: list[np.ndarray] | None = None,
        detection_results: list[Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> MaskRefinementPipelineResult:
        return self.run_image(
            image=image,
            rough_mask=rough_mask,
            boxes=boxes,
            support_masks=support_masks,
            detection_results=detection_results,
            context=context,
        )

    def _extract_boxes(
        self,
        boxes: list[BBox] | None,
        detection_results: list[Any],
    ) -> list[BBox]:
        output: list[BBox] = []

        if boxes:
            output.extend([tuple(map(int, box)) for box in boxes])

        for result in detection_results:
            result_boxes = None

            if isinstance(result, dict):
                result_boxes = result.get("boxes")
            else:
                result_boxes = getattr(result, "boxes", None)

            if not result_boxes:
                continue

            for box in result_boxes:
                output.append(tuple(map(int, box)))

        return self._deduplicate_boxes(output)

    def _extract_masks(
        self,
        rough_mask: np.ndarray | None,
        detection_results: list[Any],
    ) -> list[np.ndarray]:
        masks: list[np.ndarray] = []

        if rough_mask is not None:
            masks.append(to_mask_uint8(rough_mask))

        for result in detection_results:
            result_mask = None

            if isinstance(result, dict):
                result_mask = result.get("mask")
            else:
                result_mask = getattr(result, "mask", None)

            if result_mask is not None:
                masks.append(to_mask_uint8(result_mask))

        return masks

    def _deduplicate_boxes(
        self,
        boxes: list[BBox],
    ) -> list[BBox]:
        seen: set[BBox] = set()
        output: list[BBox] = []

        for box in boxes:
            x1, y1, x2, y2 = box
            normalized = (int(x1), int(y1), int(x2), int(y2))

            if normalized in seen:
                continue

            seen.add(normalized)
            output.append(normalized)

        return output

    def _first_available_mask_or_empty(
        self,
        image: np.ndarray,
        masks: list[np.ndarray],
    ) -> np.ndarray:
        for mask in masks:
            if mask is not None and np.count_nonzero(mask) > 0:
                return to_mask_uint8(mask)

        return empty_mask(image)

    def _log(self, event: str, payload: dict[str, Any]) -> None:
        if not self.config.log_events:
            return

        print(f"{self.name} {event}:", payload)

    def _log_output(self, result: MaskRefinementPipelineResult) -> None:
        self._log(
            "output",
            {
                "enabled": result.metadata.get("enabled"),
                "reason": result.metadata.get("reason"),
                "box_count": len(result.boxes),
                "mask_shape": result.mask.shape,
                "mask_area": result.metadata.get("mask_area"),
                "fusion_mask_count": result.metadata.get("fusion_mask_count"),
            },
        )