# pipelines/mask_refinement_pipeline.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.logger import log_event
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
    fuse_intersection,
    fuse_union,
    fuse_vote,
    fuse_weighted,
    MaskFusion,
    MaskFusionConfig,
)


@dataclass(slots=True)
class MaskCandidate:
    name: str
    mask: np.ndarray
    source_count: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)


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

    # Build multiple masks from available sources and select the strongest one.
    use_candidate_selection: bool = True
    candidate_min_area: int = 10
    candidate_max_area_ratio: float = 0.35

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

        candidates: list[MaskCandidate] = []
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
            self._add_candidate(
                candidates,
                name="support_fusion",
                mask=support_mask,
                source_count=len(support_masks),
            )

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
                self._add_candidate(
                    candidates,
                    name="box_filter",
                    mask=box_filter_result.mask,
                    source_count=len(extracted_boxes),
                )

        if self.config.include_detection_masks:
            masks_for_fusion.extend(detection_masks)
            self._add_mask_candidates(
                candidates,
                prefix="detection",
                masks=detection_masks,
            )

        if not masks_for_fusion and self.config.fallback_to_detection_mask:
            masks_for_fusion.extend(detection_masks)
            self._add_mask_candidates(
                candidates,
                prefix="fallback_detection",
                masks=detection_masks,
            )

        if not masks_for_fusion and support_masks:
            masks_for_fusion.extend(support_masks)
            self._add_mask_candidates(
                candidates,
                prefix="fallback_support",
                masks=support_masks,
            )

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

        if self.config.use_candidate_selection:
            fusion_candidates = self._build_fusion_candidates(masks_for_fusion)
            candidates.extend(fusion_candidates)

            selected_candidate = self._select_best_candidate(
                candidates=candidates,
                image_shape=image.shape[:2],
                boxes=extracted_boxes,
                reference_masks=[
                    *detection_masks,
                    *support_masks,
                ],
            )

            refined_mask = selected_candidate.mask
            refinement_metadata = {
                "candidate_selection_enabled": True,
                "selected_candidate": selected_candidate.name,
                "candidate_count": len(candidates),
                "candidate_score": selected_candidate.metadata.get("score"),
                "candidate_scores": [
                    {
                        "name": candidate.name,
                        "score": candidate.metadata.get("score"),
                        "area": candidate.metadata.get("area"),
                        "area_ratio": candidate.metadata.get("area_ratio"),
                    }
                    for candidate in candidates
                ],
            }

        elif self.config.use_fusion:
            fusion_result = self.fusion.apply(
                masks=masks_for_fusion,
                image=image,
                context={**context, "stage": "final_fusion"},
            )
            refined_mask = fusion_result.mask
            refinement_metadata = {
                "candidate_selection_enabled": False,
                "selected_candidate": "configured_fusion",
            }
        else:
            refined_mask = masks_for_fusion[0]
            refinement_metadata = {
                "candidate_selection_enabled": False,
                "selected_candidate": "first_available",
            }

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
                **refinement_metadata,
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

    def _add_candidate(
        self,
        candidates: list[MaskCandidate],
        name: str,
        mask: np.ndarray | None,
        source_count: int = 1,
    ) -> None:
        if mask is None:
            return

        mask_uint8 = to_mask_uint8(mask)

        if np.count_nonzero(mask_uint8) == 0:
            return

        candidates.append(
            MaskCandidate(
                name=name,
                mask=mask_uint8,
                source_count=source_count,
            )
        )

    def _add_mask_candidates(
        self,
        candidates: list[MaskCandidate],
        prefix: str,
        masks: list[np.ndarray],
    ) -> None:
        for index, mask in enumerate(masks):
            self._add_candidate(
                candidates,
                name=f"{prefix}_{index}",
                mask=mask,
            )

    def _build_fusion_candidates(
        self,
        masks: list[np.ndarray],
    ) -> list[MaskCandidate]:
        normalized = [
            to_mask_uint8(mask)
            for mask in masks
            if mask is not None and np.count_nonzero(mask) > 0
        ]

        if not normalized:
            return []

        candidates: list[MaskCandidate] = []
        self._add_candidate(candidates, "fusion_union", fuse_union(normalized), len(normalized))

        if len(normalized) > 1:
            self._add_candidate(
                candidates,
                "fusion_intersection",
                fuse_intersection(normalized),
                len(normalized),
            )
            self._add_candidate(
                candidates,
                "fusion_vote",
                fuse_vote(normalized, min_votes=min(2, len(normalized))),
                len(normalized),
            )
            self._add_candidate(
                candidates,
                "fusion_weighted",
                fuse_weighted(
                    normalized,
                    threshold=self.config.fusion.weighted_threshold,
                ),
                len(normalized),
            )

        return candidates

    def _select_best_candidate(
        self,
        candidates: list[MaskCandidate],
        image_shape: tuple[int, int],
        boxes: list[BBox],
        reference_masks: list[np.ndarray],
    ) -> MaskCandidate:
        if not candidates:
            return MaskCandidate(
                name="empty",
                mask=np.zeros(image_shape, dtype=np.uint8),
                metadata={
                    "score": 0.0,
                    "area": 0,
                    "area_ratio": 0.0,
                },
            )

        scored = [
            self._score_candidate(
                candidate=candidate,
                image_shape=image_shape,
                boxes=boxes,
                reference_masks=reference_masks,
            )
            for candidate in candidates
        ]

        return max(
            scored,
            key=lambda candidate: float(candidate.metadata.get("score", 0.0)),
        )

    def _score_candidate(
        self,
        candidate: MaskCandidate,
        image_shape: tuple[int, int],
        boxes: list[BBox],
        reference_masks: list[np.ndarray],
    ) -> MaskCandidate:
        mask = to_mask_uint8(candidate.mask)
        area = int(np.count_nonzero(mask))
        image_area = max(1, int(image_shape[0] * image_shape[1]))
        area_ratio = area / image_area

        if area <= 0:
            score = 0.0
        else:
            score = 1.0
            score += self._reference_overlap_score(mask, reference_masks) * 2.0
            score += self._box_coverage_score(mask, boxes, image_shape)
            score += min(candidate.source_count, 3) * 0.15

            if area < self.config.candidate_min_area:
                score -= 2.0

            if area_ratio > self.config.candidate_max_area_ratio:
                score -= (area_ratio - self.config.candidate_max_area_ratio) * 10.0

        candidate.mask = mask
        candidate.metadata.update(
            {
                "score": round(float(score), 6),
                "area": area,
                "area_ratio": round(float(area_ratio), 6),
            }
        )

        return candidate

    def _reference_overlap_score(
        self,
        mask: np.ndarray,
        reference_masks: list[np.ndarray],
    ) -> float:
        scores: list[float] = []

        mask_bool = to_mask_uint8(mask) > 0

        for reference_mask in reference_masks:
            reference_bool = to_mask_uint8(reference_mask) > 0

            if reference_bool.shape != mask_bool.shape:
                continue

            union = np.logical_or(mask_bool, reference_bool)
            union_area = int(np.count_nonzero(union))

            if union_area <= 0:
                continue

            intersection_area = int(np.count_nonzero(mask_bool & reference_bool))
            scores.append(intersection_area / union_area)

        if not scores:
            return 0.0

        return float(max(scores))

    def _box_coverage_score(
        self,
        mask: np.ndarray,
        boxes: list[BBox],
        image_shape: tuple[int, int],
    ) -> float:
        if not boxes:
            return 0.0

        box_mask = np.zeros(image_shape, dtype=np.uint8)

        for x1, y1, x2, y2 in boxes:
            x1 = max(0, min(image_shape[1], int(x1)))
            x2 = max(0, min(image_shape[1], int(x2)))
            y1 = max(0, min(image_shape[0], int(y1)))
            y2 = max(0, min(image_shape[0], int(y2)))

            if x2 > x1 and y2 > y1:
                box_mask[y1:y2, x1:x2] = 255

        mask_bool = to_mask_uint8(mask) > 0
        area = int(np.count_nonzero(mask_bool))

        if area <= 0:
            return 0.0

        inside_area = int(np.count_nonzero(mask_bool & (box_mask > 0)))
        return inside_area / area

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

        log_event(self.name, event, payload)

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
                "selected_candidate": result.metadata.get("selected_candidate"),
                "candidate_count": result.metadata.get("candidate_count"),
            },
        )
