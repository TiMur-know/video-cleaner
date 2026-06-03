# detectors/mobile_sam_2_detector.py

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from mobile_sam import (
    SamAutomaticMaskGenerator,
    SamPredictor,
    sam_model_registry,
)

from detectors.utils import (
    best_mask_index,
    extract_auto_mask,
    has_prompts,
    mask_to_bbox,
    merge_masks,
    normalize_boxes,
    normalize_point_labels,
    normalize_points,
    points_inside_box,
    prompt_value,
    summarize_prompts,
    to_mask_uint8,
    to_rgb,
    validate_image,
)


@dataclass
class SAMResult:
    masks: np.ndarray
    scores: Optional[np.ndarray] = None
    logits: Optional[np.ndarray] = None


@dataclass(slots=True)
class MobileSAM2DetectorConfig:
    enabled: bool = True
    checkpoint_path: str | None = "models/mobile_sam/mobile_sam.pt"
    model_type: str = "vit_t"
    device: str = "auto"
    automatic_mask_kwargs: Optional[Dict[str, Any]] = None
    fallback_to_empty: bool = True
    strict_loading: bool = False
    use_boxes: bool = True
    use_points: bool = True
    use_mask_input: bool = False
    require_prompts: bool = True
    mask_threshold: float = 0.5
    mask_value: int = 255
    min_area: int = 25


@dataclass(slots=True)
class MobileSAM2DetectorResult:
    mask: np.ndarray
    boxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    detections: list[Any] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class MobileSAM2Detector:
    """
    MobileSAM2 detector based on MobileSAM.

    Supports:
        - prompt-based segmentation with boxes
        - prompt-based segmentation with points
        - prompt-based segmentation with boxes + points
        - automatic full-image mask generation
    """

    name = "mobile_sam_2"

    def __init__(
        self,
        config: MobileSAM2DetectorConfig | None = None,
        model: Any | None = None,
        predictor: Any | None = None,
    ) -> None:
        self.config = config or MobileSAM2DetectorConfig()
        self.model = model
        self.predictor = predictor

        self.image_rgb: Optional[np.ndarray] = None
        self.image_bgr: Optional[np.ndarray] = None

        self._loaded = model is not None or predictor is not None

    def detect(
        self,
        image: np.ndarray,
        prompts: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> MobileSAM2DetectorResult:
        context = context or {}
        prompts = prompts or {}

        self._log_detect_input(image=image, prompts=prompts)

        validate_image(image, self.name)

        if not self.config.enabled:
            result = self._empty_result(
                image=image,
                reason="disabled",
                context=context,
            )
            self._log_detect_output(result)
            return result

        if self.config.require_prompts and not has_prompts(prompts):
            result = self._empty_result(
                image=image,
                reason="no_prompts",
                context=context,
            )
            self._log_detect_output(result)
            return result

        try:
            self._ensure_loaded()
        except Exception as exc:
            if self.config.strict_loading or not self.config.fallback_to_empty:
                raise

            result = self._empty_result(
                image=image,
                reason=f"model_load_failed: {type(exc).__name__}: {exc}",
                context=context,
            )
            self._log_detect_output(result)
            return result

        try:
            self.set_image(image, input_format="BGR")

            if has_prompts(prompts):
                sam_result = self._run_prompts(prompts)
            else:
                sam_result = self._segment_everything_result()

            mask = self.get_best_mask(sam_result)

            mask_uint8 = to_mask_uint8(
                mask,
                threshold=self.config.mask_threshold,
            )

            bbox = mask_to_bbox(mask_uint8)
            boxes = [bbox] if bbox is not None else []

            scores = (
                sam_result.scores.tolist()
                if sam_result.scores is not None
                else [1.0]
            )

            result = MobileSAM2DetectorResult(
                mask=mask_uint8,
                boxes=boxes,
                scores=scores,
                labels=[self.name for _ in boxes],
                detections=[],
                metadata={
                    "detector": self.name,
                    "enabled": True,
                    "input_shape": image.shape,
                    "mask_shape": mask_uint8.shape,
                    "mask_area": int(np.count_nonzero(mask_uint8)),
                    "box_count": len(boxes),
                    "prompt_keys": list(prompts.keys()),
                    "device": self._resolve_device(),
                    "checkpoint_path": self.config.checkpoint_path,
                    "model_type": self.config.model_type,
                },
            )

            self._log_detect_output(result)
            return result

        except Exception as exc:
            if self.config.strict_loading or not self.config.fallback_to_empty:
                raise

            result = self._empty_result(
                image=image,
                reason=f"predict_failed: {type(exc).__name__}: {exc}",
                context=context,
            )
            self._log_detect_output(result)
            return result

    def __call__(
        self,
        image: np.ndarray,
        prompts: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> MobileSAM2DetectorResult:
        return self.detect(
            image=image,
            prompts=prompts,
            context=context,
        )

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return

        if self.predictor is not None and self.model is not None:
            self._loaded = True
            return

        if self.model is None:
            self.model = self._load_model()

        if self.predictor is None:
            self.predictor = SamPredictor(self.model)

        self._loaded = True

    def _load_model(self) -> Any:
        if not self.config.checkpoint_path:
            raise ValueError("checkpoint_path is required for MobileSAM2Detector.")

        checkpoint = Path(self.config.checkpoint_path)

        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

        model_type = self.config.model_type

        if model_type not in sam_model_registry:
            raise ValueError(
                f"Unknown model_type '{model_type}'. "
                f"Available keys: {sorted(sam_model_registry.keys())}"
            )

        model = sam_model_registry[model_type](checkpoint=str(checkpoint))
        model.to(device=self._resolve_device())
        model.eval()

        return model

    def _resolve_device(self) -> str:
        if self.config.device != "auto":
            return self.config.device

        if torch.cuda.is_available():
            return "cuda"

        return "cpu"

    def set_image(
        self,
        image: np.ndarray,
        input_format: str = "BGR",
    ) -> None:
        validate_image(image, self.name)

        image_rgb = to_rgb(
            image,
            input_color_order=input_format.lower(),
        )

        if input_format.upper() == "BGR":
            image_bgr = image[..., :3]
        elif input_format.upper() == "RGB":
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        else:
            raise ValueError("input_format must be 'BGR' or 'RGB'")

        self.image_rgb = image_rgb
        self.image_bgr = image_bgr

        if self.predictor is None:
            raise RuntimeError("MobileSAM predictor is not loaded.")

        self.predictor.set_image(image_rgb)

    def segment_by_points(
        self,
        points: Sequence[Tuple[int, int]],
        labels: Sequence[int],
        multimask_output: bool = True,
    ) -> SAMResult:
        self._check_image_loaded()

        print(
            "MobileSAM2Detector segment_by_points input:",
            {
                "points_shape": getattr(points, "shape", None),
                "labels_shape": getattr(labels, "shape", None),
                "points_length": len(points),
                "labels_length": len(labels),
            },
        )

        input_points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        input_labels = np.asarray(labels, dtype=np.int32).reshape(-1)

        if len(input_points) != len(input_labels):
            raise ValueError(
                f"points and labels length mismatch: "
                f"{len(input_points)} points, {len(input_labels)} labels"
            )

        masks, scores, logits = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=multimask_output,
        )

        result = SAMResult(masks=masks, scores=scores, logits=logits)

        print(
            "MobileSAM2Detector segment_by_points output:",
            {
                "masks_shape": masks.shape,
                "scores_shape": None if scores is None else scores.shape,
            },
        )

        return result

    def segment_by_box(
        self,
        box: Tuple[int, int, int, int],
        multimask_output: bool = True,
    ) -> SAMResult:
        self._check_image_loaded()

        print("MobileSAM2Detector segment_by_box input:", {"box": box})

        input_box = np.asarray(box, dtype=np.int32).reshape(4)

        masks, scores, logits = self.predictor.predict(
            box=input_box,
            multimask_output=multimask_output,
        )

        result = SAMResult(masks=masks, scores=scores, logits=logits)

        print(
            "MobileSAM2Detector segment_by_box output:",
            {
                "masks_shape": masks.shape,
                "scores_shape": None if scores is None else scores.shape,
            },
        )

        return result

    def segment_by_points_and_box(
        self,
        points: Sequence[Tuple[int, int]],
        labels: Sequence[int],
        box: Tuple[int, int, int, int],
        multimask_output: bool = True,
    ) -> SAMResult:
        self._check_image_loaded()

        print(
            "MobileSAM2Detector segment_by_points_and_box input:",
            {
                "points_shape": getattr(points, "shape", None),
                "labels_shape": getattr(labels, "shape", None),
                "points_length": len(points),
                "labels_length": len(labels),
                "box": box,
            },
        )

        input_points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        input_labels = np.asarray(labels, dtype=np.int32).reshape(-1)
        input_box = np.asarray(box, dtype=np.int32).reshape(4)

        if len(input_points) != len(input_labels):
            raise ValueError(
                f"points and labels length mismatch: "
                f"{len(input_points)} points, {len(input_labels)} labels"
            )

        masks, scores, logits = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            box=input_box,
            multimask_output=multimask_output,
        )

        result = SAMResult(masks=masks, scores=scores, logits=logits)

        print(
            "MobileSAM2Detector segment_by_points_and_box output:",
            {
                "masks_shape": masks.shape,
                "scores_shape": None if scores is None else scores.shape,
            },
        )

        return result

    def segment_everything(self) -> List[Dict[str, Any]]:
        self._check_image_loaded()

        print("MobileSAM2Detector segment_everything input: none")

        masks = SamAutomaticMaskGenerator(
            self.model,
            **(self.config.automatic_mask_kwargs or {}),
        ).generate(self.image_rgb)

        print(
            "MobileSAM2Detector segment_everything output:",
            {"mask_count": len(masks)},
        )

        return masks

    def _segment_everything_result(self) -> SAMResult:
        masks = self.segment_everything()

        mask_arrays: list[np.ndarray] = []

        for item in masks:
            raw_mask = extract_auto_mask(item)

            if raw_mask is not None:
                mask_arrays.append(raw_mask)

        if not mask_arrays:
            raise ValueError(
                "MobileSAM2Detector segment_everything did not return segmentation masks."
            )

        stacked = np.stack(mask_arrays, axis=0)

        return SAMResult(masks=stacked)

    def get_best_mask(self, result: SAMResult) -> np.ndarray:
        index = best_mask_index(result.scores)
        return result.masks[index]

    def save_mask(
        self,
        mask: np.ndarray,
        output_path: str,
        scale_to_255: bool = True,
    ) -> None:
        if scale_to_255:
            mask_to_save = to_mask_uint8(
                mask,
                threshold=self.config.mask_threshold,
            )
        else:
            mask_to_save = np.asarray(mask).astype(np.uint8)

        cv2.imwrite(output_path, mask_to_save)

    def overlay_mask(
        self,
        mask: np.ndarray,
        alpha: float = 0.5,
    ) -> np.ndarray:
        self._check_image_loaded()

        if self.image_bgr is None:
            raise RuntimeError("No BGR image available.")

        overlay = self.image_bgr.copy()

        mask_uint8 = to_mask_uint8(
            mask,
            threshold=self.config.mask_threshold,
        )

        colored_mask = np.zeros_like(overlay)
        colored_mask[mask_uint8 > 0] = (0, 255, 0)

        blended = cv2.addWeighted(
            overlay,
            1.0,
            colored_mask,
            alpha,
            0,
        )

        return blended

    def save_overlay(
        self,
        mask: np.ndarray,
        output_path: str,
        alpha: float = 0.5,
    ) -> None:
        overlay = self.overlay_mask(mask, alpha=alpha)
        cv2.imwrite(output_path, overlay)

    def _check_image_loaded(self) -> None:
        if self.image_rgb is None:
            raise RuntimeError("No image loaded. Call set_image() first.")

        if self.predictor is None:
            raise RuntimeError("MobileSAM predictor is not loaded.")

    def _run_prompts(self, prompts: dict[str, Any]) -> SAMResult:
        boxes = prompt_value(prompts, "boxes") if self.config.use_boxes else None
        points = prompt_value(prompts, "points") if self.config.use_points else None
        point_labels = prompt_value(prompts, "point_labels", "labels")

        boxes_array = normalize_boxes(boxes)
        points_array = normalize_points(points)

        labels_array = None

        if points_array is not None:
            labels_array = normalize_point_labels(
                point_labels=point_labels,
                point_count=len(points_array),
            )

        if boxes_array is not None and len(boxes_array) > 0:
            masks: list[np.ndarray] = []
            scores: list[float] = []

            for box in boxes_array:
                box_tuple = tuple(int(v) for v in box.tolist())

                box_points = None
                box_labels = None

                if points_array is not None and labels_array is not None:
                    inside = points_inside_box(points_array, box)

                    if np.any(inside):
                        box_points = points_array[inside]
                        box_labels = labels_array[inside]

                if box_points is not None and box_labels is not None:
                    result = self.segment_by_points_and_box(
                        points=box_points,
                        labels=box_labels,
                        box=box_tuple,
                        multimask_output=False,
                    )
                else:
                    result = self.segment_by_box(
                        box=box_tuple,
                        multimask_output=False,
                    )

                index = best_mask_index(result.scores)
                masks.append(result.masks[index])

                if result.scores is not None:
                    scores.append(float(result.scores[index]))
                else:
                    scores.append(1.0)

            if self.image_rgb is None:
                raise RuntimeError("No image loaded before merging masks.")

            merged_mask = merge_masks(
                masks=masks,
                image_shape=self.image_rgb.shape[:2],
            )

            return SAMResult(
                masks=np.expand_dims(merged_mask, axis=0),
                scores=np.asarray(
                    [max(scores) if scores else 1.0],
                    dtype=np.float32,
                ),
                logits=None,
            )

        if points_array is not None and len(points_array) > 0:
            return self.segment_by_points(
                points=points_array,
                labels=labels_array,
                multimask_output=False,
            )

        raise ValueError("No supported prompts found for MobileSAM2Detector.")

    def _empty_result(
        self,
        image: np.ndarray,
        reason: str,
        context: dict[str, Any] | None = None,
    ) -> MobileSAM2DetectorResult:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)

        return MobileSAM2DetectorResult(
            mask=mask,
            boxes=[],
            scores=[],
            labels=[],
            detections=[],
            metadata={
                "detector": self.name,
                "enabled": self.config.enabled,
                "empty": True,
                "reason": reason,
                "mask_area": 0,
                "context": context or {},
            },
        )

    def _log_detect_input(
        self,
        image: np.ndarray,
        prompts: dict[str, Any],
    ) -> None:
        print(
            "MobileSAM2Detector input:",
            {
                "image_shape": image.shape if image is not None else None,
                "prompt_keys": list(prompts.keys()),
                "prompts": summarize_prompts(prompts),
            },
        )

    def _log_detect_output(self, result: MobileSAM2DetectorResult) -> None:
        print(
            "MobileSAM2Detector output:",
            {
                "enabled": result.metadata.get("enabled"),
                "reason": result.metadata.get("reason"),
                "box_count": len(result.boxes),
                "mask_area": result.metadata.get("mask_area"),
                "prompt_keys": result.metadata.get("prompt_keys"),
            },
        )