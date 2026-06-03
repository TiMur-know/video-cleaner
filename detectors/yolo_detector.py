# detectors/yolo_detector.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from detectors.utils import (
    BBox,
    bbox_to_mask,
    crop_mask_to_bbox,
    dilate_mask,
    empty_mask,
    lazy_import,
    merge_masks,
    pad_bbox,
    resize_mask_nearest,
    to_mask_uint8,
    to_numpy,
    validate_image,
)


@dataclass(slots=True)
class YOLODetection:
    bbox: BBox
    confidence: float
    class_id: int
    class_name: str
    mask: np.ndarray
    label: str = "yolo_watermark"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class YOLODetectorResult:
    mask: np.ndarray
    detections: list[YOLODetection] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class YOLODetectorConfig:
    enabled: bool = True

    model_path: str = "models/yolo/watermarks_s_yolov8_v1.pt"

    device: str | int | None = None

    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    image_size: int = 640

    # None means accept all classes.
    class_ids: list[int] | None = None

    # If model is segmentation YOLO, use masks.
    # If model is detection-only YOLO, use bbox masks.
    use_segmentation_masks: bool = True

    box_padding: int = 4
    dilate_iterations: int = 1
    morph_kernel_size: int = 5

    return_debug: bool = False


class YOLODetector:
    """
    YOLO-based watermark detector.

    Good for:
        - logo watermarks
        - known watermark patterns
        - trained watermark classes

    Backend:
        ultralytics

    Install:
        pip install ultralytics
    """

    name = "yolo"

    def __init__(
        self,
        config: YOLODetectorConfig | None = None,
        model: Any | None = None,
    ) -> None:
        self.config = config or YOLODetectorConfig()
        self.model = model

    def detect(
        self,
        image: np.ndarray,
        context: dict[str, Any] | None = None,
    ) -> YOLODetectorResult:
        if not self.config.enabled:
            return YOLODetectorResult(mask=empty_mask(image))

        validate_image(image, self.name)

        if self.model is None:
            self.model = self._load_model()

        raw_results = self._run_model(image)

        detections = self._parse_results(
            raw_results,
            image_shape=image.shape[:2],
        )

        full_mask = merge_masks(
            [detection.mask for detection in detections],
            image_shape=image.shape[:2],
        )

        if self.config.dilate_iterations > 0:
            full_mask = dilate_mask(
                full_mask,
                iterations=self.config.dilate_iterations,
                kernel_size=self.config.morph_kernel_size,
            )

        metadata: dict[str, Any] = {
            "detector": self.name,
            "model_path": self.config.model_path,
            "num_detections": len(detections),
        }

        if self.config.return_debug:
            metadata["raw_results"] = raw_results

        return YOLODetectorResult(
            mask=full_mask,
            detections=detections,
            metadata=metadata,
        )

    def _load_model(self) -> Any:
        ultralytics = lazy_import("ultralytics")
        return ultralytics.YOLO(self.config.model_path)

    def _run_model(self, image: np.ndarray) -> Any:
        kwargs: dict[str, Any] = {
            "conf": self.config.confidence_threshold,
            "iou": self.config.iou_threshold,
            "imgsz": self.config.image_size,
            "verbose": False,
        }

        if self.config.device is not None:
            kwargs["device"] = self.config.device

        if self.config.class_ids is not None:
            kwargs["classes"] = self.config.class_ids

        return self.model.predict(
            source=image,
            **kwargs,
        )

    def _parse_results(
        self,
        results: Any,
        image_shape: tuple[int, int],
    ) -> list[YOLODetection]:
        detections: list[YOLODetection] = []

        if not results:
            return detections

        for result in results:
            detections.extend(
                self._parse_single_result(
                    result,
                    image_shape=image_shape,
                )
            )

        return detections

    def _parse_single_result(
        self,
        result: Any,
        image_shape: tuple[int, int],
    ) -> list[YOLODetection]:
        boxes = getattr(result, "boxes", None)

        if boxes is None:
            return []

        names = getattr(result, "names", {}) or {}
        masks_object = getattr(result, "masks", None)

        xyxy = (
            to_numpy(boxes.xyxy)
            if hasattr(boxes, "xyxy")
            else np.empty((0, 4))
        )

        confs = (
            to_numpy(boxes.conf)
            if hasattr(boxes, "conf")
            else np.ones((len(xyxy),))
        )

        classes = (
            to_numpy(boxes.cls)
            if hasattr(boxes, "cls")
            else np.zeros((len(xyxy),))
        )

        segmentation_masks = self._extract_segmentation_masks(
            masks_object,
            image_shape=image_shape,
        )

        detections: list[YOLODetection] = []

        for idx, box in enumerate(xyxy):
            confidence = float(confs[idx])

            if confidence < self.config.confidence_threshold:
                continue

            class_id = int(classes[idx])

            if self.config.class_ids is not None and class_id not in self.config.class_ids:
                continue

            bbox = pad_bbox(
                bbox=(
                    int(round(float(box[0]))),
                    int(round(float(box[1]))),
                    int(round(float(box[2]))),
                    int(round(float(box[3]))),
                ),
                image_shape=image_shape,
                padding=self.config.box_padding,
            )

            mask_source = "bbox"

            if (
                self.config.use_segmentation_masks
                and segmentation_masks is not None
                and idx < len(segmentation_masks)
            ):
                mask = segmentation_masks[idx]

                if mask.shape[:2] != image_shape:
                    mask = resize_mask_nearest(mask, image_shape)

                mask = crop_mask_to_bbox(mask, bbox)
                mask_source = "segmentation"
            else:
                mask = bbox_to_mask(bbox, image_shape)

            class_name = str(names.get(class_id, class_id))

            detections.append(
                YOLODetection(
                    bbox=bbox,
                    confidence=confidence,
                    class_id=class_id,
                    class_name=class_name,
                    mask=mask,
                    metadata={
                        "source": mask_source,
                    },
                )
            )

        return detections

    def _extract_segmentation_masks(
        self,
        masks_object: Any,
        image_shape: tuple[int, int],
    ) -> np.ndarray | None:
        if masks_object is None:
            return None

        if not self.config.use_segmentation_masks:
            return None

        data = getattr(masks_object, "data", None)

        if data is None:
            return None

        masks = to_numpy(data)

        if masks.ndim == 2:
            masks = masks[None, ...]

        if masks.ndim != 3:
            return None

        output_masks: list[np.ndarray] = []

        for mask in masks:
            mask_uint8 = to_mask_uint8(mask)

            if mask_uint8.shape[:2] != image_shape:
                mask_uint8 = resize_mask_nearest(mask_uint8, image_shape)

            output_masks.append(mask_uint8)

        return np.stack(output_masks, axis=0)