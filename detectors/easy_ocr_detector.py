# detectors/easy_ocr_detector.py

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
    polygon_to_bbox,
    to_rgb,
    validate_image,
)


ColorOrder = Literal["bgr", "rgb"]


@dataclass(slots=True)
class EasyOCRDetection:
    bbox: BBox
    text: str
    confidence: float
    mask: np.ndarray
    label: str = "easyocr_text_watermark"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EasyOCRDetectorResult:
    mask: np.ndarray
    detections: list[EasyOCRDetection] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EasyOCRDetectorConfig:
    enabled: bool = True

    # EasyOCR uses language codes like:
    # "en", "fr", "de", "es", "ru", etc.
    languages: tuple[str, ...] = ("en",)

    input_color_order: ColorOrder = "bgr"

    gpu: bool = False

    min_confidence: float = 0.35

    # Expand text boxes for safer inpainting.
    box_padding: int = 4

    # Keep results even when OCR text is empty.
    detect_empty_text: bool = False

    apply_morphology: bool = True
    morph_kernel_size: int = 3

    return_debug: bool = False


class EasyOCRDetector:
    """
    EasyOCR-based text watermark detector.

    Purpose:
        Detect text-like watermarks.

    Good for:
        - usernames
        - copyright text
        - stock image text overlays
        - repeated text watermarks

    Install:
        pip install easyocr
    """

    name = "easy_ocr"

    def __init__(
        self,
        config: EasyOCRDetectorConfig | None = None,
        reader: Any | None = None,
    ) -> None:
        self.config = config or EasyOCRDetectorConfig()
        self.reader = reader

    def detect(
        self,
        image: np.ndarray,
        context: dict[str, Any] | None = None,
    ) -> EasyOCRDetectorResult:
        if not self.config.enabled:
            return EasyOCRDetectorResult(mask=empty_mask(image))

        validate_image(image, self.name)

        if self.reader is None:
            self.reader = self._load_reader()

        rgb_image = to_rgb(
            image,
            input_color_order=self.config.input_color_order,
        )

        raw_results = self.reader.readtext(rgb_image)

        height, width = image.shape[:2]

        full_mask = np.zeros((height, width), dtype=np.uint8)
        detections: list[EasyOCRDetection] = []

        for raw_item in raw_results:
            # EasyOCR output:
            # [
            #   [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
            #   "text",
            #   confidence
            # ]
            bbox_points, text, confidence = raw_item

            text = str(text).strip()
            confidence = float(confidence)

            if confidence < self.config.min_confidence:
                continue

            if not text and not self.config.detect_empty_text:
                continue

            bbox = polygon_to_bbox(bbox_points)
            bbox = pad_bbox(
                bbox=bbox,
                image_shape=(height, width),
                padding=self.config.box_padding,
            )

            component_mask = self._polygon_to_mask(
                points=bbox_points,
                image_shape=(height, width),
            )

            x1, y1, x2, y2 = bbox
            full_mask[y1:y2, x1:x2] = np.maximum(
                full_mask[y1:y2, x1:x2],
                component_mask[y1:y2, x1:x2],
            )

            detections.append(
                EasyOCRDetection(
                    bbox=bbox,
                    text=text,
                    confidence=confidence,
                    mask=component_mask,
                    metadata={
                        "raw_points": bbox_points,
                    },
                )
            )

        if self.config.apply_morphology:
            full_mask = clean_binary_mask(
                full_mask,
                kernel_size=self.config.morph_kernel_size,
                close=True,
                open_=False,
                dilate_iterations=0,
            )

        metadata: dict[str, Any] = {
            "detector": self.name,
            "backend": "easyocr",
            "languages": list(self.config.languages),
            "num_detections": len(detections),
        }

        if self.config.return_debug:
            metadata["raw_ocr_data"] = raw_results

        return EasyOCRDetectorResult(
            mask=full_mask,
            detections=detections,
            metadata=metadata,
        )

    def _load_reader(self) -> Any:
        easyocr = lazy_import("easyocr")

        return easyocr.Reader(
            list(self.config.languages),
            gpu=self.config.gpu,
        )

    @staticmethod
    def _polygon_to_mask(
        points: Any,
        image_shape: tuple[int, int],
    ) -> np.ndarray:
        cv2 = lazy_import("cv2")

        height, width = image_shape

        mask = np.zeros((height, width), dtype=np.uint8)

        points_np = np.asarray(points, dtype=np.int32)
        cv2.fillPoly(mask, [points_np], 255)

        return mask