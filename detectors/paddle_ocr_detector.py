# detectors/paddle_ocr_detector.py

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
MaskMode = Literal["bbox", "polygon"]


@dataclass(slots=True)
class PaddleOCRDetection:
    bbox: BBox
    text: str
    confidence: float
    mask: np.ndarray
    label: str = "paddleocr_text_watermark"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PaddleOCRDetectorResult:
    mask: np.ndarray
    detections: list[PaddleOCRDetection] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PaddleOCRDetectorConfig:
    enabled: bool = True

    # PaddleOCR language examples:
    # "en", "ch", "fr", "german", "japan", "korean"
    language: str = "en"

    input_color_order: ColorOrder = "bgr"

    use_angle_cls: bool = True
    use_gpu: bool = False

    min_confidence: float = 0.35
    box_padding: int = 4

    # "bbox" is safer for inpainting.
    # "polygon" is tighter.
    mask_mode: MaskMode = "bbox"

    detect_empty_text: bool = False

    apply_morphology: bool = True
    morph_kernel_size: int = 3
    dilate_iterations: int = 0

    return_debug: bool = False


class PaddleOCRDetector:
    """
    PaddleOCR-based text watermark detector.

    Purpose:
        Detect text-like watermarks.

    Good for:
        - copyright text
        - usernames
        - stock image watermarks
        - multilingual text watermarks

    Install:
        pip install paddleocr paddlepaddle
    """

    name = "paddle_ocr"

    def __init__(
        self,
        config: PaddleOCRDetectorConfig | None = None,
        ocr_model: Any | None = None,
    ) -> None:
        self.config = config or PaddleOCRDetectorConfig()
        self.ocr_model = ocr_model

    def detect(
        self,
        image: np.ndarray,
        context: dict[str, Any] | None = None,
    ) -> PaddleOCRDetectorResult:
        if not self.config.enabled:
            return PaddleOCRDetectorResult(mask=empty_mask(image))

        validate_image(image, self.name)

        if self.ocr_model is None:
            self.ocr_model = self._load_model()

        rgb_image = to_rgb(
            image,
            input_color_order=self.config.input_color_order,
        )

        raw_results = self._run_ocr(rgb_image)

        height, width = image.shape[:2]
        full_mask = np.zeros((height, width), dtype=np.uint8)

        detections: list[PaddleOCRDetection] = []

        parsed_items = self._parse_raw_results(raw_results)

        for item in parsed_items:
            points = item["points"]
            text = item["text"]
            confidence = item["confidence"]

            if confidence < self.config.min_confidence:
                continue

            if not text and not self.config.detect_empty_text:
                continue

            bbox = polygon_to_bbox(points)
            bbox = pad_bbox(
                bbox=bbox,
                image_shape=(height, width),
                padding=self.config.box_padding,
            )

            component_mask = self._make_component_mask(
                points=points,
                bbox=bbox,
                image_shape=(height, width),
            )

            full_mask = np.maximum(full_mask, component_mask)

            detections.append(
                PaddleOCRDetection(
                    bbox=bbox,
                    text=text,
                    confidence=confidence,
                    mask=component_mask,
                    metadata={
                        "raw_points": points,
                    },
                )
            )

        if self.config.apply_morphology or self.config.dilate_iterations > 0:
            full_mask = clean_binary_mask(
                full_mask,
                kernel_size=self.config.morph_kernel_size,
                close=True,
                open_=False,
                dilate_iterations=self.config.dilate_iterations,
            )

        metadata: dict[str, Any] = {
            "detector": self.name,
            "backend": "paddleocr",
            "language": self.config.language,
            "num_detections": len(detections),
        }

        if self.config.return_debug:
            metadata["raw_ocr_data"] = raw_results

        return PaddleOCRDetectorResult(
            mask=full_mask,
            detections=detections,
            metadata=metadata,
        )

    def _load_model(self) -> Any:
        paddleocr = lazy_import("paddleocr")
        PaddleOCR = paddleocr.PaddleOCR

        return PaddleOCR(
            use_angle_cls=self.config.use_angle_cls,
            lang=self.config.language,
            use_gpu=self.config.use_gpu,
        )

    def _run_ocr(self, rgb_image: np.ndarray) -> Any:
        """
        PaddleOCR classic API usually supports:
            ocr_model.ocr(image, cls=True)

        Some newer wrappers may expose:
            ocr_model.predict(image)
        """
        if hasattr(self.ocr_model, "ocr"):
            return self.ocr_model.ocr(
                rgb_image,
                cls=self.config.use_angle_cls,
            )

        if hasattr(self.ocr_model, "predict"):
            return self.ocr_model.predict(rgb_image)

        if callable(self.ocr_model):
            return self.ocr_model(rgb_image)

        raise TypeError(
            "Unsupported PaddleOCR model interface. Expected "
            "ocr(image), predict(image), or callable model."
        )

    def _parse_raw_results(self, raw_results: Any) -> list[dict[str, Any]]:
        """
        Normalize PaddleOCR outputs.

        Common classic format:
            [
                [
                    [points, (text, confidence)],
                    ...
                ]
            ]

        Sometimes:
            [
                [points, (text, confidence)],
                ...
            ]
        """
        items: list[dict[str, Any]] = []

        if raw_results is None:
            return items

        # Some PaddleOCR versions wrap results per image.
        if (
            isinstance(raw_results, list)
            and len(raw_results) == 1
            and isinstance(raw_results[0], list)
        ):
            maybe_first = raw_results[0]

            if self._looks_like_line(maybe_first):
                lines = raw_results
            else:
                lines = maybe_first
        else:
            lines = raw_results

        if not isinstance(lines, list):
            return items

        for line in lines:
            parsed = self._parse_line(line)

            if parsed is not None:
                items.append(parsed)

        return items

    @staticmethod
    def _looks_like_line(value: Any) -> bool:
        """
        Checks whether value looks like:
            [points, (text, confidence)]
        """
        if not isinstance(value, list):
            return False

        if len(value) < 2:
            return False

        second = value[1]

        return isinstance(second, (tuple, list)) and len(second) >= 2

    def _parse_line(self, line: Any) -> dict[str, Any] | None:
        if not isinstance(line, (list, tuple)):
            return None

        if len(line) < 2:
            return None

        points = line[0]
        text_info = line[1]

        if not isinstance(text_info, (list, tuple)) or len(text_info) < 2:
            return None

        text = str(text_info[0]).strip()

        try:
            confidence = float(text_info[1])
        except (TypeError, ValueError):
            confidence = 0.0

        points_np = np.asarray(points, dtype=np.float32)

        if points_np.ndim != 2 or points_np.shape[-1] != 2:
            return None

        return {
            "points": points_np,
            "text": text,
            "confidence": confidence,
        }

    def _make_component_mask(
        self,
        points: Any,
        bbox: BBox,
        image_shape: tuple[int, int],
    ) -> np.ndarray:
        height, width = image_shape
        mask = np.zeros((height, width), dtype=np.uint8)

        if self.config.mask_mode == "bbox":
            x1, y1, x2, y2 = bbox
            mask[y1:y2, x1:x2] = 255
            return mask

        if self.config.mask_mode == "polygon":
            cv2 = lazy_import("cv2")
            points_np = np.asarray(points, dtype=np.int32)
            cv2.fillPoly(mask, [points_np], 255)
            return mask

        raise ValueError(f"Unsupported mask_mode: {self.config.mask_mode}")