# detectors/ocr_detector.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import importlib
import numpy as np


_MODULE_CACHE: dict[str, Any] = {}


def lazy_import(module_name: str) -> Any:
    if module_name not in _MODULE_CACHE:
        try:
            _MODULE_CACHE[module_name] = importlib.import_module(module_name)
        except ImportError as exc:
            raise ImportError(
                f"Missing optional dependency '{module_name}'."
            ) from exc

    return _MODULE_CACHE[module_name]


OCRBackend = Literal["auto", "pytesseract", "easyocr"]


BBox = tuple[int, int, int, int]


@dataclass(slots=True)
class OCRDetection:
    bbox: BBox
    text: str
    confidence: float
    mask: np.ndarray
    label: str = "ocr_text_watermark"


@dataclass(slots=True)
class OCRDetectorResult:
    mask: np.ndarray
    detections: list[OCRDetection] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OCRDetectorConfig:
    enabled: bool = True

    backend: OCRBackend = "auto"

    # OCR language.
    language: str = "eng"

    # Minimum confidence from OCR engine.
    min_confidence: float = 0.35

    # Expand OCR boxes slightly for inpainting.
    box_padding: int = 4

    # Useful for watermark text.
    detect_empty_text: bool = False

    # EasyOCR options.
    easyocr_gpu: bool = False

    # Tesseract config.
    tesseract_config: str = "--psm 6"

    # Morphological cleanup of final text mask.
    apply_morphology: bool = True
    morph_kernel_size: int = 3

    return_debug: bool = False


class OCRDetector:
    """
    OCR-based watermark detector.

    Purpose:
        Detect text-like watermarks.

    Good for:
        - usernames
        - copyright text
        - stock image watermarks
        - repeated text overlays

    Backends:
        - pytesseract
        - easyocr

    Output:
        OCRDetectorResult containing:
            - binary mask
            - text detections
            - bboxes
    """

    name = "ocr"

    def __init__(self, config: OCRDetectorConfig | None = None) -> None:
        self.config = config or OCRDetectorConfig()
        self._easyocr_reader: Any | None = None

    def detect(
        self,
        image: np.ndarray,
        context: dict[str, Any] | None = None,
    ) -> OCRDetectorResult:
        if not self.config.enabled:
            empty_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            return OCRDetectorResult(mask=empty_mask)

        self._validate_image(image)

        backend = self._resolve_backend()

        if backend == "pytesseract":
            return self._detect_pytesseract(image)

        if backend == "easyocr":
            return self._detect_easyocr(image)

        raise ValueError(f"Unsupported OCR backend: {backend}")

    def _resolve_backend(self) -> str:
        if self.config.backend != "auto":
            return self.config.backend

        try:
            lazy_import("pytesseract")
            return "pytesseract"
        except ImportError:
            pass

        try:
            lazy_import("easyocr")
            return "easyocr"
        except ImportError:
            pass

        raise ImportError(
            "No OCR backend found. Install one of:\n"
            "  pip install pytesseract\n"
            "  pip install easyocr\n\n"
            "For pytesseract you also need the Tesseract binary installed."
        )

    def _detect_pytesseract(self, image: np.ndarray) -> OCRDetectorResult:
        pytesseract = lazy_import("pytesseract")

        cv2 = lazy_import("cv2")

        rgb_image = self._to_rgb(image)

        data = pytesseract.image_to_data(
            rgb_image,
            lang=self.config.language,
            config=self.config.tesseract_config,
            output_type=pytesseract.Output.DICT,
        )

        height, width = image.shape[:2]
        full_mask = np.zeros((height, width), dtype=np.uint8)

        detections: list[OCRDetection] = []

        total_items = len(data.get("text", []))

        for idx in range(total_items):
            text = str(data["text"][idx]).strip()

            if not text and not self.config.detect_empty_text:
                continue

            confidence = self._parse_tesseract_confidence(data["conf"][idx])

            if confidence < self.config.min_confidence:
                continue

            x = int(data["left"][idx])
            y = int(data["top"][idx])
            w = int(data["width"][idx])
            h = int(data["height"][idx])

            bbox = self._pad_bbox(
                bbox=(x, y, x + w, y + h),
                image_shape=(height, width),
            )

            component_mask = np.zeros((height, width), dtype=np.uint8)

            x1, y1, x2, y2 = bbox
            component_mask[y1:y2, x1:x2] = 255
            full_mask[y1:y2, x1:x2] = 255

            detections.append(
                OCRDetection(
                    bbox=bbox,
                    text=text,
                    confidence=confidence,
                    mask=component_mask,
                )
            )

        if self.config.apply_morphology:
            full_mask = self._clean_mask(full_mask)

        metadata: dict[str, Any] = {
            "detector": self.name,
            "backend": "pytesseract",
            "num_detections": len(detections),
        }

        if self.config.return_debug:
            metadata["raw_ocr_data"] = data

        return OCRDetectorResult(
            mask=full_mask,
            detections=detections,
            metadata=metadata,
        )

    def _detect_easyocr(self, image: np.ndarray) -> OCRDetectorResult:
        cv2 = lazy_import("cv2")

        reader = self._get_easyocr_reader()

        rgb_image = self._to_rgb(image)

        raw_results = reader.readtext(rgb_image)

        height, width = image.shape[:2]
        full_mask = np.zeros((height, width), dtype=np.uint8)

        detections: list[OCRDetection] = []

        for item in raw_results:
            # EasyOCR format:
            # [bbox_points, text, confidence]
            bbox_points, text, confidence = item

            confidence = float(confidence)

            if confidence < self.config.min_confidence:
                continue

            text = str(text).strip()

            if not text and not self.config.detect_empty_text:
                continue

            bbox = polygon_to_bbox(bbox_points)
            bbox = self._pad_bbox(bbox, image_shape=(height, width))

            component_mask = np.zeros((height, width), dtype=np.uint8)

            points = np.array(bbox_points, dtype=np.int32)

            cv2.fillPoly(component_mask, [points], 255)

            x1, y1, x2, y2 = bbox
            full_mask[y1:y2, x1:x2] = np.maximum(
                full_mask[y1:y2, x1:x2],
                component_mask[y1:y2, x1:x2],
            )

            detections.append(
                OCRDetection(
                    bbox=bbox,
                    text=text,
                    confidence=confidence,
                    mask=component_mask,
                )
            )

        if self.config.apply_morphology:
            full_mask = self._clean_mask(full_mask)

        metadata: dict[str, Any] = {
            "detector": self.name,
            "backend": "easyocr",
            "num_detections": len(detections),
        }

        if self.config.return_debug:
            metadata["raw_ocr_data"] = raw_results

        return OCRDetectorResult(
            mask=full_mask,
            detections=detections,
            metadata=metadata,
        )

    def _get_easyocr_reader(self) -> Any:
        if self._easyocr_reader is None:
            easyocr = lazy_import("easyocr")

            self._easyocr_reader = easyocr.Reader(
                [self.config.language],
                gpu=self.config.easyocr_gpu,
            )

        return self._easyocr_reader

    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        cv2 = lazy_import("cv2")

        kernel_size = max(1, self.config.morph_kernel_size)

        kernel = np.ones(
            (kernel_size, kernel_size),
            dtype=np.uint8,
        )

        cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return cleaned

    def _pad_bbox(
        self,
        bbox: BBox,
        image_shape: tuple[int, int],
    ) -> BBox:
        height, width = image_shape
        padding = self.config.box_padding

        x1, y1, x2, y2 = bbox

        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(width, x2 + padding)
        y2 = min(height, y2 + padding)

        return int(x1), int(y1), int(x2), int(y2)

    @staticmethod
    def _parse_tesseract_confidence(value: Any) -> float:
        try:
            confidence = float(value)
        except ValueError:
            return 0.0

        if confidence < 0:
            return 0.0

        # Tesseract confidence is normally 0-100.
        if confidence > 1.0:
            confidence = confidence / 100.0

        return float(confidence)

    @staticmethod
    def _to_rgb(image: np.ndarray) -> np.ndarray:
        cv2 = lazy_import("cv2")

        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        if image.ndim == 3 and image.shape[-1] == 1:
            return cv2.cvtColor(image[..., 0], cv2.COLOR_GRAY2RGB)

        if image.ndim == 3 and image.shape[-1] == 3:
            # Project default can use OpenCV-loaded BGR images.
            # Convert BGR to RGB for OCR engines.
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if image.ndim == 3 and image.shape[-1] == 4:
            return cv2.cvtColor(image[..., :3], cv2.COLOR_BGR2RGB)

        raise ValueError(f"Unsupported image shape: {image.shape}")

    @staticmethod
    def _validate_image(image: np.ndarray) -> None:
        if image is None:
            raise ValueError("OCRDetector received image=None")

        if not isinstance(image, np.ndarray):
            raise TypeError(f"OCRDetector expected np.ndarray, got {type(image)}")

        if image.ndim not in (2, 3):
            raise ValueError(f"Unsupported image shape: {image.shape}")

        if image.ndim == 3 and image.shape[-1] not in (1, 3, 4):
            raise ValueError(
                f"Expected 1, 3, or 4 channels, got {image.shape[-1]}"
            )


def polygon_to_bbox(points: Any) -> BBox:
    points_np = np.asarray(points, dtype=np.float32)

    xs = points_np[:, 0]
    ys = points_np[:, 1]

    x1 = int(np.floor(xs.min()))
    y1 = int(np.floor(ys.min()))
    x2 = int(np.ceil(xs.max()))
    y2 = int(np.ceil(ys.max()))

    return x1, y1, x2, y2