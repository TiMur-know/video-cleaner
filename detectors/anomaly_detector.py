# detectors/anomaly_detector.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from detectors.utils import (
    BBox,
    clean_binary_mask,
    empty_mask,
    lazy_import,
    normalize01,
    to_gray_float,
    validate_image,
)


AnomalyMethod = Literal[
    "local_residual",
    "gradient",
    "laplacian",
]


@dataclass(slots=True)
class AnomalyDetection:
    bbox: BBox
    confidence: float
    mask: np.ndarray
    label: str = "anomaly_watermark"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AnomalyDetectorResult:
    mask: np.ndarray
    detections: list[AnomalyDetection] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AnomalyDetectorConfig:
    enabled: bool = True

    method: AnomalyMethod = "local_residual"

    input_color_order: Literal["bgr", "rgb"] = "bgr"

    # Used for local residual.
    blur_kernel_size: int = 31
    blur_sigma: float = 0.0

    # Threshold percentile of anomaly map.
    threshold_percentile: float = 97.0

    min_component_area: int = 30

    apply_morphology: bool = True
    morph_kernel_size: int = 5
    dilate_iterations: int = 1

    return_debug: bool = False


class AnomalyDetector:
    """
    Simple anomaly-based watermark detector.

    Purpose:
        Find suspicious local structures that differ from surrounding image.

    Good for:
        - faint transparent logos
        - subtle overlays
        - local brightness inconsistencies
        - text/logo remnants after preprocessing

    This is not a learned model.
    It is a lightweight classical detector.
    """

    name = "anomaly"

    def __init__(
        self,
        config: AnomalyDetectorConfig | None = None,
    ) -> None:
        self.config = config or AnomalyDetectorConfig()

    def detect(
        self,
        image: np.ndarray,
        context: dict[str, Any] | None = None,
    ) -> AnomalyDetectorResult:
        if not self.config.enabled:
            return AnomalyDetectorResult(mask=empty_mask(image))

        validate_image(image, self.name)

        gray = to_gray_float(
            image,
            input_color_order=self.config.input_color_order,
        )

        anomaly_map = self._compute_anomaly_map(gray)

        mask = self._anomaly_map_to_mask(anomaly_map)

        detections = self._mask_to_detections(mask, anomaly_map)

        metadata: dict[str, Any] = {
            "detector": self.name,
            "method": self.config.method,
            "num_detections": len(detections),
        }

        if self.config.return_debug:
            metadata["anomaly_map"] = anomaly_map

        return AnomalyDetectorResult(
            mask=mask,
            detections=detections,
            metadata=metadata,
        )

    def _compute_anomaly_map(self, gray: np.ndarray) -> np.ndarray:
        if self.config.method == "local_residual":
            return self._local_residual(gray)

        if self.config.method == "gradient":
            return self._gradient_anomaly(gray)

        if self.config.method == "laplacian":
            return self._laplacian_anomaly(gray)

        raise ValueError(f"Unsupported anomaly method: {self.config.method}")

    def _local_residual(self, gray: np.ndarray) -> np.ndarray:
        """
        Difference between image and blurred local background.

        Watermarks often appear as local residuals on smooth image regions.
        """
        cv2 = lazy_import("cv2")

        kernel_size = self.config.blur_kernel_size

        if kernel_size < 3:
            kernel_size = 3

        if kernel_size % 2 == 0:
            kernel_size += 1

        blurred = cv2.GaussianBlur(
            gray,
            ksize=(kernel_size, kernel_size),
            sigmaX=self.config.blur_sigma,
        )

        residual = np.abs(gray - blurred)

        return normalize01(residual)

    def _gradient_anomaly(self, gray: np.ndarray) -> np.ndarray:
        """
        Highlights sudden local changes.
        Good for text-like or logo edges.
        """
        cv2 = lazy_import("cv2")

        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

        magnitude = np.sqrt(grad_x**2 + grad_y**2)

        return normalize01(magnitude)

    def _laplacian_anomaly(self, gray: np.ndarray) -> np.ndarray:
        """
        Highlights high-frequency local structures.
        """
        cv2 = lazy_import("cv2")

        laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        anomaly = np.abs(laplian) if False else np.abs(laplacian)

        return normalize01(anomaly)

    def _anomaly_map_to_mask(self, anomaly_map: np.ndarray) -> np.ndarray:
        threshold = np.percentile(
            anomaly_map,
            self.config.threshold_percentile,
        )

        mask = (anomaly_map >= threshold).astype(np.uint8) * 255

        if self.config.apply_morphology or self.config.dilate_iterations > 0:
            mask = clean_binary_mask(
                mask,
                kernel_size=self.config.morph_kernel_size,
                close=True,
                open_=True,
                dilate_iterations=self.config.dilate_iterations,
            )

        return mask

    def _mask_to_detections(
        self,
        mask: np.ndarray,
        anomaly_map: np.ndarray,
    ) -> list[AnomalyDetection]:
        cv2 = lazy_import("cv2")

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )

        detections: list[AnomalyDetection] = []

        for label_id in range(1, num_labels):
            x = int(stats[label_id, cv2.CC_STAT_LEFT])
            y = int(stats[label_id, cv2.CC_STAT_TOP])
            w = int(stats[label_id, cv2.CC_STAT_WIDTH])
            h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
            area = int(stats[label_id, cv2.CC_STAT_AREA])

            if area < self.config.min_component_area:
                continue

            component_mask = (labels == label_id).astype(np.uint8) * 255
            component_values = anomaly_map[labels == label_id]

            confidence = (
                float(np.mean(component_values))
                if component_values.size
                else 0.0
            )

            detections.append(
                AnomalyDetection(
                    bbox=(x, y, x + w, y + h),
                    confidence=confidence,
                    mask=component_mask,
                    metadata={
                        "area": area,
                    },
                )
            )

        return detections