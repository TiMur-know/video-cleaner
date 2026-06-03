# detectors/fft_detector.py

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


FFTDetectorMode = Literal["high_pass", "band_pass", "periodic"]


@dataclass(slots=True)
class FFTDetection:
    bbox: BBox
    confidence: float
    mask: np.ndarray
    label: str = "fft_watermark"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FFTDetectorResult:
    mask: np.ndarray
    detections: list[FFTDetection] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FFTDetectorConfig:
    enabled: bool = True

    mode: FFTDetectorMode = "high_pass"

    # Frequency filtering.
    high_pass_radius: int = 20
    low_radius: int = 10
    high_radius: int = 90

    # Saliency threshold.
    threshold_percentile: float = 96.0

    # Ignore tiny components.
    min_component_area: int = 30

    # Mask cleanup.
    apply_morphology: bool = True
    morph_kernel_size: int = 5
    dilate_iterations: int = 1

    # Color input style.
    input_color_order: Literal["bgr", "rgb"] = "bgr"

    return_debug: bool = False


class FFTDetector:
    """
    FFT-based watermark detector.

    Purpose:
        Detect faint, repeated, periodic, or frequency-structured watermarks.

    Good for:
        - repeated transparent logos
        - diagonal watermark patterns
        - tiled watermark patterns
        - subtle frequency overlays

    Usually used together with:
        preprocessing/fft_enhance.py
        fusion_detector.py
    """

    name = "fft"

    def __init__(self, config: FFTDetectorConfig | None = None) -> None:
        self.config = config or FFTDetectorConfig()

    def detect(
        self,
        image: np.ndarray,
        context: dict[str, Any] | None = None,
    ) -> FFTDetectorResult:
        if not self.config.enabled:
            return FFTDetectorResult(mask=empty_mask(image))

        validate_image(image, self.name)

        gray = to_gray_float(
            image,
            input_color_order=self.config.input_color_order,
        )

        saliency = self._compute_frequency_saliency(gray)

        mask = self._saliency_to_mask(saliency)

        detections = self._mask_to_detections(mask, saliency)

        metadata: dict[str, Any] = {
            "detector": self.name,
            "mode": self.config.mode,
            "num_detections": len(detections),
        }

        if self.config.return_debug:
            metadata["saliency"] = saliency

        return FFTDetectorResult(
            mask=mask,
            detections=detections,
            metadata=metadata,
        )

    def _compute_frequency_saliency(self, gray: np.ndarray) -> np.ndarray:
        spectrum = np.fft.fft2(gray)
        spectrum_shifted = np.fft.fftshift(spectrum)

        frequency_mask = self._build_frequency_mask(gray.shape)

        filtered_spectrum = spectrum_shifted * frequency_mask

        filtered = np.fft.ifftshift(filtered_spectrum)
        reconstructed = np.fft.ifft2(filtered)

        saliency = np.abs(reconstructed)

        return normalize01(saliency)

    def _build_frequency_mask(self, shape: tuple[int, int]) -> np.ndarray:
        height, width = shape

        y = np.arange(height) - height // 2
        x = np.arange(width) - width // 2

        xx, yy = np.meshgrid(x, y)
        radius = np.sqrt(xx**2 + yy**2)

        if self.config.mode == "high_pass":
            mask = np.ones_like(radius, dtype=np.float32)
            mask[radius < self.config.high_pass_radius] = 0.0
            return mask

        if self.config.mode == "band_pass":
            mask = np.zeros_like(radius, dtype=np.float32)

            keep = (
                (radius >= self.config.low_radius)
                & (radius <= self.config.high_radius)
            )

            mask[keep] = 1.0
            return mask

        if self.config.mode == "periodic":
            radius_safe = np.maximum(radius, 1.0)

            mask = 1.0 - np.exp(
                -(radius_safe**2)
                / (2.0 * self.config.high_pass_radius**2)
            )

            return mask.astype(np.float32)

        raise ValueError(f"Unsupported FFT detector mode: {self.config.mode}")

    def _saliency_to_mask(self, saliency: np.ndarray) -> np.ndarray:
        threshold = np.percentile(
            saliency,
            self.config.threshold_percentile,
        )

        mask = (saliency >= threshold).astype(np.uint8) * 255

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
        saliency: np.ndarray,
    ) -> list[FFTDetection]:
        cv2 = lazy_import("cv2")

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )

        detections: list[FFTDetection] = []

        for label_id in range(1, num_labels):
            x = int(stats[label_id, cv2.CC_STAT_LEFT])
            y = int(stats[label_id, cv2.CC_STAT_TOP])
            w = int(stats[label_id, cv2.CC_STAT_WIDTH])
            h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
            area = int(stats[label_id, cv2.CC_STAT_AREA])

            if area < self.config.min_component_area:
                continue

            component_mask = (labels == label_id).astype(np.uint8) * 255
            component_saliency = saliency[labels == label_id]

            confidence = (
                float(np.mean(component_saliency))
                if component_saliency.size
                else 0.0
            )

            detections.append(
                FFTDetection(
                    bbox=(x, y, x + w, y + h),
                    confidence=confidence,
                    mask=component_mask,
                    metadata={
                        "area": area,
                    },
                )
            )

        return detections