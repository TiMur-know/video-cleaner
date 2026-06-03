# preprocessing/edge_enhance.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from preprocessing.utils import (
    ensure_odd_kernel,
    lazy_import,
    to_uint8,
    validate_image,
)


BackendName = Literal["auto", "opencv"]
EdgeMethod = Literal["unsharp", "laplacian", "sobel"]


@dataclass(slots=True)
class EdgeEnhanceConfig:
    enabled: bool = True

    # For now, OpenCV is the main backend.
    backend: BackendName = "auto"

    # Recommended default: "unsharp"
    method: EdgeMethod = "unsharp"

    # General enhancement strength.
    strength: float = 1.0

    # Used by unsharp masking.
    blur_kernel_size: int = 5
    blur_sigma: float = 1.0

    # Used by laplacian / sobel.
    edge_weight: float = 0.5


class EdgeEnhancePreprocessor:
    """
    Edge enhancement preprocessor.

    Purpose:
        Make watermark borders, text edges, logos, and faint structures easier
        for detectors to find.

    Used before:
        OCR, YOLO, SAM2, FFT detector, anomaly detector.

    Recommended method:
        "unsharp" for general use.
    """

    name = "edge_enhance"

    def __init__(self, config: EdgeEnhanceConfig | None = None) -> None:
        self.config = config or EdgeEnhanceConfig()

    def __call__(
        self,
        image: np.ndarray,
        context: dict[str, Any] | None = None,
    ) -> np.ndarray:
        return self.apply(image, context=context)

    def apply(
        self,
        image: np.ndarray,
        context: dict[str, Any] | None = None,
    ) -> np.ndarray:
        if not self.config.enabled:
            return image

        validate_image(image, self.name)

        backend = self._resolve_backend()

        if backend == "opencv":
            return self._apply_opencv(image)

        raise ValueError(f"Unsupported edge enhancement backend: {backend}")

    def _resolve_backend(self) -> str:
        if self.config.backend != "auto":
            return self.config.backend

        try:
            lazy_import("cv2")
            return "opencv"
        except ImportError as exc:
            raise ImportError(
                "No edge enhancement backend found. Install OpenCV:\n"
                "  pip install opencv-python"
            ) from exc

    def _apply_opencv(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 3 and image.shape[-1] == 4:
            color = image[..., :3]
            alpha = image[..., 3:]
            output = self._apply_opencv(color)
            return np.concatenate([output, alpha], axis=-1)

        if self.config.method == "unsharp":
            return self._unsharp(image)

        if self.config.method == "laplacian":
            return self._laplacian(image)

        if self.config.method == "sobel":
            return self._sobel(image)

        raise ValueError(f"Unsupported edge enhancement method: {self.config.method}")

    def _unsharp(self, image: np.ndarray) -> np.ndarray:
        """
        Unsharp masking:

            enhanced = image + strength * (image - blurred)

        Good default for improving faint watermark edges.
        """
        cv2 = lazy_import("cv2")

        image_uint8, restore = to_uint8(image)

        kernel_size = ensure_odd_kernel(self.config.blur_kernel_size)

        blurred = cv2.GaussianBlur(
            image_uint8,
            ksize=(kernel_size, kernel_size),
            sigmaX=self.config.blur_sigma,
        )

        output = cv2.addWeighted(
            image_uint8,
            1.0 + self.config.strength,
            blurred,
            -self.config.strength,
            0,
        )

        return restore(output)

    def _laplacian(self, image: np.ndarray) -> np.ndarray:
        """
        Laplacian edge enhancement.

        Stronger than unsharp. Useful for text-like watermarks,
        but can amplify noise.
        """
        cv2 = lazy_import("cv2")

        image_uint8, restore = to_uint8(image)

        if image_uint8.ndim == 2:
            gray = image_uint8
            edges = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
            edges = cv2.convertScaleAbs(edges)

            output = cv2.addWeighted(
                image_uint8,
                1.0,
                edges,
                self.config.edge_weight * self.config.strength,
                0,
            )

            return restore(output)

        channels = []

        for idx in range(image_uint8.shape[-1]):
            channel = image_uint8[..., idx]
            edges = cv2.Laplacian(channel, cv2.CV_16S, ksize=3)
            edges = cv2.convertScaleAbs(edges)

            enhanced = cv2.addWeighted(
                channel,
                1.0,
                edges,
                self.config.edge_weight * self.config.strength,
                0,
            )

            channels.append(enhanced)

        output = np.stack(channels, axis=-1)
        return restore(output)

    def _sobel(self, image: np.ndarray) -> np.ndarray:
        """
        Sobel edge enhancement.

        Good for directional edges and structured watermarks.
        """
        cv2 = lazy_import("cv2")

        image_uint8, restore = to_uint8(image)

        if image_uint8.ndim == 2:
            edges = self._sobel_edges(image_uint8)

            output = cv2.addWeighted(
                image_uint8,
                1.0,
                edges,
                self.config.edge_weight * self.config.strength,
                0,
            )

            return restore(output)

        channels = []

        for idx in range(image_uint8.shape[-1]):
            channel = image_uint8[..., idx]
            edges = self._sobel_edges(channel)

            enhanced = cv2.addWeighted(
                channel,
                1.0,
                edges,
                self.config.edge_weight * self.config.strength,
                0,
            )

            channels.append(enhanced)

        output = np.stack(channels, axis=-1)
        return restore(output)

    @staticmethod
    def _sobel_edges(channel: np.ndarray) -> np.ndarray:
        cv2 = lazy_import("cv2")

        grad_x = cv2.Sobel(channel, cv2.CV_16S, 1, 0, ksize=3)
        grad_y = cv2.Sobel(channel, cv2.CV_16S, 0, 1, ksize=3)

        abs_x = cv2.convertScaleAbs(grad_x)
        abs_y = cv2.convertScaleAbs(grad_y)

        return cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)


def edge_enhance(
    image: np.ndarray,
    method: EdgeMethod = "unsharp",
    strength: float = 1.0,
) -> np.ndarray:
    processor = EdgeEnhancePreprocessor(
        EdgeEnhanceConfig(
            method=method,
            strength=strength,
        )
    )

    return processor(image)