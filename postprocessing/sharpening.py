# postprocessing/sharpening.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from postprocessing.utils import (
    blend_masked,
    clip_like_input,
    ensure_odd,
    lazy_import,
    validate_image,
)


SharpenMethod = Literal["unsharp", "laplacian"]


@dataclass(slots=True)
class SharpeningConfig:
    enabled: bool = True

    method: SharpenMethod = "unsharp"

    strength: float = 0.4

    blur_kernel_size: int = 5
    blur_sigma: float = 1.0

    # If True and mask is provided, sharpen only repaired area.
    masked_only: bool = True


class SharpeningPostprocessor:
    """
    Sharpen final repaired image/frame.

    Usually used after:
        color matching
        seam blending
        artifact removal
    """

    name = "sharpening"

    def __init__(self, config: SharpeningConfig | None = None) -> None:
        self.config = config or SharpeningConfig()

    def __call__(
        self,
        image: np.ndarray,
        mask: np.ndarray | None = None,
        context: dict[str, Any] | None = None,
    ) -> np.ndarray:
        return self.apply(image, mask=mask, context=context)

    def apply(
        self,
        image: np.ndarray,
        mask: np.ndarray | None = None,
        context: dict[str, Any] | None = None,
    ) -> np.ndarray:
        if not self.config.enabled:
            return image

        validate_image(image, self.name)

        if self.config.method == "unsharp":
            sharpened = self._unsharp(image)

        elif self.config.method == "laplacian":
            sharpened = self._laplacian(image)

        else:
            raise ValueError(f"Unsupported sharpening method: {self.config.method}")

        if mask is not None and self.config.masked_only:
            return blend_masked(
                original=image,
                processed=sharpened,
                mask=mask,
            )

        return sharpened

    def _unsharp(self, image: np.ndarray) -> np.ndarray:
        cv2 = lazy_import("cv2")

        kernel_size = ensure_odd(self.config.blur_kernel_size)

        blurred = cv2.GaussianBlur(
            image,
            ksize=(kernel_size, kernel_size),
            sigmaX=self.config.blur_sigma,
        )

        output = cv2.addWeighted(
            image,
            1.0 + self.config.strength,
            blurred,
            -self.config.strength,
            0,
        )

        return clip_like_input(output, image)

    def _laplacian(self, image: np.ndarray) -> np.ndarray:
        cv2 = lazy_import("cv2")

        if image.ndim == 2:
            laplacian = cv2.Laplacian(image, cv2.CV_16S, ksize=3)
            laplacian = cv2.convertScaleAbs(laplacian)

            output = cv2.addWeighted(
                image,
                1.0,
                laplacian,
                self.config.strength,
                0,
            )

            return clip_like_input(output, image)

        channels = []

        for idx in range(image.shape[-1]):
            channel = image[..., idx]

            laplacian = cv2.Laplacian(channel, cv2.CV_16S, ksize=3)
            laplacian = cv2.convertScaleAbs(laplacian)

            enhanced = cv2.addWeighted(
                channel,
                1.0,
                laplacian,
                self.config.strength,
                0,
            )

            channels.append(enhanced)

        output = np.stack(channels, axis=-1)

        return clip_like_input(output, image)


def sharpen(
    image: np.ndarray,
    mask: np.ndarray | None = None,
    strength: float = 0.4,
) -> np.ndarray:
    processor = SharpeningPostprocessor(
        SharpeningConfig(
            strength=strength,
        )
    )

    return processor(image, mask=mask)