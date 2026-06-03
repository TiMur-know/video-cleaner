# postprocessing/artifact_removal.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from postprocessing.utils import (
    blend_masked,
    dilate_mask,
    ensure_odd,
    lazy_import,
    validate_image,
)


ArtifactRemovalMethod = Literal["median", "bilateral", "morphological"]


@dataclass(slots=True)
class ArtifactRemovalConfig:
    enabled: bool = True

    method: ArtifactRemovalMethod = "median"

    kernel_size: int = 3

    bilateral_diameter: int = 5
    bilateral_sigma_color: float = 50.0
    bilateral_sigma_space: float = 50.0

    # Apply only near repaired region when mask is provided.
    masked_only: bool = True

    dilate_mask_iterations: int = 1


class ArtifactRemovalPostprocessor:
    """
    Removes small artifacts after inpainting.

    Useful for:
        small noisy spots
        blocky artifacts
        tiny inconsistent pixels
        rough inpainting texture near mask
    """

    name = "artifact_removal"

    def __init__(self, config: ArtifactRemovalConfig | None = None) -> None:
        self.config = config or ArtifactRemovalConfig()

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

        if self.config.method == "median":
            processed = self._median(image)

        elif self.config.method == "bilateral":
            processed = self._bilateral(image)

        elif self.config.method == "morphological":
            processed = self._morphological(image)

        else:
            raise ValueError(
                f"Unsupported artifact removal method: {self.config.method}"
            )

        if mask is not None and self.config.masked_only:
            mask_for_blend = mask

            if self.config.dilate_mask_iterations > 0:
                mask_for_blend = dilate_mask(
                    mask,
                    iterations=self.config.dilate_mask_iterations,
                    kernel_size=3,
                )

            return blend_masked(
                original=image,
                processed=processed,
                mask=mask_for_blend,
            )

        return processed

    def _median(self, image: np.ndarray) -> np.ndarray:
        cv2 = lazy_import("cv2")

        kernel_size = ensure_odd(self.config.kernel_size)

        return cv2.medianBlur(image, kernel_size)

    def _bilateral(self, image: np.ndarray) -> np.ndarray:
        cv2 = lazy_import("cv2")

        return cv2.bilateralFilter(
            image,
            d=self.config.bilateral_diameter,
            sigmaColor=self.config.bilateral_sigma_color,
            sigmaSpace=self.config.bilateral_sigma_space,
        )

    def _morphological(self, image: np.ndarray) -> np.ndarray:
        cv2 = lazy_import("cv2")

        kernel_size = ensure_odd(self.config.kernel_size)
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

        if image.ndim == 2:
            opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
            closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
            return closed

        channels = []

        for idx in range(image.shape[-1]):
            channel = image[..., idx]
            opened = cv2.morphologyEx(channel, cv2.MORPH_OPEN, kernel)
            closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
            channels.append(closed)

        return np.stack(channels, axis=-1).astype(image.dtype)


def remove_artifacts(
    image: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    processor = ArtifactRemovalPostprocessor()
    return processor(image, mask=mask)