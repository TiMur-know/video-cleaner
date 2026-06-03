# postprocessing/seam_blending.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from postprocessing.utils import (
    blend_with_alpha,
    lazy_import,
    soft_alpha_from_mask,
    validate_image,
    validate_mask,
)


@dataclass(slots=True)
class SeamBlendingConfig:
    enabled: bool = True

    # Expands mask around repaired area.
    dilate_iterations: int = 2

    # Blur size for soft transition.
    blur_kernel_size: int = 21

    # Blend amount.
    strength: float = 1.0


class SeamBlendingPostprocessor:
    """
    Softly blends inpainted region with surrounding pixels.

    Inputs:
        image:
            repaired/inpainted image

        mask:
            watermark/inpaint mask

        original_image:
            original image before inpainting

    If original_image is provided:
        blend original and repaired around the seam.

    If original_image is None:
        smooth seam area inside image itself.
    """

    name = "seam_blending"

    def __init__(self, config: SeamBlendingConfig | None = None) -> None:
        self.config = config or SeamBlendingConfig()

    def __call__(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        original_image: np.ndarray | None = None,
        context: dict[str, Any] | None = None,
    ) -> np.ndarray:
        return self.apply(
            image=image,
            mask=mask,
            original_image=original_image,
            context=context,
        )

    def apply(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        original_image: np.ndarray | None = None,
        context: dict[str, Any] | None = None,
    ) -> np.ndarray:
        if not self.config.enabled:
            return image

        validate_image(image, self.name)
        validate_mask(mask, image.shape[:2], self.name)

        alpha = soft_alpha_from_mask(
            mask,
            dilate_iterations=self.config.dilate_iterations,
            blur_kernel_size=self.config.blur_kernel_size,
        )

        alpha = alpha * float(np.clip(self.config.strength, 0.0, 1.0))

        if original_image is not None:
            validate_image(original_image, self.name)

            if original_image.shape != image.shape:
                raise ValueError(
                    f"{self.name} original_image shape {original_image.shape} "
                    f"must match image shape {image.shape}"
                )

            return blend_with_alpha(
                original=original_image,
                processed=image,
                alpha=alpha,
            )

        return self._smooth_seam(
            image=image,
            alpha=alpha,
        )

    def _smooth_seam(
        self,
        image: np.ndarray,
        alpha: np.ndarray,
    ) -> np.ndarray:
        cv2 = lazy_import("cv2")

        blurred = cv2.GaussianBlur(
            image,
            ksize=(self.config.blur_kernel_size | 1, self.config.blur_kernel_size | 1),
            sigmaX=0,
        )

        return blend_with_alpha(
            original=image,
            processed=blurred,
            alpha=alpha,
        )


def seam_blend(
    image: np.ndarray,
    mask: np.ndarray,
    original_image: np.ndarray | None = None,
) -> np.ndarray:
    processor = SeamBlendingPostprocessor()
    return processor(image, mask=mask, original_image=original_image)