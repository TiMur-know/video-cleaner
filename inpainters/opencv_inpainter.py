# inpainters/opencv_inpainter.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from inpainters.utils import (
    lazy_import,
    prepare_mask,
    prepare_opencv_image,
    to_mask_uint8,
    validate_image,
    validate_mask,
)


OpenCVInpaintMethod = Literal["telea", "ns"]


@dataclass(slots=True)
class OpenCVInpaintResult:
    image: np.ndarray
    mask: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OpenCVInpainterConfig:
    enabled: bool = True

    method: OpenCVInpaintMethod = "telea"
    radius: float = 3.0

    # Expand mask before inpainting.
    dilate_mask_iterations: int = 1
    mask_kernel_size: int = 3


class OpenCVInpainter:
    """
    Fast classical inpainting using OpenCV.

    Best for:
        - small text watermarks
        - thin logos
        - simple backgrounds
        - fast video frame processing
    """

    name = "opencv"

    def __init__(self, config: OpenCVInpainterConfig | None = None) -> None:
        self.config = config or OpenCVInpainterConfig()

    def __call__(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        context: dict[str, Any] | None = None,
    ) -> OpenCVInpaintResult:
        return self.inpaint(image=image, mask=mask, context=context)

    def inpaint(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        context: dict[str, Any] | None = None,
    ) -> OpenCVInpaintResult:
        mask_uint8 = to_mask_uint8(mask)

        if not self.config.enabled:
            return OpenCVInpaintResult(
                image=image,
                mask=mask_uint8,
                metadata={
                    "enabled": False,
                    "inpainter": self.name,
                },
            )

        validate_image(image, self.name)
        validate_mask(mask, image.shape[:2], self.name)

        cv2 = lazy_import("cv2")

        prepared_mask = prepare_mask(
            mask,
            dilate_iterations=self.config.dilate_mask_iterations,
            kernel_size=self.config.mask_kernel_size,
        )

        prepared_image = prepare_opencv_image(image)

        output = cv2.inpaint(
            src=prepared_image,
            inpaintMask=prepared_mask,
            inpaintRadius=float(self.config.radius),
            flags=self._method_flag(),
        )

        output = output.astype(image.dtype)

        metadata = {
            "inpainter": self.name,
            "method": self.config.method,
            "radius": self.config.radius,
            "mask_area": int(np.count_nonzero(prepared_mask)),
        }

        return OpenCVInpaintResult(
            image=output,
            mask=prepared_mask,
            metadata=metadata,
        )

    def _method_flag(self) -> int:
        cv2 = lazy_import("cv2")

        if self.config.method == "telea":
            return cv2.INPAINT_TELEA

        if self.config.method == "ns":
            return cv2.INPAINT_NS

        raise ValueError(f"Unsupported OpenCV inpaint method: {self.config.method}")


def opencv_inpaint(
    image: np.ndarray,
    mask: np.ndarray,
    method: OpenCVInpaintMethod = "telea",
    radius: float = 3.0,
) -> np.ndarray:
    inpainter = OpenCVInpainter(
        OpenCVInpainterConfig(
            method=method,
            radius=radius,
        )
    )

    return inpainter.inpaint(image, mask).image