# preprocessing/patchify.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from preprocessing.utils import validate_image


@dataclass(slots=True)
class Patch:
    image: np.ndarray
    x: int
    y: int
    width: int
    height: int
    original_shape: tuple[int, ...]


@dataclass(slots=True)
class PatchifyConfig:
    enabled: bool = True

    patch_size: int = 512
    stride: int = 256

    # If True, pad image so edge patches are full patch_size.
    padding: bool = True

    # Used when padding=True.
    pad_value: int | float = 0

    # If True, return Patch objects with x/y coordinates.
    # If False, return only patch images.
    return_metadata: bool = True


class PatchifyPreprocessor:
    """
    Patchify preprocessor.

    Purpose:
        Split large images into smaller patches for detectors.

    Useful before:
        YOLO, SAM2, anomaly detector, OCR detector.

    Why:
        Some detectors work better or faster on fixed-size crops.
    """

    name = "patchify"

    def __init__(self, config: PatchifyConfig | None = None) -> None:
        self.config = config or PatchifyConfig()

    def __call__(
        self,
        image: np.ndarray,
        context: dict[str, Any] | None = None,
    ) -> list[Patch] | list[np.ndarray] | np.ndarray:
        return self.apply(image, context=context)

    def apply(
        self,
        image: np.ndarray,
        context: dict[str, Any] | None = None,
    ) -> list[Patch] | list[np.ndarray] | np.ndarray:
        if not self.config.enabled:
            return image

        validate_image(image, self.name)

        if self.config.patch_size <= 0:
            raise ValueError("patch_size must be greater than 0")

        if self.config.stride <= 0:
            raise ValueError("stride must be greater than 0")

        original_shape = image.shape

        padded_image = self._pad_image_if_needed(image)

        patches: list[Patch] | list[np.ndarray] = []

        height, width = padded_image.shape[:2]
        patch_size = self.config.patch_size
        stride = self.config.stride

        for y in range(0, height - patch_size + 1, stride):
            for x in range(0, width - patch_size + 1, stride):
                patch_image = padded_image[
                    y : y + patch_size,
                    x : x + patch_size,
                    ...,
                ]

                if self.config.return_metadata:
                    patches.append(
                        Patch(
                            image=patch_image,
                            x=x,
                            y=y,
                            width=patch_size,
                            height=patch_size,
                            original_shape=original_shape,
                        )
                    )
                else:
                    patches.append(patch_image)

        return patches

    def _pad_image_if_needed(self, image: np.ndarray) -> np.ndarray:
        if not self.config.padding:
            return image

        height, width = image.shape[:2]
        patch_size = self.config.patch_size
        stride = self.config.stride

        target_height = self._compute_padded_size(height, patch_size, stride)
        target_width = self._compute_padded_size(width, patch_size, stride)

        pad_bottom = target_height - height
        pad_right = target_width - width

        if pad_bottom == 0 and pad_right == 0:
            return image

        if image.ndim == 2:
            pad_width = (
                (0, pad_bottom),
                (0, pad_right),
            )
        else:
            pad_width = (
                (0, pad_bottom),
                (0, pad_right),
                (0, 0),
            )

        return np.pad(
            image,
            pad_width=pad_width,
            mode="constant",
            constant_values=self.config.pad_value,
        )

    @staticmethod
    def _compute_padded_size(
        size: int,
        patch_size: int,
        stride: int,
    ) -> int:
        if size <= patch_size:
            return patch_size

        remainder = (size - patch_size) % stride

        if remainder == 0:
            return size

        return size + (stride - remainder)


def patchify(
    image: np.ndarray,
    patch_size: int = 512,
    stride: int = 256,
    padding: bool = True,
    return_metadata: bool = True,
) -> list[Patch] | list[np.ndarray] | np.ndarray:
    processor = PatchifyPreprocessor(
        PatchifyConfig(
            patch_size=patch_size,
            stride=stride,
            padding=padding,
            return_metadata=return_metadata,
        )
    )

    return processor(image)