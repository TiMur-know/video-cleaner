# preprocessing/gamma.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from preprocessing.utils import restore_dtype, to_float01, validate_image


@dataclass(slots=True)
class GammaConfig:
    enabled: bool = True

    # gamma < 1.0 brightens image
    # gamma > 1.0 darkens image
    gamma: float = 1.0

    # Optional gain multiplier after gamma correction.
    gain: float = 1.0


class GammaPreprocessor:
    """
    Gamma correction preprocessor.

    Purpose:
        Adjust image brightness/contrast before detection.

    Useful before:
        OCR, YOLO, SAM2, anomaly detector.

    Example:
        gamma=0.8 -> brighter
        gamma=1.2 -> darker
    """

    name = "gamma"

    def __init__(self, config: GammaConfig | None = None) -> None:
        self.config = config or GammaConfig()

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

        if self.config.gamma <= 0:
            raise ValueError("gamma must be greater than 0")

        original_dtype = image.dtype
        image_float = to_float01(image)

        output = self.config.gain * np.power(image_float, self.config.gamma)
        output = np.clip(output, 0.0, 1.0)

        return restore_dtype(output, original_dtype)


def gamma_correct(
    image: np.ndarray,
    gamma: float = 1.0,
    gain: float = 1.0,
) -> np.ndarray:
    processor = GammaPreprocessor(
        GammaConfig(
            gamma=gamma,
            gain=gain,
        )
    )

    return processor(image)