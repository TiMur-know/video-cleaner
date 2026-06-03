# preprocessing/resize.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from preprocessing.utils import lazy_import, validate_image


BackendName = Literal["auto", "opencv", "pillow"]
InterpolationMode = Literal[
    "nearest",
    "linear",
    "cubic",
    "area",
    "lanczos",
]


@dataclass(slots=True)
class ResizeConfig:
    enabled: bool = True

    backend: BackendName = "auto"

    width: int | None = None
    height: int | None = None

    keep_aspect_ratio: bool = True

    # If True, image will not be enlarged beyond original size.
    only_downscale: bool = False

    interpolation: InterpolationMode = "linear"


class ResizePreprocessor:
    """
    Resize preprocessor.

    Purpose:
        Normalize image/frame size before detection.

    Useful before:
        All detectors.

    Notes:
        - If width and height are both None, image is returned unchanged.
        - If keep_aspect_ratio=True, one dimension can be inferred.
    """

    name = "resize"

    def __init__(self, config: ResizeConfig | None = None) -> None:
        self.config = config or ResizeConfig()

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

        target_size = self._compute_target_size(image)

        if target_size is None:
            return image

        target_width, target_height = target_size

        if target_width == image.shape[1] and target_height == image.shape[0]:
            return image

        backend = self._resolve_backend()

        if backend == "opencv":
            return self._resize_opencv(image, target_width, target_height)

        if backend == "pillow":
            return self._resize_pillow(image, target_width, target_height)

        raise ValueError(f"Unsupported resize backend: {backend}")

    def _compute_target_size(
        self,
        image: np.ndarray,
    ) -> tuple[int, int] | None:
        original_height, original_width = image.shape[:2]

        target_width = self.config.width
        target_height = self.config.height

        if target_width is None and target_height is None:
            return None

        if target_width is not None and target_width <= 0:
            raise ValueError("resize width must be greater than 0")

        if target_height is not None and target_height <= 0:
            raise ValueError("resize height must be greater than 0")

        if self.config.keep_aspect_ratio:
            if target_width is None:
                scale = target_height / original_height
                target_width = round(original_width * scale)

            elif target_height is None:
                scale = target_width / original_width
                target_height = round(original_height * scale)

            else:
                scale = min(
                    target_width / original_width,
                    target_height / original_height,
                )

                target_width = round(original_width * scale)
                target_height = round(original_height * scale)

        else:
            if target_width is None:
                target_width = original_width

            if target_height is None:
                target_height = original_height

        if self.config.only_downscale:
            if target_width > original_width or target_height > original_height:
                return original_width, original_height

        return int(target_width), int(target_height)

    def _resolve_backend(self) -> str:
        if self.config.backend != "auto":
            return self.config.backend

        try:
            lazy_import("cv2")
            return "opencv"
        except ImportError:
            pass

        try:
            lazy_import("PIL.Image")
            return "pillow"
        except ImportError:
            pass

        raise ImportError(
            "No resize backend found. Install one of:\n"
            "  pip install opencv-python\n"
            "  pip install pillow"
        )

    def _resize_opencv(
        self,
        image: np.ndarray,
        width: int,
        height: int,
    ) -> np.ndarray:
        cv2 = lazy_import("cv2")

        interpolation_map = {
            "nearest": cv2.INTER_NEAREST,
            "linear": cv2.INTER_LINEAR,
            "cubic": cv2.INTER_CUBIC,
            "area": cv2.INTER_AREA,
            "lanczos": cv2.INTER_LANCZOS4,
        }

        interpolation = interpolation_map[self.config.interpolation]

        return cv2.resize(
            image,
            dsize=(width, height),
            interpolation=interpolation,
        )

    def _resize_pillow(
        self,
        image: np.ndarray,
        width: int,
        height: int,
    ) -> np.ndarray:
        pil_image_module = lazy_import("PIL.Image")

        interpolation_map = {
            "nearest": pil_image_module.Resampling.NEAREST,
            "linear": pil_image_module.Resampling.BILINEAR,
            "cubic": pil_image_module.Resampling.BICUBIC,
            "area": pil_image_module.Resampling.BOX,
            "lanczos": pil_image_module.Resampling.LANCZOS,
        }

        interpolation = interpolation_map[self.config.interpolation]

        pil_image = pil_image_module.fromarray(image)
        resized = pil_image.resize((width, height), interpolation)

        return np.asarray(resized)


def resize(
    image: np.ndarray,
    width: int | None = None,
    height: int | None = None,
    keep_aspect_ratio: bool = True,
    only_downscale: bool = False,
    interpolation: InterpolationMode = "linear",
) -> np.ndarray:
    processor = ResizePreprocessor(
        ResizeConfig(
            width=width,
            height=height,
            keep_aspect_ratio=keep_aspect_ratio,
            only_downscale=only_downscale,
            interpolation=interpolation,
        )
    )

    return processor(image)