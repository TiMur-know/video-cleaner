# preprocessing/clahe.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from preprocessing.utils import (
    lazy_import,
    restore_dtype,
    to_float01,
    to_uint8,
    validate_image,
)


BackendName = Literal["auto", "opencv", "skimage"]
ColorSpace = Literal["lab", "ycrcb", "channel"]
ColorOrder = Literal["rgb", "bgr"]


@dataclass(slots=True)
class CLAHEConfig:
    enabled: bool = True

    # "auto" tries OpenCV first, then skimage.
    backend: BackendName = "auto"

    clip_limit: float = 2.0
    tile_grid_size: tuple[int, int] = (8, 8)

    # "lab" is recommended for normal color images.
    color_space: ColorSpace = "lab"

    # Use "bgr" if image comes directly from cv2.imread().
    input_color_order: ColorOrder = "bgr"


class CLAHEPreprocessor:
    """
    Contrast Limited Adaptive Histogram Equalization.

    Purpose:
        Improve local contrast before detectors.

    Used before:
        OCR, YOLO, SAM2, FFT detector, anomaly detector.

    Lazy loading:
        OpenCV or skimage is imported only when apply() is called.
    """

    name = "clahe"

    def __init__(self, config: CLAHEConfig | None = None) -> None:
        self.config = config or CLAHEConfig()
        self._opencv_clahe: Any | None = None

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

        if backend == "skimage":
            return self._apply_skimage(image)

        raise ValueError(f"Unsupported CLAHE backend: {backend}")

    def _resolve_backend(self) -> str:
        if self.config.backend != "auto":
            return self.config.backend

        try:
            lazy_import("cv2")
            return "opencv"
        except ImportError:
            pass

        try:
            lazy_import("skimage.exposure")
            return "skimage"
        except ImportError:
            pass

        raise ImportError(
            "No CLAHE backend found. Install one of:\n"
            "  pip install opencv-python\n"
            "  pip install scikit-image"
        )

    def _get_opencv_clahe(self) -> Any:
        cv2 = lazy_import("cv2")

        if self._opencv_clahe is None:
            self._opencv_clahe = cv2.createCLAHE(
                clipLimit=self.config.clip_limit,
                tileGridSize=self.config.tile_grid_size,
            )

        return self._opencv_clahe

    def _apply_opencv_gray(self, image: np.ndarray) -> np.ndarray:
        image_uint8, restore = to_uint8(image)
        output = self._get_opencv_clahe().apply(image_uint8)
        return restore(output)

    def _apply_opencv(self, image: np.ndarray) -> np.ndarray:
        cv2 = lazy_import("cv2")

        if image.ndim == 2:
            return self._apply_opencv_gray(image)

        if image.shape[-1] == 1:
            output = self._apply_opencv_gray(image[..., 0])
            return output[..., None]

        if image.shape[-1] == 4:
            color = image[..., :3]
            alpha = image[..., 3:]
            output = self._apply_opencv(color)
            return np.concatenate([output, alpha], axis=-1)

        if self.config.color_space == "channel":
            image_uint8, restore = to_uint8(image)

            channels = [
                self._get_opencv_clahe().apply(image_uint8[..., idx])
                for idx in range(image_uint8.shape[-1])
            ]

            output = np.stack(channels, axis=-1)
            return restore(output)

        image_uint8, restore = to_uint8(image)

        if self.config.color_space == "lab":
            if self.config.input_color_order == "rgb":
                converted = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2LAB)
                back_code = cv2.COLOR_LAB2RGB
            else:
                converted = cv2.cvtColor(image_uint8, cv2.COLOR_BGR2LAB)
                back_code = cv2.COLOR_LAB2BGR

            converted[..., 0] = self._get_opencv_clahe().apply(converted[..., 0])

            output = cv2.cvtColor(converted, back_code)
            return restore(output)

        if self.config.color_space == "ycrcb":
            if self.config.input_color_order == "rgb":
                converted = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2YCrCb)
                back_code = cv2.COLOR_YCrCb2RGB
            else:
                converted = cv2.cvtColor(image_uint8, cv2.COLOR_BGR2YCrCb)
                back_code = cv2.COLOR_YCrCb2BGR

            converted[..., 0] = self._get_opencv_clahe().apply(converted[..., 0])

            output = cv2.cvtColor(converted, back_code)
            return restore(output)

        raise ValueError(f"Unsupported CLAHE color_space: {self.config.color_space}")

    def _apply_skimage(self, image: np.ndarray) -> np.ndarray:
        exposure = lazy_import("skimage.exposure")

        original_dtype = image.dtype

        if image.ndim == 3 and image.shape[-1] == 4:
            color = image[..., :3]
            alpha = image[..., 3:]

            output = self._apply_skimage(color)

            return np.concatenate([output, alpha], axis=-1)

        image_float = to_float01(image)

        # skimage clip_limit uses a smaller scale than OpenCV.
        clip_limit = min(max(self.config.clip_limit / 100.0, 0.001), 1.0)

        output = exposure.equalize_adapthist(
            image_float,
            kernel_size=self.config.tile_grid_size,
            clip_limit=clip_limit,
        )

        return restore_dtype(output, original_dtype)


def clahe(
    image: np.ndarray,
    backend: BackendName = "auto",
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
    color_space: ColorSpace = "lab",
    input_color_order: ColorOrder = "bgr",
) -> np.ndarray:
    processor = CLAHEPreprocessor(
        CLAHEConfig(
            backend=backend,
            clip_limit=clip_limit,
            tile_grid_size=tile_grid_size,
            color_space=color_space,
            input_color_order=input_color_order,
        )
    )

    return processor(image)