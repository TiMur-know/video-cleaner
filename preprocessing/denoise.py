# preprocessing/denoise.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from preprocessing.utils import (
    ensure_odd_kernel,
    lazy_import,
    restore_dtype,
    to_float01,
    to_uint8,
    validate_image,
)


BackendName = Literal["auto", "opencv", "skimage"]
DenoiseMethod = Literal[
    "auto",
    "gaussian",
    "median",
    "bilateral",
    "nl_means",
    "tv_chambolle",
]


@dataclass(slots=True)
class DenoiseConfig:
    enabled: bool = True

    # "auto" tries OpenCV first, then skimage.
    backend: BackendName = "auto"

    # "auto" uses bilateral for OpenCV and tv_chambolle for skimage.
    method: DenoiseMethod = "auto"

    kernel_size: int = 5
    gaussian_sigma: float = 1.0

    bilateral_diameter: int = 7
    bilateral_sigma_color: float = 50.0
    bilateral_sigma_space: float = 50.0

    nl_means_h: float = 7.0
    nl_means_h_color: float = 7.0
    nl_means_template_window_size: int = 7
    nl_means_search_window_size: int = 21

    tv_weight: float = 0.08


class DenoisePreprocessor:
    """
    Generic denoise preprocessor.

    Purpose:
        Reduce noise before detection or enhancement.

    Used before:
        CLAHE, OCR, YOLO, SAM2, FFT detector, anomaly detector.

    Lazy loading:
        OpenCV or skimage is imported only when apply() is called.
    """

    name = "denoise"

    def __init__(self, config: DenoiseConfig | None = None) -> None:
        self.config = config or DenoiseConfig()

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
        method = self._resolve_method(backend)

        if backend == "opencv":
            return self._apply_opencv(image, method)

        if backend == "skimage":
            return self._apply_skimage(image, method)

        raise ValueError(f"Unsupported denoise backend: {backend}")

    def _resolve_backend(self) -> str:
        if self.config.backend != "auto":
            return self.config.backend

        try:
            lazy_import("cv2")
            return "opencv"
        except ImportError:
            pass

        try:
            lazy_import("skimage.restoration")
            return "skimage"
        except ImportError:
            pass

        raise ImportError(
            "No denoise backend found. Install one of:\n"
            "  pip install opencv-python\n"
            "  pip install scikit-image"
        )

    def _resolve_method(self, backend: str) -> str:
        if self.config.method != "auto":
            return self.config.method

        if backend == "opencv":
            return "bilateral"

        if backend == "skimage":
            return "tv_chambolle"

        raise ValueError(f"Cannot resolve method for backend: {backend}")

    def _apply_opencv(self, image: np.ndarray, method: str) -> np.ndarray:
        if method == "gaussian":
            return self._opencv_gaussian(image)

        if method == "median":
            return self._opencv_median(image)

        if method == "bilateral":
            return self._opencv_bilateral(image)

        if method == "nl_means":
            return self._opencv_nl_means(image)

        raise ValueError(f"OpenCV denoise does not support method: {method}")

    def _opencv_gaussian(self, image: np.ndarray) -> np.ndarray:
        cv2 = lazy_import("cv2")

        if image.ndim == 3 and image.shape[-1] == 4:
            color = image[..., :3]
            alpha = image[..., 3:]
            output = self._opencv_gaussian(color)
            return np.concatenate([output, alpha], axis=-1)

        image_uint8, restore = to_uint8(image)
        kernel_size = ensure_odd_kernel(self.config.kernel_size)

        output = cv2.GaussianBlur(
            image_uint8,
            ksize=(kernel_size, kernel_size),
            sigmaX=self.config.gaussian_sigma,
        )

        return restore(output)

    def _opencv_median(self, image: np.ndarray) -> np.ndarray:
        cv2 = lazy_import("cv2")

        if image.ndim == 3 and image.shape[-1] == 4:
            color = image[..., :3]
            alpha = image[..., 3:]
            output = self._opencv_median(color)
            return np.concatenate([output, alpha], axis=-1)

        image_uint8, restore = to_uint8(image)
        kernel_size = ensure_odd_kernel(self.config.kernel_size)

        output = cv2.medianBlur(image_uint8, kernel_size)

        return restore(output)

    def _opencv_bilateral(self, image: np.ndarray) -> np.ndarray:
        cv2 = lazy_import("cv2")

        if image.ndim == 3 and image.shape[-1] == 4:
            color = image[..., :3]
            alpha = image[..., 3:]
            output = self._opencv_bilateral(color)
            return np.concatenate([output, alpha], axis=-1)

        image_uint8, restore = to_uint8(image)

        output = cv2.bilateralFilter(
            image_uint8,
            d=self.config.bilateral_diameter,
            sigmaColor=self.config.bilateral_sigma_color,
            sigmaSpace=self.config.bilateral_sigma_space,
        )

        return restore(output)

    def _opencv_nl_means(self, image: np.ndarray) -> np.ndarray:
        cv2 = lazy_import("cv2")

        if image.ndim == 3 and image.shape[-1] == 4:
            color = image[..., :3]
            alpha = image[..., 3:]
            output = self._opencv_nl_means(color)
            return np.concatenate([output, alpha], axis=-1)

        image_uint8, restore = to_uint8(image)

        if image_uint8.ndim == 2:
            output = cv2.fastNlMeansDenoising(
                image_uint8,
                None,
                h=self.config.nl_means_h,
                templateWindowSize=self.config.nl_means_template_window_size,
                searchWindowSize=self.config.nl_means_search_window_size,
            )

            return restore(output)

        if image_uint8.shape[-1] == 1:
            gray = image_uint8[..., 0]

            output = cv2.fastNlMeansDenoising(
                gray,
                None,
                h=self.config.nl_means_h,
                templateWindowSize=self.config.nl_means_template_window_size,
                searchWindowSize=self.config.nl_means_search_window_size,
            )

            return restore(output[..., None])

        output = cv2.fastNlMeansDenoisingColored(
            image_uint8,
            None,
            h=self.config.nl_means_h,
            hColor=self.config.nl_means_h_color,
            templateWindowSize=self.config.nl_means_template_window_size,
            searchWindowSize=self.config.nl_means_search_window_size,
        )

        return restore(output)

    def _apply_skimage(self, image: np.ndarray, method: str) -> np.ndarray:
        if image.ndim == 3 and image.shape[-1] == 4:
            color = image[..., :3]
            alpha = image[..., 3:]
            output = self._apply_skimage(color, method)
            return np.concatenate([output, alpha], axis=-1)

        if method == "gaussian":
            return self._skimage_gaussian(image)

        if method == "tv_chambolle":
            return self._skimage_tv_chambolle(image)

        if method == "nl_means":
            return self._skimage_nl_means(image)

        raise ValueError(f"skimage denoise does not support method: {method}")

    def _skimage_gaussian(self, image: np.ndarray) -> np.ndarray:
        filters = lazy_import("skimage.filters")

        original_dtype = image.dtype
        image_float = to_float01(image)

        output = filters.gaussian(
            image_float,
            sigma=self.config.gaussian_sigma,
            channel_axis=-1 if image_float.ndim == 3 else None,
            preserve_range=True,
        )

        return restore_dtype(output, original_dtype)

    def _skimage_tv_chambolle(self, image: np.ndarray) -> np.ndarray:
        restoration = lazy_import("skimage.restoration")

        original_dtype = image.dtype
        image_float = to_float01(image)

        output = restoration.denoise_tv_chambolle(
            image_float,
            weight=self.config.tv_weight,
            channel_axis=-1 if image_float.ndim == 3 else None,
        )

        return restore_dtype(output, original_dtype)

    def _skimage_nl_means(self, image: np.ndarray) -> np.ndarray:
        restoration = lazy_import("skimage.restoration")

        original_dtype = image.dtype
        image_float = to_float01(image)

        output = restoration.denoise_nl_means(
            image_float,
            h=0.08,
            fast_mode=True,
            channel_axis=-1 if image_float.ndim == 3 else None,
            preserve_range=True,
        )

        return restore_dtype(output, original_dtype)


def denoise(
    image: np.ndarray,
    backend: BackendName = "auto",
    method: DenoiseMethod = "auto",
) -> np.ndarray:
    processor = DenoisePreprocessor(
        DenoiseConfig(
            backend=backend,
            method=method,
        )
    )

    return processor(image)