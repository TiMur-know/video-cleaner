# preprocessing/utils.py

from __future__ import annotations

import importlib
from typing import Any, Callable

import numpy as np


_MODULE_CACHE: dict[str, Any] = {}


def lazy_import(module_name: str) -> Any:
    """
    Lazy import optional/heavy dependencies.

    Example:
        cv2 = lazy_import("cv2")
        exposure = lazy_import("skimage.exposure")
    """
    if module_name not in _MODULE_CACHE:
        try:
            _MODULE_CACHE[module_name] = importlib.import_module(module_name)
        except ImportError as exc:
            raise ImportError(
                f"Missing optional dependency '{module_name}'. "
                f"Install it before using this preprocessing backend."
            ) from exc

    return _MODULE_CACHE[module_name]


def validate_image(image: np.ndarray, module_name: str = "preprocessor") -> None:
    """
    Validate image input.

    Supported shapes:
        H x W
        H x W x 1
        H x W x 3
        H x W x 4
    """
    if image is None:
        raise ValueError(f"{module_name} received image=None")

    if not isinstance(image, np.ndarray):
        raise TypeError(
            f"{module_name} expected np.ndarray, got {type(image)}"
        )

    if image.ndim not in (2, 3):
        raise ValueError(
            f"{module_name} expected image with shape HxW or HxWxC, "
            f"got shape={image.shape}"
        )

    if image.ndim == 3 and image.shape[-1] not in (1, 3, 4):
        raise ValueError(
            f"{module_name} expected 1, 3, or 4 channels, "
            f"got {image.shape[-1]}"
        )


def ensure_odd_kernel(kernel_size: int) -> int:
    """
    Ensure kernel size is positive and odd.

    OpenCV blur kernels usually require odd values.
    """
    if kernel_size < 1:
        raise ValueError("kernel_size must be >= 1")

    if kernel_size % 2 == 0:
        kernel_size += 1

    return kernel_size


def split_alpha(image: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Split RGBA image into RGB + alpha.

    If image has no alpha channel, alpha is None.
    """
    validate_image(image, "split_alpha")

    if image.ndim == 3 and image.shape[-1] == 4:
        return image[..., :3], image[..., 3:]

    return image, None


def merge_alpha(image: np.ndarray, alpha: np.ndarray | None) -> np.ndarray:
    """
    Merge color image with alpha channel if alpha exists.
    """
    if alpha is None:
        return image

    return np.concatenate([image, alpha], axis=-1)


def to_uint8(
    image: np.ndarray,
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    """
    Convert image to uint8 and return a restore function.

    Handles:
        uint8 images
        float images in [0, 1]
        float images in [0, 255]
        integer images
    """
    original_dtype = image.dtype

    if original_dtype == np.uint8:
        return image.copy(), lambda output: output

    if np.issubdtype(original_dtype, np.floating):
        max_value = float(np.nanmax(image)) if image.size else 1.0

        if max_value <= 1.0:
            image_uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)

            def restore(output: np.ndarray) -> np.ndarray:
                return (output.astype(np.float32) / 255.0).astype(original_dtype)

            return image_uint8, restore

        image_uint8 = np.clip(image, 0, 255).astype(np.uint8)

        def restore(output: np.ndarray) -> np.ndarray:
            return output.astype(original_dtype)

        return image_uint8, restore

    if np.issubdtype(original_dtype, np.integer):
        info = np.iinfo(original_dtype)

        image_float = image.astype(np.float32)
        image_float = (image_float - info.min) / max(info.max - info.min, 1)

        image_uint8 = np.clip(image_float * 255.0, 0, 255).astype(np.uint8)

        def restore(output: np.ndarray) -> np.ndarray:
            restored = output.astype(np.float32) / 255.0
            restored = restored * (info.max - info.min) + info.min
            return np.clip(restored, info.min, info.max).round().astype(original_dtype)

        return image_uint8, restore

    raise TypeError(f"Unsupported image dtype: {original_dtype}")


def to_float01(image: np.ndarray) -> np.ndarray:
    """
    Convert image to float32 in range [0, 1].

    Used by:
        gamma.py
        fft_enhance.py
        skimage-based preprocessors
    """
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0

    if np.issubdtype(image.dtype, np.floating):
        max_value = float(np.nanmax(image)) if image.size else 1.0

        if max_value <= 1.0:
            return np.clip(image.astype(np.float32), 0.0, 1.0)

        return np.clip(image.astype(np.float32) / 255.0, 0.0, 1.0)

    if np.issubdtype(image.dtype, np.integer):
        info = np.iinfo(image.dtype)

        image_float = image.astype(np.float32)
        image_float = (image_float - info.min) / max(info.max - info.min, 1)

        return np.clip(image_float, 0.0, 1.0)

    raise TypeError(f"Unsupported image dtype: {image.dtype}")


def restore_dtype(image_float: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """
    Restore float image in [0, 1] back to original dtype.
    """
    image_float = np.clip(image_float, 0.0, 1.0)

    if dtype == np.uint8:
        return (image_float * 255.0).round().astype(np.uint8)

    if np.issubdtype(dtype, np.floating):
        return image_float.astype(dtype)

    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)

        restored = image_float * (info.max - info.min) + info.min

        return np.clip(restored, info.min, info.max).round().astype(dtype)

    raise TypeError(f"Unsupported image dtype: {dtype}")


def normalize01(image: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Normalize any numeric image to [0, 1].

    Useful for visualization or FFT outputs.
    """
    image = image.astype(np.float32)

    min_value = float(np.min(image))
    max_value = float(np.max(image))

    if max_value - min_value < eps:
        return np.zeros_like(image, dtype=np.float32)

    return ((image - min_value) / (max_value - min_value)).astype(np.float32)


def clip_like_input(output: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Clip and cast output to match reference image dtype.
    """
    dtype = reference.dtype

    if dtype == np.uint8:
        return np.clip(output, 0, 255).round().astype(np.uint8)

    if np.issubdtype(dtype, np.floating):
        return np.clip(output, 0.0, 1.0).astype(dtype)

    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.clip(output, info.min, info.max).round().astype(dtype)

    raise TypeError(f"Unsupported image dtype: {dtype}")