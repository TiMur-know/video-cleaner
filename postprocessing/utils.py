# postprocessing/utils.py

from __future__ import annotations

import importlib
from typing import Any

import numpy as np


_MODULE_CACHE: dict[str, Any] = {}


def lazy_import(module_name: str) -> Any:
    """
    Lazy import optional/heavy postprocessing dependencies.
    """
    if module_name not in _MODULE_CACHE:
        try:
            _MODULE_CACHE[module_name] = importlib.import_module(module_name)
        except ImportError as exc:
            raise ImportError(
                f"Missing optional dependency '{module_name}'."
            ) from exc

    return _MODULE_CACHE[module_name]


def validate_image(image: np.ndarray, module_name: str = "postprocessor") -> None:
    """
    Validate image/frame input.
    """
    if image is None:
        raise ValueError(f"{module_name} received image=None")

    if not isinstance(image, np.ndarray):
        raise TypeError(f"{module_name} expected np.ndarray, got {type(image)}")

    if image.ndim not in (2, 3):
        raise ValueError(
            f"{module_name} expected HxW or HxWxC image, got shape={image.shape}"
        )


def validate_mask(
    mask: np.ndarray,
    image_shape: tuple[int, int],
    module_name: str = "postprocessor",
) -> None:
    """
    Validate mask shape against image shape.
    """
    if mask is None:
        raise ValueError(f"{module_name} received mask=None")

    if not isinstance(mask, np.ndarray):
        raise TypeError(f"{module_name} expected mask as np.ndarray, got {type(mask)}")

    if mask.shape[:2] != image_shape:
        raise ValueError(
            f"{module_name} mask shape {mask.shape[:2]} does not match "
            f"image shape {image_shape}"
        )


def validate_image_pair(
    image: np.ndarray,
    reference_image: np.ndarray,
    module_name: str = "postprocessor",
) -> None:
    """
    Validate two images have the same shape.
    """
    validate_image(image, module_name)
    validate_image(reference_image, module_name)

    if image.shape != reference_image.shape:
        raise ValueError(
            f"{module_name} image shape {image.shape} must match "
            f"reference shape {reference_image.shape}"
        )


def validate_frames(frames: list[np.ndarray], module_name: str = "postprocessor") -> None:
    """
    Validate a list of same-shape video frames.
    """
    if not frames:
        return

    first_shape = frames[0].shape

    for idx, frame in enumerate(frames):
        validate_image(frame, f"{module_name}[frame={idx}]")

        if frame.shape != first_shape:
            raise ValueError(
                f"{module_name} all frames must have same shape. "
                f"Frame 0 has {first_shape}, frame {idx} has {frame.shape}"
            )


def ensure_odd(value: int) -> int:
    """
    Ensure positive odd kernel size.
    """
    if value < 1:
        return 1

    if value % 2 == 0:
        value += 1

    return value


def to_mask_uint8(mask: np.ndarray) -> np.ndarray:
    """
    Convert mask to uint8 binary mask with values 0 or 255.
    """
    if mask.ndim == 3:
        mask = mask[..., 0]

    if mask.dtype == np.bool_:
        return mask.astype(np.uint8) * 255

    if mask.dtype == np.uint8:
        if mask.max() <= 1:
            return mask * 255
        return mask

    if mask.max() <= 1.0:
        return (mask > 0.5).astype(np.uint8) * 255

    return (mask > 127).astype(np.uint8) * 255


def to_mask_bool(mask: np.ndarray) -> np.ndarray:
    """
    Convert mask to boolean mask.
    """
    if mask.ndim == 3:
        mask = mask[..., 0]

    if mask.dtype == np.bool_:
        return mask

    if mask.max() <= 1:
        return mask > 0

    return mask > 127


def to_mask_alpha(mask: np.ndarray) -> np.ndarray:
    """
    Convert mask to float alpha in [0, 1].
    """
    if mask.ndim == 3:
        mask = mask[..., 0]

    if mask.dtype == np.bool_:
        return mask.astype(np.float32)

    if mask.max() <= 1:
        return mask.astype(np.float32)

    return (mask.astype(np.float32) / 255.0).clip(0.0, 1.0)


def dilate_mask(
    mask: np.ndarray,
    iterations: int = 1,
    kernel_size: int = 3,
) -> np.ndarray:
    """
    Dilate binary mask.
    """
    if iterations <= 0:
        return to_mask_uint8(mask)

    cv2 = lazy_import("cv2")

    kernel = np.ones(
        (max(1, kernel_size), max(1, kernel_size)),
        dtype=np.uint8,
    )

    return cv2.dilate(
        to_mask_uint8(mask),
        kernel,
        iterations=iterations,
    )


def soft_alpha_from_mask(
    mask: np.ndarray,
    dilate_iterations: int = 2,
    blur_kernel_size: int = 21,
) -> np.ndarray:
    """
    Create soft alpha matte from binary mask.

    Used for:
        seam blending
        soft transitions
        local postprocessing around repaired region
    """
    cv2 = lazy_import("cv2")

    mask_uint8 = to_mask_uint8(mask)

    if dilate_iterations > 0:
        mask_uint8 = dilate_mask(
            mask_uint8,
            iterations=dilate_iterations,
            kernel_size=3,
        )

    kernel_size = ensure_odd(blur_kernel_size)

    alpha = cv2.GaussianBlur(
        mask_uint8.astype(np.float32) / 255.0,
        ksize=(kernel_size, kernel_size),
        sigmaX=0,
    )

    return np.clip(alpha, 0.0, 1.0)


def blend_with_alpha(
    original: np.ndarray,
    processed: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    """
    Blend two images using alpha.

    output = original * (1 - alpha) + processed * alpha
    """
    if original.shape != processed.shape:
        raise ValueError(
            f"original shape {original.shape} must match processed shape {processed.shape}"
        )

    if original.ndim == 3 and alpha.ndim == 2:
        alpha = alpha[..., None]

    output = original.astype(np.float32) * (1.0 - alpha)
    output += processed.astype(np.float32) * alpha

    return clip_like_input(output, original)


def blend_masked(
    original: np.ndarray,
    processed: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Blend processed image into original only where mask is active.
    """
    alpha = to_mask_alpha(mask)
    return blend_with_alpha(original, processed, alpha)


def clip_like_input(output: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Clip and cast output to match reference dtype.
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
def image_to_3_channels(image: np.ndarray) -> np.ndarray:
    """
    Convert HxW, HxWx1, HxWx3, or HxWx4 image to HxWx3.

    Drops alpha if present.
    """
    validate_image(image, "image_to_3_channels")

    if image.ndim == 2:
        return np.repeat(image[:, :, None], 3, axis=2)

    channels = image.shape[-1]

    if channels == 1:
        return np.repeat(image, 3, axis=2)

    if channels == 3:
        return image

    if channels == 4:
        return image[:, :, :3]

    raise ValueError(f"Unsupported channel count: {channels}")


def extract_alpha_channel(image: np.ndarray) -> np.ndarray | None:
    """
    Return alpha channel if image is HxWx4.
    """
    if image is None:
        return None

    if isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[-1] == 4:
        return image[:, :, 3]

    return None


def restore_alpha_channel(
    image: np.ndarray,
    alpha: np.ndarray | None,
) -> np.ndarray:
    """
    Restore alpha channel to a 3-channel image.
    """
    if alpha is None:
        return image

    image_3 = image_to_3_channels(image)

    if alpha.shape[:2] != image_3.shape[:2]:
        raise ValueError(
            f"Alpha shape {alpha.shape} does not match image shape {image_3.shape}"
        )

    return np.dstack([image_3, alpha])


def prepare_image_pair_for_color_matching(
    image: np.ndarray,
    reference_image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Prepare repaired image and reference image for color matching.

    Allows:
        image: HxWx3
        reference_image: HxWx4

    Matching is done on 3 channels only.
    Reference alpha is returned so it can be restored later.
    """
    validate_image(image, "color_matching")
    validate_image(reference_image, "color_matching")

    reference_alpha = extract_alpha_channel(reference_image)

    image_3 = image_to_3_channels(image)
    reference_3 = image_to_3_channels(reference_image)

    if image_3.shape != reference_3.shape:
        raise ValueError(
            f"color_matching image shape {image_3.shape} must match "
            f"reference shape {reference_3.shape}"
        )

    return image_3, reference_3, reference_alpha
