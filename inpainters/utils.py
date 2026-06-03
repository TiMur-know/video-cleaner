# inpainters/utils.py

from __future__ import annotations

import importlib
from typing import Any, Literal

import numpy as np


ColorOrder = Literal["bgr", "rgb"]

_MODULE_CACHE: dict[str, Any] = {}


def lazy_import(module_name: str) -> Any:
    """
    Lazy import optional inpainter dependencies.
    """
    if module_name not in _MODULE_CACHE:
        try:
            _MODULE_CACHE[module_name] = importlib.import_module(module_name)
        except ImportError as exc:
            raise ImportError(
                f"Missing optional dependency '{module_name}'."
            ) from exc

    return _MODULE_CACHE[module_name]


def validate_image(image: np.ndarray, module_name: str = "inpainter") -> None:
    if image is None:
        raise ValueError(f"{module_name} received image=None")

    if not isinstance(image, np.ndarray):
        raise TypeError(f"{module_name} expected image as np.ndarray, got {type(image)}")

    if image.ndim not in (2, 3):
        raise ValueError(f"{module_name} unsupported image shape: {image.shape}")


def validate_mask(
    mask: np.ndarray,
    image_shape: tuple[int, int],
    module_name: str = "inpainter",
) -> None:
    if mask is None:
        raise ValueError(f"{module_name} received mask=None")

    if not isinstance(mask, np.ndarray):
        raise TypeError(f"{module_name} expected mask as np.ndarray, got {type(mask)}")

    if mask.shape[:2] != image_shape:
        raise ValueError(
            f"{module_name} mask shape {mask.shape[:2]} does not match "
            f"image shape {image_shape}"
        )


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


def prepare_mask(
    mask: np.ndarray,
    dilate_iterations: int = 1,
    kernel_size: int = 5,
) -> np.ndarray:
    """
    Convert and optionally dilate mask before inpainting.
    """
    mask_uint8 = to_mask_uint8(mask)

    if dilate_iterations <= 0:
        return mask_uint8

    cv2 = lazy_import("cv2")

    kernel_size = max(1, kernel_size)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    return cv2.dilate(
        mask_uint8,
        kernel,
        iterations=dilate_iterations,
    )


def prepare_opencv_image(image: np.ndarray) -> np.ndarray:
    """
    Prepare image for cv2.inpaint.

    OpenCV inpaint supports:
        uint8 grayscale
        uint8 3-channel image
    """
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    if image.ndim == 3 and image.shape[-1] == 4:
        image = image[..., :3]

    if image.ndim == 2:
        return image

    if image.ndim == 3 and image.shape[-1] == 3:
        return image

    raise ValueError(f"Unsupported image shape for OpenCV inpaint: {image.shape}")


def image_to_pil_rgb(
    image: np.ndarray,
    input_color_order: ColorOrder = "bgr",
) -> Any:
    """
    Convert NumPy image to PIL RGB image.
    """
    pil_module = lazy_import("PIL.Image")

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    if image.ndim == 2:
        return pil_module.fromarray(image).convert("RGB")

    if image.ndim == 3 and image.shape[-1] == 4:
        image = image[..., :3]

    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    if input_color_order == "bgr":
        image = image[..., ::-1]

    return pil_module.fromarray(image).convert("RGB")


def mask_to_pil(
    mask: np.ndarray,
    dilate_iterations: int = 1,
    kernel_size: int = 5,
) -> Any:
    """
    Convert mask to PIL grayscale mask.
    """
    pil_module = lazy_import("PIL.Image")

    mask_uint8 = prepare_mask(
        mask,
        dilate_iterations=dilate_iterations,
        kernel_size=kernel_size,
    )

    return pil_module.fromarray(mask_uint8).convert("L")


def pil_to_numpy(
    image: Any,
    output_color_order: ColorOrder = "bgr",
) -> np.ndarray:
    """
    Convert PIL image to NumPy image.
    """
    output = np.asarray(image.convert("RGB"))

    if output_color_order == "bgr":
        output = output[..., ::-1]

    return output.astype(np.uint8)


def model_output_to_numpy(
    output: Any,
    output_color_order: ColorOrder = "bgr",
) -> np.ndarray:
    """
    Convert common model outputs to NumPy image.

    Supports:
        PIL image
        NumPy array
        dict with image/result/output
    """
    pil_module = lazy_import("PIL.Image")

    if isinstance(output, np.ndarray):
        image = output

    elif isinstance(output, pil_module.Image):
        image = np.asarray(output.convert("RGB"))

    elif isinstance(output, dict):
        for key in ("image", "result", "output"):
            if key in output:
                return model_output_to_numpy(
                    output[key],
                    output_color_order=output_color_order,
                )

        raise ValueError("Output dict does not contain image/result/output.")

    else:
        image = np.asarray(output)

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)

    if image.ndim == 3 and image.shape[-1] == 4:
        image = image[..., :3]

    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Unsupported model output shape: {image.shape}")

    if output_color_order == "bgr":
        image = image[..., ::-1]

    return image.astype(np.uint8)


def resize_pil_to_multiple(
    image: Any,
    multiple: int = 8,
) -> Any:
    """
    Resize PIL image so width/height are divisible by multiple.
    """
    multiple = max(1, multiple)

    width, height = image.size

    new_width = width - (width % multiple)
    new_height = height - (height % multiple)

    new_width = max(multiple, new_width)
    new_height = max(multiple, new_height)

    if (new_width, new_height) == (width, height):
        return image

    return image.resize((new_width, new_height))


def resize_pil_to_size(
    image: Any,
    size: tuple[int, int],
) -> Any:
    return image.resize(size)


def resize_numpy_to_shape(
    image: np.ndarray,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """
    Resize NumPy image to HxW shape.
    """
    cv2 = lazy_import("cv2")

    height, width = image_shape

    return cv2.resize(
        image,
        dsize=(width, height),
        interpolation=cv2.INTER_LINEAR,
    )


def resolve_device(device: str = "auto") -> str:
    """
    Resolve torch device.
    """
    if device != "auto":
        return device

    try:
        torch = lazy_import("torch")
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def resolve_torch_dtype(torch: Any, dtype_name: str) -> Any:
    if dtype_name == "float16":
        return torch.float16

    if dtype_name == "float32":
        return torch.float32

    if dtype_name == "bfloat16":
        return torch.bfloat16

    raise ValueError(f"Unsupported torch_dtype: {dtype_name}")


def make_generator(
    seed: int | None,
    device: str = "auto",
) -> Any | None:
    """
    Create deterministic torch generator if seed is provided.
    """
    if seed is None:
        return None

    torch = lazy_import("torch")
    resolved_device = resolve_device(device)

    try:
        return torch.Generator(device=resolved_device).manual_seed(seed)
    except Exception:
        return torch.Generator().manual_seed(seed)