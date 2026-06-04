# detectors/utils.py

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
from pathlib import Path
from collections.abc import Mapping, Sequence

from core.logger import log_detector_event

BBox = tuple[int, int, int, int]

_MODULE_CACHE: dict[str, Any] = {}


def lazy_import(module_name: str) -> Any:
    """
    Lazy import optional detector dependencies.

    Examples:
        cv2 = lazy_import("cv2")
        torch = lazy_import("torch")
        ultralytics = lazy_import("ultralytics")
    """
    if module_name not in _MODULE_CACHE:
        try:
            _MODULE_CACHE[module_name] = importlib.import_module(module_name)
        except ImportError as exc:
            raise ImportError(
                f"Missing optional dependency '{module_name}'."
            ) from exc

    return _MODULE_CACHE[module_name]


def validate_image(image: np.ndarray, detector_name: str = "detector") -> None:
    """
    Validate image input for detectors.

    Supported:
        H x W
        H x W x 1
        H x W x 3
        H x W x 4
    """
    if image is None:
        raise ValueError(f"{detector_name} received image=None")

    if not isinstance(image, np.ndarray):
        raise TypeError(
            f"{detector_name} expected np.ndarray, got {type(image)}"
        )

    if image.ndim not in (2, 3):
        raise ValueError(
            f"{detector_name} expected HxW or HxWxC image, "
            f"got shape={image.shape}"
        )

    if image.ndim == 3 and image.shape[-1] not in (1, 3, 4):
        raise ValueError(
            f"{detector_name} expected 1, 3, or 4 channels, "
            f"got {image.shape[-1]}"
        )


def empty_mask(image_or_shape: np.ndarray | tuple[int, int]) -> np.ndarray:
    """
    Create an empty binary mask.
    """
    if isinstance(image_or_shape, tuple):
        height, width = image_or_shape
    else:
        height, width = image_or_shape.shape[:2]

    return np.zeros((height, width), dtype=np.uint8)


def to_numpy(value: Any) -> np.ndarray:
    """
    Convert Torch tensors or other array-like outputs to NumPy.
    """
    try:
        torch = lazy_import("torch")
    except ImportError:
        torch = None

    if torch is not None and hasattr(torch, "Tensor") and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()

    return np.asarray(value)


def to_mask_uint8(mask: Any, threshold: float = 0.5) -> np.ndarray:
    """
    Convert bool/probability/logit-like masks to binary uint8 mask.

    Output:
        uint8 mask with values 0 or 255.
    """
    mask_np = to_numpy(mask)

    while mask_np.ndim > 2:
        mask_np = mask_np[0]

    if mask_np.dtype == np.bool_:
        return mask_np.astype(np.uint8) * 255

    if mask_np.dtype == np.uint8:
        if mask_np.max() <= 1:
            return mask_np * 255
        return mask_np

    if mask_np.max() <= 1.0:
        return (mask_np > threshold).astype(np.uint8) * 255

    return (mask_np > 127).astype(np.uint8) * 255


def normalize01(image: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Normalize image to [0, 1].
    """
    image = image.astype(np.float32)

    min_value = float(np.min(image))
    max_value = float(np.max(image))

    if max_value - min_value < eps:
        return np.zeros_like(image, dtype=np.float32)

    return ((image - min_value) / (max_value - min_value)).astype(np.float32)


def to_gray_float(
    image: np.ndarray,
    input_color_order: str = "bgr",
) -> np.ndarray:
    """
    Convert image to grayscale float32 in [0, 1].
    """
    validate_image(image, "to_gray_float")

    if image.ndim == 2:
        gray = image

    elif image.ndim == 3 and image.shape[-1] == 1:
        gray = image[..., 0]

    elif image.ndim == 3 and image.shape[-1] in (3, 4):
        color = image[..., :3].astype(np.float32)

        if input_color_order == "rgb":
            r = color[..., 0]
            g = color[..., 1]
            b = color[..., 2]
        else:
            b = color[..., 0]
            g = color[..., 1]
            r = color[..., 2]

        gray = 0.299 * r + 0.587 * g + 0.114 * b

    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    gray = gray.astype(np.float32)

    if gray.max() > 1.0:
        gray = gray / 255.0

    return np.clip(gray, 0.0, 1.0)


def to_rgb(
    image: np.ndarray,
    input_color_order: str = "bgr",
) -> np.ndarray:
    """
    Convert image to RGB for OCR/SAM-like models.
    """
    validate_image(image, "to_rgb")

    cv2 = lazy_import("cv2")

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    if image.ndim == 3 and image.shape[-1] == 1:
        return cv2.cvtColor(image[..., 0], cv2.COLOR_GRAY2RGB)

    if image.ndim == 3 and image.shape[-1] == 3:
        if input_color_order == "rgb":
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if image.ndim == 3 and image.shape[-1] == 4:
        color = image[..., :3]
        if input_color_order == "rgb":
            return color
        return cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

    raise ValueError(f"Unsupported image shape: {image.shape}")


def mask_to_bbox(mask: np.ndarray) -> BBox | None:
    """
    Convert binary mask to bbox.

    Returns:
        x1, y1, x2, y2

    x2/y2 are exclusive.
    """
    mask_uint8 = to_mask_uint8(mask)

    ys, xs = np.where(mask_uint8 > 0)

    if len(xs) == 0 or len(ys) == 0:
        return None

    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1

    return x1, y1, x2, y2


def bbox_area(bbox: BBox) -> int:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def bbox_to_mask(
    bbox: BBox,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """
    Convert bbox to binary mask.
    """
    height, width = image_shape

    x1, y1, x2, y2 = bbox

    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))

    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255

    return mask


def polygon_to_bbox(points: Any) -> BBox:
    """
    Convert polygon points to bbox.

    Used by EasyOCR and segmentation outputs.
    """
    points_np = np.asarray(points, dtype=np.float32)

    xs = points_np[:, 0]
    ys = points_np[:, 1]

    x1 = int(np.floor(xs.min()))
    y1 = int(np.floor(ys.min()))
    x2 = int(np.ceil(xs.max()))
    y2 = int(np.ceil(ys.max()))

    return x1, y1, x2, y2


def pad_bbox(
    bbox: BBox,
    image_shape: tuple[int, int],
    padding: int,
) -> BBox:
    """
    Pad bbox while keeping it inside image.
    """
    height, width = image_shape

    x1, y1, x2, y2 = bbox

    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(width, x2 + padding)
    y2 = min(height, y2 + padding)

    return int(x1), int(y1), int(x2), int(y2)


def merge_masks(
    masks: list[np.ndarray],
    image_shape: tuple[int, int],
) -> np.ndarray:
    """
    Merge multiple binary masks into one mask.
    """
    full_mask = np.zeros(image_shape, dtype=np.uint8)

    for mask in masks:
        mask_uint8 = to_mask_uint8(mask)

        if mask_uint8.shape[:2] != image_shape:
            mask_uint8 = resize_mask_nearest(mask_uint8, image_shape)

        full_mask = np.maximum(full_mask, mask_uint8)

    return full_mask


def resize_mask_nearest(
    mask: np.ndarray,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """
    Resize mask using nearest-neighbor interpolation.
    """
    cv2 = lazy_import("cv2")

    height, width = image_shape

    return cv2.resize(
        to_mask_uint8(mask),
        dsize=(width, height),
        interpolation=cv2.INTER_NEAREST,
    )


def dilate_mask(
    mask: np.ndarray,
    iterations: int = 1,
    kernel_size: int = 5,
) -> np.ndarray:
    """
    Dilate binary mask.
    """
    if iterations <= 0:
        return to_mask_uint8(mask)

    cv2 = lazy_import("cv2")

    kernel_size = max(1, kernel_size)

    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    return cv2.dilate(
        to_mask_uint8(mask),
        kernel,
        iterations=iterations,
    )


def clean_binary_mask(
    mask: np.ndarray,
    kernel_size: int = 3,
    close: bool = True,
    open_: bool = False,
    dilate_iterations: int = 0,
) -> np.ndarray:
    """
    Common binary-mask cleanup for detectors.
    """
    cv2 = lazy_import("cv2")

    kernel_size = max(1, kernel_size)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    cleaned = to_mask_uint8(mask)

    if close:
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    if open_:
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    if dilate_iterations > 0:
        cleaned = cv2.dilate(
            cleaned,
            kernel,
            iterations=dilate_iterations,
        )

    return cleaned


def crop_mask_to_bbox(
    mask: np.ndarray,
    bbox: BBox,
) -> np.ndarray:
    """
    Keep only mask area inside bbox.
    """
    mask_uint8 = to_mask_uint8(mask)

    output = np.zeros_like(mask_uint8, dtype=np.uint8)

    x1, y1, x2, y2 = bbox

    height, width = mask_uint8.shape[:2]

    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))

    output[y1:y2, x1:x2] = mask_uint8[y1:y2, x1:x2]

    return output


def sam_bbox_to_xyxy(bbox: Any) -> BBox:
    """
    Convert SAM/SAM2 automatic mask bbox from:
        x, y, width, height

    to:
        x1, y1, x2, y2
    """
    values = list(map(float, bbox))

    if len(values) != 4:
        raise ValueError(f"Invalid bbox: {bbox}")

    x, y, w, h = values

    return int(x), int(y), int(x + w), int(y + h)
def prompt_value(
    prompts: dict[str, Any],
    primary_key: str,
    fallback_key: str | None = None,
) -> Any:
    """
    Safe replacement for:
        prompts.get("a") or prompts.get("b")

    Important:
        NumPy arrays cannot be used with Python `or`.
    """
    value = prompts.get(primary_key)

    if value is not None:
        return value

    if fallback_key is None:
        return None

    return prompts.get(fallback_key)


def has_prompt_value(value: Any) -> bool:
    if value is None:
        return False

    if isinstance(value, np.ndarray):
        return value.size > 0

    try:
        return len(value) > 0
    except TypeError:
        return True


def has_prompts(prompts: dict[str, Any]) -> bool:
    for key in ["boxes", "points", "point_labels", "labels", "mask_input"]:
        if has_prompt_value(prompts.get(key)):
            return True

    return False


def summarize_prompts(prompts: dict[str, Any]) -> dict[str, Any]:
    return {
        key: {
            "type": type(value).__name__,
            "shape": getattr(value, "shape", None),
            "length": len(value) if value is not None and hasattr(value, "__len__") else None,
        }
        for key, value in prompts.items()
    }


def normalize_boxes(boxes: Any) -> np.ndarray | None:
    """
    Normalize boxes to shape:
        N x 4

    Box format:
        x1, y1, x2, y2
    """
    if boxes is None:
        return None

    boxes_array = np.asarray(boxes, dtype=np.int32)

    if boxes_array.size == 0:
        return None

    if boxes_array.size % 4 != 0:
        raise ValueError(
            f"boxes must contain groups of 4 values. "
            f"Got shape={boxes_array.shape}, size={boxes_array.size}"
        )

    return boxes_array.reshape(-1, 4)


def normalize_points(points: Any) -> np.ndarray | None:
    """
    Normalize points to shape:
        N x 2

    Point format:
        x, y
    """
    if points is None:
        return None

    points_array = np.asarray(points, dtype=np.float32)

    if points_array.size == 0:
        return None

    if points_array.size % 2 != 0:
        raise ValueError(
            f"points must contain groups of 2 values. "
            f"Got shape={points_array.shape}, size={points_array.size}"
        )

    return points_array.reshape(-1, 2)


def normalize_point_labels(
    point_labels: Any,
    point_count: int,
) -> np.ndarray:
    """
    Normalize point labels to shape:
        N

    If labels are missing, default all points to foreground.
    """
    if point_labels is None:
        return np.ones((point_count,), dtype=np.int32)

    labels_array = np.asarray(point_labels, dtype=np.int32).reshape(-1)

    if len(labels_array) != point_count:
        raise ValueError(
            f"point_labels length must match points length. "
            f"Got {len(labels_array)} labels and {point_count} points."
        )

    return labels_array


def points_inside_box(
    points: np.ndarray,
    box: np.ndarray,
) -> np.ndarray:
    """
    Return a boolean mask for points inside one XYXY box.
    """
    x1, y1, x2, y2 = box.tolist()

    return (
        (points[:, 0] >= x1)
        & (points[:, 0] <= x2)
        & (points[:, 1] >= y1)
        & (points[:, 1] <= y2)
    )


def best_mask_index(scores: np.ndarray | None) -> int:
    if scores is None:
        return 0

    return int(np.argmax(scores))


def extract_auto_mask(item: Any) -> np.ndarray | None:
    """
    Safe replacement for:
        item.get("segmentation") or item.get("mask")
    """
    if isinstance(item, dict):
        raw_mask = item.get("segmentation")

        if raw_mask is None:
            raw_mask = item.get("mask")

        if raw_mask is None:
            return None

        return np.asarray(raw_mask)

    if hasattr(item, "shape"):
        return np.asarray(item)

    return None


def check_packages(
    packages: Mapping[str, str] | Sequence[str],
    component_name: str = "component",
    strict: bool = True,
    log: bool = True,
) -> dict[str, dict[str, Any]]:
    """
    Check optional/runtime Python packages.

    Args:
        packages:
            Either:
                {"pip_name": "import_name"}
            or:
                ["torch", "transformers"]

        component_name:
            Used in logs/errors.

        strict:
            If True, raise ImportError when a package is missing.
            If False, return status only.

    Returns:
        Package status dictionary.
    """
    if isinstance(packages, Mapping):
        package_items = list(packages.items())
    else:
        package_items = [(name, name) for name in packages]

    status: dict[str, dict[str, Any]] = {}

    for package_name, import_name in package_items:
        try:
            module = lazy_import(import_name)

            status[package_name] = {
                "ok": True,
                "package": package_name,
                "import_name": import_name,
                "version": getattr(module, "__version__", None),
            }

        except ImportError as exc:
            status[package_name] = {
                "ok": False,
                "package": package_name,
                "import_name": import_name,
                "error": str(exc),
            }

            if strict:
                if log:
                    log_detector_event(
                        component_name,
                        "package_check_failed",
                        status[package_name],
                    )

                raise ImportError(
                    f"{component_name} requires package '{package_name}' "
                    f"imported as '{import_name}'. Install it inside your project venv."
                ) from exc

    if log:
        log_detector_event(
            component_name,
            "package_check",
            status,
        )

    return status


def check_file_exists(
    path: str | Path | None,
    label: str = "file",
    component_name: str = "component",
    required: bool = True,
    allowed_suffixes: Sequence[str] | None = None,
    log: bool = True,
) -> dict[str, Any]:
    """
    Check one local file path.

    Use this for local checkpoints like:
        models/mobile_sam/mobile_sam.pt

    Do not use this for Hugging Face model IDs like:
        IDEA-Research/grounding-dino-tiny
    """
    status: dict[str, Any] = {
        "label": label,
        "path": None if path is None else str(path),
        "required": required,
        "ok": False,
        "exists": False,
        "is_file": False,
        "suffix": None,
    }

    if path is None:
        status["error"] = "path_is_none"

        if required:
            if log:
                log_detector_event(component_name, "file_check_failed", status)

            raise FileNotFoundError(f"{component_name} required {label}, got None")

        if log:
            log_detector_event(component_name, "file_check", status)

        return status

    file_path = Path(path).expanduser()

    status.update(
        {
            "path": str(file_path),
            "exists": file_path.exists(),
            "is_file": file_path.is_file(),
            "suffix": file_path.suffix,
        }
    )

    if allowed_suffixes is not None:
        allowed = set(allowed_suffixes)
        status["allowed_suffixes"] = sorted(allowed)

        if file_path.suffix not in allowed:
            status["error"] = f"invalid_suffix:{file_path.suffix}"

            if required:
                if log:
                    log_detector_event(component_name, "file_check_failed", status)

                raise ValueError(
                    f"{component_name} {label} must have suffix in "
                    f"{sorted(allowed)}, got '{file_path.suffix}'"
                )

    if not file_path.exists() or not file_path.is_file():
        status["error"] = "missing_or_not_file"

        if required:
            if log:
                log_detector_event(component_name, "file_check_failed", status)

            raise FileNotFoundError(
                f"{component_name} required {label} not found: {file_path}"
            )

    status["ok"] = True

    if log:
        log_detector_event(component_name, "file_check", status)

    return status


def check_directory_exists(
    path: str | Path | None,
    label: str = "directory",
    component_name: str = "component",
    required: bool = True,
    log: bool = True,
) -> dict[str, Any]:
    """
    Check one local directory path.
    """
    status: dict[str, Any] = {
        "label": label,
        "path": None if path is None else str(path),
        "required": required,
        "ok": False,
        "exists": False,
        "is_dir": False,
    }

    if path is None:
        status["error"] = "path_is_none"

        if required:
            if log:
                log_detector_event(component_name, "directory_check_failed", status)

            raise FileNotFoundError(f"{component_name} required {label}, got None")

        if log:
            log_detector_event(component_name, "directory_check", status)

        return status

    directory = Path(path).expanduser()

    status.update(
        {
            "path": str(directory),
            "exists": directory.exists(),
            "is_dir": directory.is_dir(),
        }
    )

    if not directory.exists() or not directory.is_dir():
        status["error"] = "missing_or_not_directory"

        if required:
            if log:
                log_detector_event(component_name, "directory_check_failed", status)

            raise FileNotFoundError(
                f"{component_name} required {label} not found: {directory}"
            )

    status["ok"] = True

    if log:
        log_detector_event(component_name, "directory_check", status)

    return status


def is_probably_local_path(value: str | Path | None) -> bool:
    """
    Distinguish local paths from Hugging Face IDs.

    Examples:
        models/foo             -> local
        ./models/foo           -> local
        /abs/path/foo          -> local
        IDEA-Research/model    -> remote/HF
    """
    if value is None:
        return False

    text = str(value).strip()

    if not text:
        return False

    if text.startswith(("./", "../", "/", "~")):
        return True

    path = Path(text).expanduser()

    if path.exists():
        return True

    if path.suffix:
        return True

    return False


def check_model_source(
    model_id_or_path: str,
    component_name: str = "component",
    local_required_files: Sequence[str] | None = None,
    strict: bool = True,
    log: bool = True,
) -> dict[str, Any]:
    """
    Check whether model source is:
        - local path
        - remote Hugging Face model ID

    For GroundingDINO via Transformers, this usually returns remote_hf.
    For local checkpoints/folders, it validates the path.
    """
    if is_probably_local_path(model_id_or_path):
        model_path = Path(model_id_or_path).expanduser()

        if model_path.is_file():
            status = check_file_exists(
                model_path,
                label="model_file",
                component_name=component_name,
                required=strict,
                log=log,
            )
            status["source_type"] = "local_file"
            return status

        if model_path.is_dir():
            status = check_directory_exists(
                model_path,
                label="model_directory",
                component_name=component_name,
                required=strict,
                log=log,
            )
            status["source_type"] = "local_directory"

            missing_files: list[str] = []

            for filename in local_required_files or ():
                required_path = model_path / filename

                if not required_path.exists():
                    missing_files.append(filename)

            status["missing_required_files"] = missing_files

            if missing_files and strict:
                if log:
                    log_detector_event(
                        component_name,
                        "model_source_check_failed",
                        status,
                    )

                raise FileNotFoundError(
                    f"{component_name} local model directory is missing files: "
                    f"{missing_files}"
                )

            return status

        status = {
            "source_type": "local_missing",
            "model_id_or_path": model_id_or_path,
            "ok": False,
            "error": "local_path_missing",
        }

        if strict:
            if log:
                log_detector_event(component_name, "model_source_check_failed", status)

            raise FileNotFoundError(
                f"{component_name} local model path not found: {model_id_or_path}"
            )

        if log:
            log_detector_event(component_name, "model_source_check", status)

        return status

    status = {
        "source_type": "remote_hf",
        "model_id": model_id_or_path,
        "ok": True,
        "note": "will be resolved by from_pretrained/cache",
    }

    if log:
        log_detector_event(component_name, "model_source_check", status)

    return status
