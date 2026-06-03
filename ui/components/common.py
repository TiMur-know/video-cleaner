# ui/components/common.py

from __future__ import annotations

from typing import Any

import numpy as np


def get_file_path(file_value: Any) -> str:
    """
    Gradio can return different object types depending on version.

    Supported:
        str
        object with .name
        object with .path
        dict with "name" or "path"
    """

    if isinstance(file_value, str):
        return file_value

    if hasattr(file_value, "name"):
        return str(file_value.name)

    if hasattr(file_value, "path"):
        return str(file_value.path)

    if isinstance(file_value, dict):
        if "name" in file_value:
            return str(file_value["name"])

        if "path" in file_value:
            return str(file_value["path"])

    raise ValueError(f"Unsupported file input value: {type(file_value)}")


def to_preview_rgb(image: np.ndarray | None) -> np.ndarray | None:
    """
    Convert OpenCV-style BGR image to RGB for Gradio preview.
    """

    if image is None:
        return None

    if image.ndim == 2:
        return image

    if image.ndim == 3 and image.shape[-1] == 3:
        return image[..., ::-1]

    if image.ndim == 3 and image.shape[-1] == 4:
        return image[..., [2, 1, 0, 3]]

    return image


def to_preview_mask(mask: np.ndarray | None) -> np.ndarray | None:
    """
    Convert mask to uint8 preview.

    Supports:
        bool masks
        0/1 masks
        0..255 masks
        float masks
        3-channel masks
    """

    if mask is None:
        return None

    if mask.ndim == 3:
        mask = mask[..., 0]

    if mask.dtype == np.bool_:
        return mask.astype(np.uint8) * 255

    if mask.dtype == np.uint8:
        if mask.size > 0 and mask.max() <= 1:
            return mask * 255

        return mask

    if mask.size > 0 and mask.max() <= 1.0:
        return (mask * 255.0).clip(0, 255).astype(np.uint8)

    return mask.clip(0, 255).astype(np.uint8)