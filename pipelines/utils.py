from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


def normalize_mask(mask: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    if mask.ndim == 3:
        mask = mask[..., 0]

    if mask.shape[:2] != image_shape:
        raise ValueError(f"Mask shape {mask.shape[:2]} does not match image shape {image_shape}")

    if mask.dtype != np.uint8:
        return (mask > 0).astype(np.uint8) * 255

    if mask.size > 0 and mask.max() <= 1:
        return mask * 255

    return np.where(mask > 0, 255, 0).astype(np.uint8)


def union_masks(masks: list[np.ndarray], image_shape: tuple[int, int]) -> np.ndarray:
    full_mask = np.zeros(image_shape, dtype=np.uint8)

    for mask in masks:
        full_mask = np.maximum(full_mask, normalize_mask(mask, image_shape))

    return full_mask


def has_non_empty_mask(result: Any) -> bool:
    mask = getattr(result, "mask", None)

    if mask is None:
        return False

    try:
        return int(np.count_nonzero(mask)) > 0
    except Exception:
        return False


def non_empty_results(detector_results: dict[str, Any]) -> dict[str, Any]:
    return {name: result for name, result in detector_results.items() if has_non_empty_mask(result)}


def empty_result_names(detector_results: dict[str, Any]) -> list[str]:
    return [name for name, result in detector_results.items() if not has_non_empty_mask(result)]


def summarize_prompts(prompts: dict[str, Any]) -> dict[str, Any]:
    return {
        key: {
            "type": type(value).__name__,
            "shape": getattr(value, "shape", None),
            "length": len(value) if value is not None and hasattr(value, "__len__") else None,
        }
        for key, value in prompts.items()
    }


def mask_to_prompt_points(mask: np.ndarray, max_points: int = 20) -> list[list[int]]:
    if mask.ndim == 3:
        mask = mask[..., 0]

    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        return []

    if len(xs) <= max_points:
        indices = range(len(xs))
    else:
        indices = np.linspace(0, len(xs) - 1, max_points, dtype=int)

    points: list[list[int]] = []

    for index in indices:
        points.append([int(ys[index]), int(xs[index])])

    return points


def extract_boxes(result: Any) -> list[Any]:
    boxes: list[Any] = []

    result_boxes = getattr(result, "boxes", None)

    if result_boxes:
        boxes.extend(result_boxes)

    detections = getattr(result, "detections", [])

    for detection in detections:
        bbox = getattr(detection, "bbox", None)

        if bbox is not None:
            boxes.append(bbox)

    return boxes


def resolve_detector_mode(proposal_detectors: list[str], refiner_detectors: list[str]) -> str:
    proposal_count = len(proposal_detectors)
    refiner_count = len(refiner_detectors)
    total = proposal_count + refiner_count

    if total == 0:
        return "disabled"

    if proposal_count > 1 and refiner_count > 1:
        return "proposal_fused_refiner_fused"

    if proposal_count > 1 and refiner_count == 1:
        return "proposal_fused_refiner_single"

    if proposal_count == 1 and refiner_count > 1:
        return "proposal_single_refiner_fused"

    if proposal_count > 1:
        return "proposal_fused"

    if refiner_count > 1:
        return "refiner_fused"

    if proposal_count == 1 and refiner_count == 1:
        return "proposal_refiner"

    return "single"


def collect_detections(detector_results: dict[str, Any]) -> list[Any]:
    detections: list[Any] = []

    for name, result in detector_results.items():
        if name in {"proposal_fusion", "refiner_fusion", "fusion"}:
            continue

        result_detections = getattr(result, "detections", [])
        detections.extend(result_detections)

    return detections


def is_cancelled(context: dict[str, Any] | None) -> bool:
    if not context:
        return False

    cancel_token = context.get("cancel_token")

    if not isinstance(cancel_token, dict):
        return False

    return bool(cancel_token.get("stop", False))


def get_optional_item(items: list[Any] | None, index: int) -> Any | None:
    if items is None:
        return None

    if index >= len(items):
        return None

    return items[index]
