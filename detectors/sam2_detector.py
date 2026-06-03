# detectors/sam2_detector.py

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from detectors.utils import (
    BBox,
    dilate_mask,
    empty_mask,
    mask_to_bbox,
    merge_masks,
    resize_mask_nearest,
    sam_bbox_to_xyxy,
    to_mask_uint8,
    to_numpy,
    validate_image,
)


@dataclass(slots=True)
class SAM2Detection:
    bbox: BBox
    confidence: float
    mask: np.ndarray
    label: str = "sam2_watermark"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SAM2DetectorResult:
    mask: np.ndarray
    detections: list[SAM2Detection] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SAM2DetectorConfig:
    enabled: bool = True

    # Local checkpoint path.
    checkpoint_path: str | None = "models/sam2/sam2_b.pt"

    # Important:
    # Official SAM2 usually wants a config name from the installed sam2 package,
    # not always a local yaml path.
    #
    # Common SAM2:
    #   configs/sam2/sam2_hiera_b+.yaml
    #
    # Common SAM2.1:
    #   configs/sam2.1/sam2.1_hiera_b+.yaml
    model_config_path: str | None = "configs/sam2/sam2_hiera_b+.yaml"

    device: str = "auto"

    # SAM2 is usually a refinement model.
    # It works best with boxes/points from other detectors or auto prompts.
    require_prompts: bool = True

    # If model cannot load, return empty mask instead of crashing.
    fallback_to_empty: bool = True

    # If True, raise the real loading/prediction error.
    strict_loading: bool = False

    min_area: int = 50
    mask_threshold: float = 0.5

    dilate_iterations: int = 1
    morph_kernel_size: int = 5

    # Optional manual injection if you already created a SAM2 predictor/model.
    model_factory: Callable[["SAM2DetectorConfig"], Any] | None = None

    return_debug: bool = False

    # Auto watermark prompts.
    # SAM2 does not understand text like "find dark watermark".
    # These options create visual prompts: boxes, points, point_labels.
    use_auto_watermark_prompts: bool = True

    prompt_dark_watermark: bool = True
    prompt_light_watermark: bool = True
    prompt_low_opacity_watermark: bool = True
    prompt_text_watermark: bool = True

    prompt_sensitivity: float = 50.0

    prompt_min_area: int = 25
    prompt_max_area_ratio: float = 0.35
    prompt_max_boxes: int = 20

    dark_percentile: float = 8.0
    light_percentile: float = 92.0

    low_opacity_edge_low: int = 30
    low_opacity_edge_high: int = 100


class SAM2Detector:
    """
    SAM2 detector/refiner.

    Recommended detector combinations:
        opencv + sam2
        yolo + sam2
        opencv + fft + anomaly + sam2

    SAM2 itself is not a text/watermark classifier. It segments regions from:
        - boxes
        - points
        - mask prompts
        - auto-generated visual watermark prompts

    Model files:
        checkpoint:
            models/sam2/sam2_b.pt

        config:
            usually package config name:
                configs/sam2/sam2_hiera_b+.yaml

            or SAM2.1:
                configs/sam2.1/sam2.1_hiera_b+.yaml
    """

    name = "sam2"

    def __init__(
        self,
        config: SAM2DetectorConfig | None = None,
        model: Any | None = None,
    ) -> None:
        self.config = config or SAM2DetectorConfig()
        self.model = model

    def detect(
        self,
        image: np.ndarray,
        prompts: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> SAM2DetectorResult:
        context = context or {}

        if not self.config.enabled:
            result = SAM2DetectorResult(
                mask=empty_mask(image),
                metadata={
                    "detector": self.name,
                    "enabled": False,
                },
            )
            self._log_detect_output(result)
            return result

        validate_image(image, self.name)

        prompts = prompts or {}
        self._log_detect_input(image=image, prompts=prompts)

        if not self._has_prompts(prompts) and self.config.use_auto_watermark_prompts:
            prompts = self._build_auto_watermark_prompts(image)

        if self.config.require_prompts and not self._has_prompts(prompts):
            result = SAM2DetectorResult(
                mask=empty_mask(image),
                metadata={
                    "detector": self.name,
                    "skipped": True,
                    "reason": "prompts required but not provided",
                    "auto_watermark_prompts": self.config.use_auto_watermark_prompts,
                    "prompt_keys": list(prompts.keys()),
                },
            )
            self._log_detect_output(result)
            return result

        if self.model is None:
            try:
                self.model = self._build_model()
            except Exception as exc:
                if self.config.strict_loading or not self.config.fallback_to_empty:
                    raise

                result = SAM2DetectorResult(
                    mask=empty_mask(image),
                    metadata={
                        "detector": self.name,
                        "skipped": True,
                        "reason": f"model load failed: {type(exc).__name__}: {exc}",
                        "checkpoint_path": self.config.checkpoint_path,
                        "model_config_path": self.config.model_config_path,
                    },
                )
                self._log_detect_output(result)
                return result

        try:
            raw_output = self._run_model(image, prompts)
        except Exception as exc:
            if self.config.strict_loading or not self.config.fallback_to_empty:
                raise

            result = SAM2DetectorResult(
                mask=empty_mask(image),
                metadata={
                    "detector": self.name,
                    "skipped": True,
                    "reason": f"prediction failed: {type(exc).__name__}: {exc}",
                    "used_prompts": bool(prompts),
                    "prompt_keys": list(prompts.keys()),
                },
            )
            self._log_detect_output(result)
            return result

        detections = self._parse_output(
            output=raw_output,
            image_shape=image.shape[:2],
        )

        full_mask = merge_masks(
            [detection.mask for detection in detections],
            image_shape=image.shape[:2],
        )

        if self.config.dilate_iterations > 0:
            full_mask = dilate_mask(
                full_mask,
                iterations=self.config.dilate_iterations,
                kernel_size=self.config.morph_kernel_size,
            )

        metadata: dict[str, Any] = {
            "detector": self.name,
            "enabled": True,
            "num_detections": len(detections),
            "mask_area": int(np.count_nonzero(full_mask)),
            "used_prompts": bool(prompts),
            "prompt_keys": list(prompts.keys()),
            "auto_watermark_prompts": self.config.use_auto_watermark_prompts,
            "checkpoint_path": self.config.checkpoint_path,
            "model_config_path": self.config.model_config_path,
            "device": self._resolve_device(),
        }

        if self.config.return_debug:
            metadata["raw_output"] = raw_output
            metadata["prompts"] = self._debug_safe_prompts(prompts)

        result = SAM2DetectorResult(
            mask=full_mask,
            detections=detections,
            metadata=metadata,
        )
        self._log_detect_output(result)
        return result

    def _summarize_prompts(self, prompts: dict[str, Any]) -> dict[str, Any]:
        return {
            key: {
                "type": type(value).__name__,
                "shape": getattr(value, "shape", None),
                "length": len(value) if value is not None and hasattr(value, "__len__") else None,
            }
            for key, value in prompts.items()
        }

    def _log_detect_input(self, image: np.ndarray, prompts: dict[str, Any]) -> None:
        print(
            "SAM2Detector input:",
            {
                "image_shape": image.shape if image is not None else None,
                "prompt_keys": list(prompts.keys()),
                "prompts": self._summarize_prompts(prompts),
            },
        )

    def _log_detect_output(self, result: SAM2DetectorResult) -> None:
        print(
            "SAM2Detector output:",
            {
                "enabled": result.metadata.get("enabled"),
                "skipped": result.metadata.get("skipped"),
                "num_detections": result.metadata.get("num_detections"),
                "mask_area": result.metadata.get("mask_area"),
                "reason": result.metadata.get("reason"),
                "prompt_keys": result.metadata.get("prompt_keys"),
            },
        )

    def _build_model(self) -> Any:
        if self.config.model_factory is not None:
            return self.config.model_factory(self.config)

        checkpoint_path = self._resolve_checkpoint_path()
        model_config_path = self._resolve_model_config_path()
        device = self._resolve_device()

        try:
            build_module = importlib.import_module("sam2.build_sam")
            predictor_module = importlib.import_module("sam2.sam2_image_predictor")
        except ImportError as exc:
            raise ImportError(
                "SAM2 package is not installed. Install/configure SAM2 first, "
                "or pass a ready predictor to SAM2Detector(model=...)."
            ) from exc

        if not hasattr(build_module, "build_sam2"):
            raise ImportError("sam2.build_sam does not expose build_sam2().")

        if not hasattr(predictor_module, "SAM2ImagePredictor"):
            raise ImportError(
                "sam2.sam2_image_predictor does not expose SAM2ImagePredictor."
            )

        build_sam2 = getattr(build_module, "build_sam2")
        predictor_cls = getattr(predictor_module, "SAM2ImagePredictor")

        model = build_sam2(
            model_config_path,
            checkpoint_path,
            device=device,
        )

        return predictor_cls(model)

    def _resolve_checkpoint_path(self) -> str:
        if not self.config.checkpoint_path:
            raise ValueError("SAM2 checkpoint_path is not set.")

        path = Path(self.config.checkpoint_path)

        if not path.exists():
            raise FileNotFoundError(
                f"SAM2 checkpoint not found: {path}. "
                "Put your checkpoint in models/sam2/ or update checkpoint_path."
            )

        return str(path)

    def _resolve_model_config_path(self) -> str:
        if not self.config.model_config_path:
            raise ValueError(
                "SAM2 model_config_path is not set. "
                "Example: configs/sam2/sam2_hiera_b+.yaml"
            )

        # Official SAM2 build_sam2 usually accepts package config names like:
        # configs/sam2/sam2_hiera_b+.yaml
        #
        # Some custom loaders may accept local YAML paths too.
        # So if a local path exists, we allow it. Otherwise we pass the string.
        path = Path(self.config.model_config_path)

        if path.exists():
            return str(path)

        return self.config.model_config_path

    def _resolve_device(self) -> str:
        if self.config.device != "auto":
            return self.config.device

        try:
            torch = importlib.import_module("torch")

            if torch.cuda.is_available():
                return "cuda"

            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"

        except Exception:
            pass

        return "cpu"

    def _has_prompts(
        self,
        prompts: dict[str, Any],
    ) -> bool:
        for key in ["boxes", "points", "labels", "point_labels", "mask_input"]:
            value = prompts.get(key)

            if value is None:
                continue

            try:
                if len(value) > 0:
                    return True
            except TypeError:
                return True

        return False

    def _run_model(
        self,
        image: np.ndarray,
        prompts: dict[str, Any],
    ) -> Any:
        """
        Supported prompt keys:
            boxes
            points
            labels
            point_labels
            mask_input
        """

        image_rgb = self._to_rgb_image(image)

        if hasattr(self.model, "set_image") and hasattr(self.model, "predict"):
            self.model.set_image(image_rgb)

            boxes = prompts.get("boxes")
            points = prompts.get("points")
            point_labels = prompts.get("point_labels", prompts.get("labels"))
            mask_input = prompts.get("mask_input")

            if boxes is not None:
                boxes_array = np.asarray(boxes)

                if boxes_array.ndim == 1:
                    boxes_array = boxes_array[None, :]

                outputs = []

                for box in boxes_array:
                    outputs.append(
                        self.model.predict(
                            box=np.asarray(box, dtype=np.float32),
                            point_coords=None,
                            point_labels=None,
                            mask_input=mask_input,
                            multimask_output=True,
                        )
                    )

                return outputs

            return self.model.predict(
                box=None,
                point_coords=points,
                point_labels=point_labels,
                mask_input=mask_input,
                multimask_output=True,
            )

        if hasattr(self.model, "generate"):
            return self.model.generate(image_rgb)

        if callable(self.model):
            try:
                return self.model(image_rgb, prompts=prompts)
            except TypeError:
                return self.model(image_rgb)

        raise TypeError(
            "Unsupported SAM2 model interface. Expected one of: "
            "set_image()+predict(), generate(), or callable model."
        )

    def _to_rgb_image(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Your project uses OpenCV BGR images.
        SAM2 predictors usually expect RGB.
        """

        if image.ndim == 2:
            return image

        if image.ndim == 3 and image.shape[-1] == 3:
            return image[..., ::-1].copy()

        if image.ndim == 3 and image.shape[-1] == 4:
            return image[..., [2, 1, 0, 3]].copy()

        return image

    def _parse_output(
        self,
        output: Any,
        image_shape: tuple[int, int],
    ) -> list[SAM2Detection]:
        if isinstance(output, list):
            detections: list[SAM2Detection] = []

            for item in output:
                detections.extend(
                    self._parse_output(
                        output=item,
                        image_shape=image_shape,
                    )
                )

            return detections

        if isinstance(output, tuple):
            masks = output[0]
            scores = output[1] if len(output) > 1 else None

            return self._parse_masks_and_scores(
                masks=masks,
                scores=scores,
                image_shape=image_shape,
            )

        if isinstance(output, dict):
            if "masks" in output:
                scores = None

                if "scores" in output:
                    scores = output["scores"]
                elif "iou_predictions" in output:
                    scores = output["iou_predictions"]

                return self._parse_masks_and_scores(
                    masks=output["masks"],
                    scores=scores,
                    image_shape=image_shape,
                )

            if "segmentation" in output or "mask" in output:
                return self._parse_list_output([output], image_shape)

        return self._parse_masks_and_scores(
            masks=output,
            scores=None,
            image_shape=image_shape,
        )

    def _parse_list_output(
        self,
        items: list[Any],
        image_shape: tuple[int, int],
    ) -> list[SAM2Detection]:
        detections: list[SAM2Detection] = []

        for item in items:
            if not isinstance(item, dict):
                continue

            raw_mask = item.get("segmentation", item.get("mask"))

            if raw_mask is None:
                continue

            mask = to_mask_uint8(raw_mask)

            if mask.shape[:2] != image_shape:
                mask = resize_mask_nearest(mask, image_shape)

            area = int(np.count_nonzero(mask))

            if area < self.config.min_area:
                continue

            raw_bbox = item.get("bbox")

            if raw_bbox is None:
                bbox = mask_to_bbox(mask)
            else:
                bbox = sam_bbox_to_xyxy(raw_bbox)

            if bbox is None:
                continue

            confidence = float(
                item.get(
                    "predicted_iou",
                    item.get(
                        "stability_score",
                        item.get("score", 1.0),
                    ),
                )
            )

            detections.append(
                SAM2Detection(
                    bbox=bbox,
                    confidence=confidence,
                    mask=mask,
                    metadata={
                        "area": area,
                    },
                )
            )

        return detections

    def _parse_masks_and_scores(
        self,
        masks: Any,
        scores: Any,
        image_shape: tuple[int, int],
    ) -> list[SAM2Detection]:
        masks_np = to_numpy(masks)

        while masks_np.ndim > 3:
            masks_np = masks_np[0]

        if masks_np.ndim == 2:
            masks_np = masks_np[None, ...]

        if masks_np.ndim != 3:
            raise ValueError(
                f"Unsupported SAM2 mask output shape: {masks_np.shape}"
            )

        scores_np = None

        if scores is not None:
            scores_np = np.asarray(to_numpy(scores)).reshape(-1)

        detections: list[SAM2Detection] = []

        for idx, mask_item in enumerate(masks_np):
            mask = self._threshold_mask(mask_item)

            if mask.shape[:2] != image_shape:
                mask = resize_mask_nearest(mask, image_shape)

            area = int(np.count_nonzero(mask))

            if area < self.config.min_area:
                continue

            bbox = mask_to_bbox(mask)

            if bbox is None:
                continue

            confidence = 1.0

            if scores_np is not None and idx < len(scores_np):
                confidence = float(scores_np[idx])

            detections.append(
                SAM2Detection(
                    bbox=bbox,
                    confidence=confidence,
                    mask=mask,
                    metadata={
                        "area": area,
                    },
                )
            )

        return detections

    def _threshold_mask(
        self,
        mask: Any,
    ) -> np.ndarray:
        mask_np = to_numpy(mask)

        if mask_np.dtype == np.bool_:
            return mask_np.astype(np.uint8) * 255

        if mask_np.dtype != np.uint8:
            return (mask_np > self.config.mask_threshold).astype(np.uint8) * 255

        if mask_np.size > 0 and mask_np.max() <= 1:
            return mask_np * 255

        return np.where(mask_np > 0, 255, 0).astype(np.uint8)

    def _build_auto_watermark_prompts(
        self,
        image: np.ndarray,
    ) -> dict[str, Any]:
        """
        Convert watermark search options into SAM2 visual prompts.

        This looks for:
            dark watermark
            light watermark
            low-opacity watermark
            text-like watermark
        """

        cv2 = lazy_import_cv2()

        gray = self._to_gray(image)

        prompt_masks: list[np.ndarray] = []

        if self.config.prompt_dark_watermark:
            prompt_masks.append(self._dark_prompt_mask(gray))

        if self.config.prompt_light_watermark:
            prompt_masks.append(self._light_prompt_mask(gray))

        if self.config.prompt_low_opacity_watermark:
            prompt_masks.append(self._low_opacity_prompt_mask(gray))

        if self.config.prompt_text_watermark:
            prompt_masks.append(self._text_like_prompt_mask(gray))

        if not prompt_masks:
            return {}

        combined = np.zeros_like(gray, dtype=np.uint8)

        for mask in prompt_masks:
            combined = np.maximum(combined, mask)

        kernel_size = self._odd_at_least(
            self.config.morph_kernel_size,
            minimum=3,
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )

        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

        boxes = self._prompt_mask_to_boxes(
            mask=combined,
            image_shape=image.shape[:2],
        )

        points = []
        point_labels = []

        for box in boxes:
            x1, y1, x2, y2 = box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            points.append([cx, cy])
            point_labels.append(1)

        return {
            "boxes": boxes,
            "points": np.asarray(points, dtype=np.float32) if points else None,
            "point_labels": np.asarray(point_labels, dtype=np.int32)
            if point_labels
            else None,
        }

    def _to_gray(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        cv2 = lazy_import_cv2()

        if image.ndim == 2:
            gray = image

        elif image.ndim == 3 and image.shape[-1] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        elif image.ndim == 3 and image.shape[-1] == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        if gray.dtype != np.uint8:
            gray = np.clip(gray, 0, 255).astype(np.uint8)

        return gray

    def _dark_prompt_mask(
        self,
        gray: np.ndarray,
    ) -> np.ndarray:
        sensitivity = self._normalized_prompt_sensitivity()

        percentile = self.config.dark_percentile + sensitivity * 8.0
        percentile = float(np.clip(percentile, 1.0, 35.0))

        threshold = np.percentile(gray, percentile)
        mask = np.where(gray <= threshold, 255, 0).astype(np.uint8)

        return self._cleanup_prompt_mask(mask)

    def _light_prompt_mask(
        self,
        gray: np.ndarray,
    ) -> np.ndarray:
        sensitivity = self._normalized_prompt_sensitivity()

        percentile = self.config.light_percentile - sensitivity * 8.0
        percentile = float(np.clip(percentile, 65.0, 99.0))

        threshold = np.percentile(gray, percentile)
        mask = np.where(gray >= threshold, 255, 0).astype(np.uint8)

        return self._cleanup_prompt_mask(mask)

    def _low_opacity_prompt_mask(
        self,
        gray: np.ndarray,
    ) -> np.ndarray:
        cv2 = lazy_import_cv2()

        sensitivity = self._normalized_prompt_sensitivity()

        edge_low = int(self.config.low_opacity_edge_low - sensitivity * 20)
        edge_high = int(self.config.low_opacity_edge_high - sensitivity * 40)

        edge_low = max(5, edge_low)
        edge_high = max(edge_low + 10, edge_high)

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        edges = cv2.Canny(
            blurred,
            edge_low,
            edge_high,
        )

        return self._cleanup_prompt_mask(edges)

    def _text_like_prompt_mask(
        self,
        gray: np.ndarray,
    ) -> np.ndarray:
        cv2 = lazy_import_cv2()

        sensitivity = self._normalized_prompt_sensitivity()

        adaptive_c = int(9 - sensitivity * 4)
        adaptive_c = max(2, adaptive_c)

        edge_low = int(50 - sensitivity * 25)
        edge_high = int(150 - sensitivity * 50)

        edge_low = max(5, edge_low)
        edge_high = max(edge_low + 10, edge_high)

        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        adaptive = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            adaptive_c,
        )

        edges = cv2.Canny(
            blurred,
            edge_low,
            edge_high,
        )

        mask = np.maximum(adaptive, edges)

        return self._cleanup_prompt_mask(mask)

    def _cleanup_prompt_mask(
        self,
        mask: np.ndarray,
    ) -> np.ndarray:
        cv2 = lazy_import_cv2()

        kernel_size = self._odd_at_least(
            self.config.morph_kernel_size,
            minimum=3,
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        if self.config.dilate_iterations > 0:
            mask = cv2.dilate(
                mask,
                kernel,
                iterations=self.config.dilate_iterations,
            )

        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

        return mask

    def _prompt_mask_to_boxes(
        self,
        mask: np.ndarray,
        image_shape: tuple[int, int],
    ) -> list[tuple[int, int, int, int]]:
        cv2 = lazy_import_cv2()

        height, width = image_shape
        image_area = height * width
        max_area = image_area * float(self.config.prompt_max_area_ratio)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        boxes: list[tuple[int, int, int, int]] = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            area = w * h

            if area < self.config.prompt_min_area:
                continue

            if area > max_area:
                continue

            boxes.append((int(x), int(y), int(x + w), int(y + h)))

        boxes = sorted(
            boxes,
            key=lambda box: (box[2] - box[0]) * (box[3] - box[1]),
            reverse=True,
        )

        return boxes[: self.config.prompt_max_boxes]

    def _normalized_prompt_sensitivity(self) -> float:
        return float(np.clip(self.config.prompt_sensitivity, 0.0, 100.0)) / 100.0

    def _debug_safe_prompts(
        self,
        prompts: dict[str, Any],
    ) -> dict[str, Any]:
        safe: dict[str, Any] = {}

        for key, value in prompts.items():
            if value is None:
                safe[key] = None
                continue

            try:
                array = np.asarray(value)
                safe[key] = {
                    "shape": tuple(array.shape),
                    "dtype": str(array.dtype),
                }
            except Exception:
                safe[key] = str(type(value))

        return safe

    @staticmethod
    def _odd_at_least(
        value: int,
        minimum: int,
    ) -> int:
        value = max(int(value), minimum)

        if value % 2 == 0:
            value += 1

        return value


def lazy_import_cv2() -> Any:
    try:
        return importlib.import_module("cv2")

    except ImportError as exc:
        raise ImportError(
            "OpenCV is required for SAM2 automatic watermark prompts. "
            "Install it with: pip install opencv-python"
        ) from exc