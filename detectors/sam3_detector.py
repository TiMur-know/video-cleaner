# detectors/sam3_detector.py

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class SAM3DetectorConfig:
    enabled: bool = True

    # SAM3 uses text prompts. You can edit this from UI.
    prompt: str = (
        "watermark, logo watermark, transparent watermark, low opacity text, "
        "dark watermark, semi-transparent text, overlaid brand logo, repeated watermark pattern"
    )

    # Optional multiple prompts. If set, each prompt is run and masks are merged.
    prompts: list[str] = field(default_factory=list)

    device: str = "auto"

    confidence_threshold: float = 0.25
    mask_threshold: float = 0.5
    min_area: int = 25

    fallback_to_empty: bool = True
    strict_loading: bool = False

    # Optional injected model/processor factory.
    model_factory: Any | None = None


@dataclass(slots=True)
class SAM3DetectorResult:
    mask: np.ndarray
    boxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    detections: list[Any] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class SAM3Detector:
    """
    SAM3 text-prompt detector.

    Good for:
        - watermark
        - transparent watermark
        - low opacity text
        - dark watermark
        - logo watermark
        - repeated overlay pattern

    SAM3 official style:
        model = build_sam3_image_model()
        processor = Sam3Processor(model)
        state = processor.set_image(image)
        output = processor.set_text_prompt(state=state, prompt="watermark")
    """

    name = "sam3"

    def __init__(
        self,
        config: SAM3DetectorConfig | None = None,
        model: Any | None = None,
        processor: Any | None = None,
    ) -> None:
        self.config = config or SAM3DetectorConfig()
        self.model = model
        self.processor = processor

    def detect(
        self,
        image: np.ndarray,
        context: dict[str, Any] | None = None,
    ) -> SAM3DetectorResult:
        context = context or {}

        if image is None:
            raise ValueError("image is required")

        if not self.config.enabled:
            return self._empty_result(image, reason="disabled", context=context)

        try:
            self._ensure_loaded()
        except Exception as exc:
            if self.config.strict_loading or not self.config.fallback_to_empty:
                raise

            return self._empty_result(
                image=image,
                reason=f"model_load_failed: {type(exc).__name__}: {exc}",
                context=context,
            )

        prompts = self._resolved_prompts()

        if not prompts:
            return self._empty_result(
                image=image,
                reason="no_prompt",
                context=context,
            )

        masks: list[np.ndarray] = []
        boxes: list[tuple[int, int, int, int]] = []
        scores: list[float] = []
        labels: list[str] = []

        for prompt in prompts:
            try:
                output = self._run_prompt(image=image, prompt=prompt)
            except Exception as exc:
                if self.config.strict_loading or not self.config.fallback_to_empty:
                    raise

                continue

            prompt_mask, prompt_boxes, prompt_scores = self._parse_output(
                output=output,
                image_shape=image.shape[:2],
            )

            if prompt_mask is not None:
                masks.append(prompt_mask)

            boxes.extend(prompt_boxes)
            scores.extend(prompt_scores)
            labels.extend([prompt for _ in prompt_boxes])

        if not masks:
            return self._empty_result(
                image=image,
                reason="no_masks",
                context=context,
            )

        final_mask = self._merge_masks(masks, image.shape[:2])
        final_mask = self._filter_small_components(final_mask)

        return SAM3DetectorResult(
            mask=final_mask,
            boxes=boxes,
            scores=scores,
            labels=labels,
            detections=[],
            metadata={
                "detector": self.name,
                "enabled": True,
                "prompts": prompts,
                "mask_area": int(np.count_nonzero(final_mask)),
                "box_count": len(boxes),
                "confidence_threshold": self.config.confidence_threshold,
                "mask_threshold": self.config.mask_threshold,
                "min_area": self.config.min_area,
            },
        )

    def _ensure_loaded(self) -> None:
        if self.processor is not None:
            return

        if self.config.model_factory is not None:
            created = self.config.model_factory(self.config)

            if isinstance(created, tuple):
                self.model, self.processor = created
            else:
                self.processor = created

            return

        try:
            builder_module = importlib.import_module("sam3.model_builder")
            processor_module = importlib.import_module("sam3.model.sam3_image_processor")
        except ImportError as exc:
            raise ImportError(
                "SAM3 package is not installed. Install SAM3 first or pass "
                "a ready processor/model into SAM3Detector."
            ) from exc

        build_sam3_image_model = getattr(
            builder_module,
            "build_sam3_image_model",
        )
        processor_cls = getattr(
            processor_module,
            "Sam3Processor",
        )

        self.model = build_sam3_image_model()
        self.processor = processor_cls(self.model)

    def _run_prompt(
        self,
        image: np.ndarray,
        prompt: str,
    ) -> Any:
        if self.processor is None:
            raise RuntimeError("SAM3 processor is not loaded.")

        pil_image = self._to_pil_image(image)

        state = self.processor.set_image(pil_image)

        return self.processor.set_text_prompt(
            state=state,
            prompt=prompt,
        )

    def _resolved_prompts(self) -> list[str]:
        if self.config.prompts:
            return [
                prompt.strip()
                for prompt in self.config.prompts
                if prompt and prompt.strip()
            ]

        prompt = self.config.prompt.strip()

        if not prompt:
            return []

        # Allow comma-separated prompts from UI.
        return [
            item.strip()
            for item in prompt.split(",")
            if item.strip()
        ]

    def _parse_output(
        self,
        output: Any,
        image_shape: tuple[int, int],
    ) -> tuple[np.ndarray | None, list[tuple[int, int, int, int]], list[float]]:
        if not isinstance(output, dict):
            return None, [], []

        masks = output.get("masks")
        boxes = output.get("boxes")
        scores = output.get("scores")

        score_list = self._to_score_list(scores)
        mask = self._masks_to_single_mask(
            masks=masks,
            scores=score_list,
            image_shape=image_shape,
        )

        box_list = self._to_box_list(boxes, score_list)

        return mask, box_list, score_list

    def _masks_to_single_mask(
        self,
        masks: Any,
        scores: list[float],
        image_shape: tuple[int, int],
    ) -> np.ndarray | None:
        if masks is None:
            return None

        masks_np = self._to_numpy(masks)

        if masks_np.ndim == 2:
            masks_np = masks_np[None, ...]

        while masks_np.ndim > 3:
            masks_np = np.squeeze(masks_np, axis=0)

        if masks_np.ndim != 3:
            return None

        selected_masks: list[np.ndarray] = []

        for index, mask_item in enumerate(masks_np):
            score = scores[index] if index < len(scores) else 1.0

            if score < self.config.confidence_threshold:
                continue

            mask = self._to_binary_mask(mask_item)

            if mask.shape[:2] != image_shape:
                mask = self._resize_mask(mask, image_shape)

            selected_masks.append(mask)

        if not selected_masks:
            return None

        return self._merge_masks(selected_masks, image_shape)

    def _to_box_list(
        self,
        boxes: Any,
        scores: list[float],
    ) -> list[tuple[int, int, int, int]]:
        if boxes is None:
            return []

        boxes_np = self._to_numpy(boxes)

        if boxes_np.size == 0:
            return []

        boxes_np = boxes_np.reshape(-1, 4)

        output: list[tuple[int, int, int, int]] = []

        for index, box in enumerate(boxes_np):
            score = scores[index] if index < len(scores) else 1.0

            if score < self.config.confidence_threshold:
                continue

            x1, y1, x2, y2 = box.tolist()

            output.append(
                (
                    int(round(x1)),
                    int(round(y1)),
                    int(round(x2)),
                    int(round(y2)),
                )
            )

        return output

    def _to_binary_mask(
        self,
        mask: np.ndarray,
    ) -> np.ndarray:
        if mask.dtype == np.bool_:
            return mask.astype(np.uint8) * 255

        if mask.dtype != np.uint8:
            return (mask > self.config.mask_threshold).astype(np.uint8) * 255

        if mask.size > 0 and mask.max() <= 1:
            return mask * 255

        return np.where(mask > 0, 255, 0).astype(np.uint8)

    def _merge_masks(
        self,
        masks: list[np.ndarray],
        image_shape: tuple[int, int],
    ) -> np.ndarray:
        full_mask = np.zeros(image_shape, dtype=np.uint8)

        for mask in masks:
            if mask.shape[:2] != image_shape:
                mask = self._resize_mask(mask, image_shape)

            full_mask = np.maximum(full_mask, self._to_binary_mask(mask))

        return full_mask

    def _filter_small_components(
        self,
        mask: np.ndarray,
    ) -> np.ndarray:
        if self.config.min_area <= 0:
            return mask

        cv2 = self._lazy_import_cv2()

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )

        output = np.zeros_like(mask, dtype=np.uint8)

        for label_index in range(1, num_labels):
            area = int(stats[label_index, cv2.CC_STAT_AREA])

            if area < self.config.min_area:
                continue

            output[labels == label_index] = 255

        return output

    def _resize_mask(
        self,
        mask: np.ndarray,
        image_shape: tuple[int, int],
    ) -> np.ndarray:
        cv2 = self._lazy_import_cv2()

        height, width = image_shape

        resized = cv2.resize(
            mask,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )

        return self._to_binary_mask(resized)

    def _to_pil_image(
        self,
        image: np.ndarray,
    ) -> Any:
        Image = importlib.import_module("PIL.Image")

        if image.ndim == 2:
            return Image.fromarray(image)

        if image.ndim == 3 and image.shape[-1] == 3:
            # Your pipeline uses OpenCV BGR.
            rgb = image[..., ::-1]
            return Image.fromarray(rgb)

        if image.ndim == 3 and image.shape[-1] == 4:
            rgba = image[..., [2, 1, 0, 3]]
            return Image.fromarray(rgba)

        raise ValueError(f"Unsupported image shape for SAM3: {image.shape}")

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        try:
            import torch

            if isinstance(value, torch.Tensor):
                return value.detach().cpu().numpy()
        except Exception:
            pass

        return np.asarray(value)

    @staticmethod
    def _to_score_list(scores: Any) -> list[float]:
        if scores is None:
            return []

        try:
            array = SAM3Detector._to_numpy(scores).reshape(-1)
            return [float(item) for item in array]
        except Exception:
            return []

    @staticmethod
    def _lazy_import_cv2() -> Any:
        try:
            return importlib.import_module("cv2")
        except ImportError as exc:
            raise ImportError(
                "OpenCV is required for SAM3Detector mask postprocessing. "
                "Install it with: pip install opencv-python"
            ) from exc

    def _empty_result(
        self,
        image: np.ndarray,
        reason: str,
        context: dict[str, Any] | None = None,
    ) -> SAM3DetectorResult:
        return SAM3DetectorResult(
            mask=np.zeros(image.shape[:2], dtype=np.uint8),
            boxes=[],
            scores=[],
            labels=[],
            detections=[],
            metadata={
                "detector": self.name,
                "enabled": self.config.enabled,
                "empty": True,
                "reason": reason,
                "prompt": self.config.prompt,
                "prompts": self.config.prompts,
                "context": context or {},
            },
        )