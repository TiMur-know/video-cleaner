# detectors/grounding_dino_detector.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.logger import log_detector_event
from detectors.utils import (
    BBox,
    bbox_area,
    bbox_to_mask,
    check_model_source,
    check_packages,
    empty_mask,
    lazy_import,
    pad_bbox,
    to_rgb,
    validate_image,
)


@dataclass(slots=True)
class GroundingDINODetectorConfig:
    enabled: bool = True

    # Use tiny for faster CPU/debug runs.
    # Use base for better quality if performance is acceptable.
    model_id: str = "IDEA-Research/grounding-dino-tiny"

    device: str = "auto"

    # Lower these if it detects nothing.
    box_threshold: float = 0.25
    text_threshold: float = 0.20

    # Prompt used when caller does not pass one.
    text_prompt: str = "watermark. logo. transparent text. text overlay."

    # Filter tiny false positives.
    min_area: int = 20

    # Add padding around boxes before MobileSAM.
    box_padding: int = 4

    # Usually BGR because OpenCV images are BGR.
    input_color_order: str = "bgr"

    fallback_to_empty: bool = True
    strict_loading: bool = False

    # If True, result.mask is union of detected boxes.
    make_box_mask: bool = True

    # Runtime checks/logging.
    check_dependencies: bool = True
    log_events: bool = True

    required_packages: dict[str, str] = field(
        default_factory=lambda: {
            "torch": "torch",
            "transformers": "transformers",
            "pillow": "PIL.Image",
        }
    )

    # Only used when model_id points to a local model directory.
    local_model_required_files: tuple[str, ...] = ()


@dataclass(slots=True)
class GroundingDINODetectorResult:
    mask: np.ndarray
    boxes: list[BBox] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    detections: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class GroundingDINODetector:
    """
    Grounding DINO detector.

    Role in pipeline:
        image + text prompt -> candidate watermark boxes

    Then:
        boxes -> MobileSAM2Detector -> refined mask
    """

    name = "grounding_dino"

    def __init__(
        self,
        config: GroundingDINODetectorConfig | None = None,
        processor: Any | None = None,
        model: Any | None = None,
    ) -> None:
        self.config = config or GroundingDINODetectorConfig()
        self.processor = processor
        self.model = model
        self._loaded = processor is not None and model is not None

    def detect(
        self,
        image: np.ndarray,
        text_prompt: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> GroundingDINODetectorResult:
        context = context or {}

        self._log_detect_input(
            image=image,
            text_prompt=text_prompt,
            context=context,
        )

        validate_image(image, self.name)

        if not self.config.enabled:
            result = self._empty_result(
                image=image,
                reason="disabled",
                context=context,
            )
            self._log_detect_output(result)
            return result

        prompt = self._normalize_prompt(text_prompt or self.config.text_prompt)

        try:
            self._check_runtime_requirements()
            self._ensure_loaded()

        except Exception as exc:
            if self.config.strict_loading or not self.config.fallback_to_empty:
                raise

            result = self._empty_result(
                image=image,
                reason=f"model_load_failed: {type(exc).__name__}: {exc}",
                context={
                    **context,
                    "prompt": prompt,
                },
            )
            self._log_detect_output(result)
            return result

        try:
            torch = lazy_import("torch")
            pil_image_module = lazy_import("PIL.Image")

            image_rgb = to_rgb(
                image,
                input_color_order=self.config.input_color_order,
            )

            pil_image = pil_image_module.fromarray(image_rgb)

            # IMPORTANT:
            # Use text=prompt, not text=[[prompt]].
            # Nested list causes:
            # TypeError: TextEncodeInput must be Union[TextInputSequence, ...]
            inputs = self.processor(
                images=pil_image,
                text=prompt,
                return_tensors="pt",
            ).to(self._model_device())

            with torch.no_grad():
                outputs = self.model(**inputs)

            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=self.config.box_threshold,
                text_threshold=self.config.text_threshold,
                target_sizes=[image.shape[:2]],
            )

            raw_result = results[0]

            boxes: list[BBox] = []
            scores: list[float] = []
            labels: list[str] = []
            detections: list[dict[str, Any]] = []

            for raw_box, raw_score, raw_label in zip(
                raw_result.get("boxes", []),
                raw_result.get("scores", []),
                raw_result.get("labels", []),
            ):
                box = self._to_bbox(raw_box)

                if self.config.box_padding > 0:
                    box = pad_bbox(
                        box,
                        image_shape=image.shape[:2],
                        padding=self.config.box_padding,
                    )

                if bbox_area(box) < self.config.min_area:
                    continue

                score = float(raw_score.detach().cpu().item())

                if isinstance(raw_label, str):
                    label = raw_label
                else:
                    label = str(raw_label)

                boxes.append(box)
                scores.append(score)
                labels.append(label)
                detections.append(
                    {
                        "box": box,
                        "score": score,
                        "label": label,
                    }
                )

            mask = self._boxes_to_mask(
                boxes=boxes,
                image_shape=image.shape[:2],
            )

            result = GroundingDINODetectorResult(
                mask=mask,
                boxes=boxes,
                scores=scores,
                labels=labels,
                detections=detections,
                metadata={
                    "detector": self.name,
                    "enabled": True,
                    "model_id": self.config.model_id,
                    "input_shape": image.shape,
                    "prompt": prompt,
                    "box_threshold": self.config.box_threshold,
                    "text_threshold": self.config.text_threshold,
                    "box_padding": self.config.box_padding,
                    "min_area": self.config.min_area,
                    "box_count": len(boxes),
                    "mask_area": int(np.count_nonzero(mask)),
                    "device": self._model_device(),
                },
            )

            self._log_detect_output(result)
            return result

        except Exception as exc:
            if self.config.strict_loading or not self.config.fallback_to_empty:
                raise

            result = self._empty_result(
                image=image,
                reason=f"predict_failed: {type(exc).__name__}: {exc}",
                context={
                    **context,
                    "prompt": prompt,
                },
            )
            self._log_detect_output(result)
            return result

    def __call__(
        self,
        image: np.ndarray,
        text_prompt: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> GroundingDINODetectorResult:
        return self.detect(
            image=image,
            text_prompt=text_prompt,
            context=context,
        )

    def _check_runtime_requirements(self) -> None:
        if not self.config.check_dependencies:
            self._log_event(
                "dependency_check",
                {
                    "enabled": False,
                    "reason": "disabled_by_config",
                },
            )
            return

        check_packages(
            self.config.required_packages,
            component_name=self.name,
            strict=True,
            log=self.config.log_events,
        )

        check_model_source(
            self.config.model_id,
            component_name=self.name,
            local_required_files=self.config.local_model_required_files,
            strict=True,
            log=self.config.log_events,
        )

    def _ensure_loaded(self) -> None:
        if self._loaded:
            self._log_event(
                "load",
                {
                    "loaded": True,
                    "model_id": self.config.model_id,
                    "device": self._model_device(),
                    "reason": "already_loaded",
                },
            )
            return

        self._log_event(
            "load",
            {
                "loaded": False,
                "model_id": self.config.model_id,
                "device": self._resolve_device(),
                "reason": "loading",
            },
        )

        transformers = lazy_import("transformers")

        AutoProcessor = transformers.AutoProcessor
        AutoModelForZeroShotObjectDetection = (
            transformers.AutoModelForZeroShotObjectDetection
        )

        self.processor = AutoProcessor.from_pretrained(
            self.config.model_id,
        )

        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.config.model_id,
        )

        self.model.to(self._resolve_device())
        self.model.eval()

        self._loaded = True

        self._log_event(
            "load",
            {
                "loaded": True,
                "model_id": self.config.model_id,
                "device": self._model_device(),
                "reason": "loaded",
            },
        )

    def _resolve_device(self) -> str:
        if self.config.device != "auto":
            return self.config.device

        torch = lazy_import("torch")

        if torch.cuda.is_available():
            return "cuda"

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"

        return "cpu"

    def _model_device(self) -> str:
        if self.model is None:
            return self._resolve_device()

        return str(next(self.model.parameters()).device)

    def _normalize_prompt(self, prompt: str) -> str:
        """
        Grounding DINO works better when classes are separated by periods.

        Good:
            "watermark. logo. text overlay."

        Bad:
            "watermark logo text overlay"
        """
        prompt = prompt.strip()

        if not prompt:
            raise ValueError("Grounding DINO text prompt cannot be empty.")

        if not prompt.endswith("."):
            prompt += "."

        return prompt

    def _to_bbox(self, box: Any) -> BBox:
        values = box.detach().cpu().numpy().tolist()

        if len(values) != 4:
            raise ValueError(f"Invalid Grounding DINO box: {box}")

        x1, y1, x2, y2 = values

        return (
            int(round(x1)),
            int(round(y1)),
            int(round(x2)),
            int(round(y2)),
        )

    def _boxes_to_mask(
        self,
        boxes: list[BBox],
        image_shape: tuple[int, int],
    ) -> np.ndarray:
        if not self.config.make_box_mask:
            return empty_mask(image_shape)

        mask = empty_mask(image_shape)

        for box in boxes:
            mask = np.maximum(
                mask,
                bbox_to_mask(
                    bbox=box,
                    image_shape=image_shape,
                ),
            )

        return mask

    def _empty_result(
        self,
        image: np.ndarray,
        reason: str,
        context: dict[str, Any] | None = None,
    ) -> GroundingDINODetectorResult:
        mask = empty_mask(image)

        return GroundingDINODetectorResult(
            mask=mask,
            boxes=[],
            scores=[],
            labels=[],
            detections=[],
            metadata={
                "detector": self.name,
                "enabled": self.config.enabled,
                "empty": True,
                "reason": reason,
                "mask_area": 0,
                "context": context or {},
            },
        )

    def _log_event(
        self,
        event: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        if not self.config.log_events:
            return

        log_detector_event(
            self.name,
            event,
            payload or {},
        )

    def _log_detect_input(
        self,
        image: np.ndarray,
        text_prompt: str | None,
        context: dict[str, Any],
    ) -> None:
        self._log_event(
            "input",
            {
                "image": image,
                "text_prompt": text_prompt,
                "default_prompt": self.config.text_prompt,
                "resolved_prompt": text_prompt or self.config.text_prompt,
                "model_id": self.config.model_id,
                "device": self.config.device,
                "box_threshold": self.config.box_threshold,
                "text_threshold": self.config.text_threshold,
                "min_area": self.config.min_area,
                "box_padding": self.config.box_padding,
                "input_color_order": self.config.input_color_order,
                "make_box_mask": self.config.make_box_mask,
                "enabled": self.config.enabled,
                "loaded": self._loaded,
                "check_dependencies": self.config.check_dependencies,
                "context_keys": list(context.keys()),
            },
        )

    def _log_detect_output(
        self,
        result: GroundingDINODetectorResult,
    ) -> None:
        self._log_event(
            "output",
            {
                "enabled": result.metadata.get("enabled"),
                "empty": result.metadata.get("empty", False),
                "reason": result.metadata.get("reason"),
                "box_count": len(result.boxes),
                "mask": result.mask,
                "mask_area": result.metadata.get("mask_area"),
                "labels": result.labels[:10],
                "scores": [
                    round(float(score), 4)
                    for score in result.scores[:10]
                ],
                "boxes_preview": result.boxes[:5],
                "prompt": result.metadata.get("prompt")
                or result.metadata.get("context", {}).get("prompt"),
                "device": result.metadata.get("device"),
                "model_id": result.metadata.get("model_id"),
            },
        )
