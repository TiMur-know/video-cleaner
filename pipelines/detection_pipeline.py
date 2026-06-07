# pipelines/detection_pipeline.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from detectors.anomaly_detector import AnomalyDetector, AnomalyDetectorConfig
from detectors.easy_ocr_detector import EasyOCRDetector, EasyOCRDetectorConfig
from detectors.fft_detector import FFTDetector, FFTDetectorConfig
from detectors.fusion_detector import FusionDetector, FusionDetectorConfig
from detectors.mobile_sam_2_detector import (
    MobileSAM2Detector,
    MobileSAM2DetectorConfig,
)
from detectors.opencv_detector import OpenCVDetector, OpenCVDetectorConfig
from detectors.paddle_ocr_detector import PaddleOCRDetector, PaddleOCRDetectorConfig
from detectors.sam2_detector import SAM2Detector, SAM2DetectorConfig
from detectors.yolo_detector import YOLODetector, YOLODetectorConfig
from detectors.utils import pad_bbox
from pipelines.utils import (
    normalize_mask,
    union_masks,
    has_non_empty_mask,
    non_empty_results,
    empty_result_names,
    summarize_prompts,
    mask_to_prompt_points,
    extract_boxes,
    resolve_detector_mode,
    collect_detections,
)


try:
    from detectors.grounding_dino_detector import (
        GroundingDINODetector,
        GroundingDINODetectorConfig,
    )

    HAS_GROUNDING_DINO = True

except ImportError:
    HAS_GROUNDING_DINO = False

    @dataclass(slots=True)
    class GroundingDINODetectorConfig:
        enabled: bool = False

    class GroundingDINODetector:
        name = "grounding_dino"

        def __init__(
            self,
            config: GroundingDINODetectorConfig | None = None,
        ) -> None:
            self.config = config or GroundingDINODetectorConfig()

        def detect(
            self,
            image: np.ndarray,
            context: dict[str, Any] | None = None,
        ) -> Any:
            raise ImportError(
                "GroundingDINO detector is not available. "
                "Create detectors/grounding_dino_detector.py or remove "
                "'grounding_dino' from proposal detectors."
            )


PROPOSAL_DETECTORS = [
    "opencv",
    "grounding_dino",
    "yolo",
    "fft",
    "anomaly",
    "paddle_ocr",
    "easy_ocr",
]

REFINER_DETECTORS = [
    "sam2",
    "mobile_sam_2",
]


@dataclass(slots=True)
class DetectionPipelineConfig:
    """
    Detection pipeline config.

    Flow:
        1. Run proposal detectors.
        2. Fuse proposal detector masks if 2+ non-empty masks exist.
        3. Build prompts from proposal boxes/masks.
        4. Run refiner detectors.
        5. Fuse refiner masks if 2+ non-empty masks exist.
        6. Final fusion from non-empty stage results.

    Important:
        Empty masks are ignored during fusion.
        This prevents empty SAM2 results from erasing valid OpenCV/FFT/anomaly masks.
    """

    opencv: OpenCVDetectorConfig = field(default_factory=OpenCVDetectorConfig)
    grounding_dino: GroundingDINODetectorConfig = field(
        default_factory=GroundingDINODetectorConfig
    )
    yolo: YOLODetectorConfig = field(default_factory=YOLODetectorConfig)
    fft: FFTDetectorConfig = field(default_factory=FFTDetectorConfig)
    anomaly: AnomalyDetectorConfig = field(default_factory=AnomalyDetectorConfig)
    paddle_ocr: PaddleOCRDetectorConfig = field(default_factory=PaddleOCRDetectorConfig)
    easy_ocr: EasyOCRDetectorConfig = field(default_factory=EasyOCRDetectorConfig)

    sam2: SAM2DetectorConfig = field(default_factory=SAM2DetectorConfig)
    mobile_sam_2: MobileSAM2DetectorConfig = field(
        default_factory=MobileSAM2DetectorConfig
    )

    fusion: FusionDetectorConfig = field(default_factory=FusionDetectorConfig)

    enable_opencv: bool = True
    enable_grounding_dino: bool = False
    enable_yolo: bool = False
    enable_fft: bool = True
    enable_anomaly: bool = True
    enable_paddle_ocr: bool = False
    enable_easy_ocr: bool = False

    enable_sam2: bool = False
    enable_mobile_sam_2: bool = False

    enable_fusion: bool = True

    # If True, SAM2 / MobileSAM2 receive boxes/points from proposal detectors.
    use_detector_boxes_as_refiner_prompts: bool = True

    # Points sampled from proposal masks for SAM2 / MobileSAM2.
    max_prompt_points_per_mask: int = 20

    # Expand proposal boxes before sending them to SAM-style refiners.
    # Text and logos often need surrounding antialias/shadow pixels removed too.
    refiner_prompt_box_padding: int = 4


@dataclass(slots=True)
class DetectionPipelineResult:
    mask: np.ndarray
    detector_results: dict[str, Any] = field(default_factory=dict)
    detections: list[Any] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class DetectionPipeline:
    """
    Runs watermark detection.

    Behavior:
        No detectors:
            empty mask

        Proposal only:
            proposal detector mask or proposal fusion

        Refiner only:
            refiner detector mask or refiner fusion

        Proposal + refiner:
            proposals create prompts
            refiners improve proposal masks
            final fusion combines non-empty results
    """

    name = "detection_pipeline"

    def __init__(
        self,
        config: DetectionPipelineConfig | None = None,
        yolo_model: Any | None = None,
        sam2_model: Any | None = None,
        mobile_sam_2_model: Any | None = None,
        easy_ocr_reader: Any | None = None,
        paddle_ocr_model: Any | None = None,
        grounding_dino_model: Any | None = None,
        grounding_dino_processor: Any | None = None,
    ) -> None:
        self.config = config or DetectionPipelineConfig()

        self.opencv = OpenCVDetector(self.config.opencv)

        self.grounding_dino = GroundingDINODetector(
            self.config.grounding_dino
        )

        self.yolo = YOLODetector(
            self.config.yolo,
            model=yolo_model,
        )

        self.fft = FFTDetector(self.config.fft)
        self.anomaly = AnomalyDetector(self.config.anomaly)

        self.paddle_ocr = PaddleOCRDetector(
            self.config.paddle_ocr,
            ocr_model=paddle_ocr_model,
        )

        self.easy_ocr = EasyOCRDetector(
            self.config.easy_ocr,
            reader=easy_ocr_reader,
        )

        self.sam2 = SAM2Detector(
            self.config.sam2,
            model=sam2_model,
        )

        self.mobile_sam_2 = MobileSAM2Detector(
            self.config.mobile_sam_2,
            model=mobile_sam_2_model,
        )

        self.fusion = FusionDetector(self.config.fusion)

        if grounding_dino_model is not None:
            self.grounding_dino.model = grounding_dino_model

        if grounding_dino_processor is not None:
            self.grounding_dino.processor = grounding_dino_processor

    def run(
        self,
        image: np.ndarray,
        context: dict[str, Any] | None = None,
    ) -> DetectionPipelineResult:
        context = context or {}

        enabled_detectors = self._enabled_detector_names()
        proposal_detectors = self._proposal_detector_names(enabled_detectors)
        refiner_detectors = self._refiner_detector_names(enabled_detectors)

        if not enabled_detectors:
            final_mask = np.zeros(image.shape[:2], dtype=np.uint8)

            return DetectionPipelineResult(
                mask=final_mask,
                detector_results={},
                detections=[],
                metadata={
                    "pipeline": self.name,
                    "enabled_detectors": [],
                    "proposal_detectors": [],
                    "refiner_detectors": [],
                    "detector_mode": "disabled",
                    "proposal_fusion_enabled": False,
                    "refiner_fusion_enabled": False,
                    "final_fusion_enabled": False,
                    "num_detector_results": 0,
                    "num_detections": 0,
                    "mask_area": 0,
                    "has_grounding_dino_file": HAS_GROUNDING_DINO,
                },
            )

        detector_results = self._run_detectors(
            image=image,
            proposal_detectors=proposal_detectors,
            refiner_detectors=refiner_detectors,
            context=context,
        )

        final_mask = self._build_final_mask(
            detector_results=detector_results,
            image_shape=image.shape[:2],
            proposal_detectors=proposal_detectors,
            refiner_detectors=refiner_detectors,
        )

        detections = self._collect_detections(detector_results)

        metadata = {
            "pipeline": self.name,
            "enabled_detectors": enabled_detectors,
            "proposal_detectors": proposal_detectors,
            "refiner_detectors": refiner_detectors,
            "detector_mode": self._resolve_detector_mode(
                proposal_detectors=proposal_detectors,
                refiner_detectors=refiner_detectors,
            ),
            "proposal_fusion_enabled": "proposal_fusion" in detector_results,
            "refiner_fusion_enabled": "refiner_fusion" in detector_results,
            "final_fusion_enabled": "fusion" in detector_results,
            "fusion_enabled": self.config.enable_fusion and self.config.fusion.enabled,
            "num_detector_results": len(detector_results),
            "num_detections": len(detections),
            "mask_area": int(np.count_nonzero(final_mask)),
            "has_grounding_dino_file": HAS_GROUNDING_DINO,
            "empty_detector_results": self._empty_result_names(detector_results),
            "non_empty_detector_results": list(
                self._non_empty_results(detector_results).keys()
            ),
        }

        return DetectionPipelineResult(
            mask=final_mask,
            detector_results=detector_results,
            detections=detections,
            metadata=metadata,
        )

    def _enabled_detector_names(self) -> list[str]:
        names: list[str] = []

        if self.config.enable_opencv and self.config.opencv.enabled:
            names.append("opencv")

        if (
            self.config.enable_grounding_dino
            and self.config.grounding_dino.enabled
        ):
            names.append("grounding_dino")

        if self.config.enable_yolo and self.config.yolo.enabled:
            names.append("yolo")

        if self.config.enable_fft and self.config.fft.enabled:
            names.append("fft")

        if self.config.enable_anomaly and self.config.anomaly.enabled:
            names.append("anomaly")

        if self.config.enable_paddle_ocr and self.config.paddle_ocr.enabled:
            names.append("paddle_ocr")

        if self.config.enable_easy_ocr and self.config.easy_ocr.enabled:
            names.append("easy_ocr")

        if self.config.enable_sam2 and self.config.sam2.enabled:
            names.append("sam2")

        if (
            self.config.enable_mobile_sam_2
            and self.config.mobile_sam_2.enabled
        ):
            names.append("mobile_sam_2")

        return names

    def _proposal_detector_names(
        self,
        enabled_detectors: list[str],
    ) -> list[str]:
        return [
            name
            for name in enabled_detectors
            if name in PROPOSAL_DETECTORS
        ]

    def _refiner_detector_names(
        self,
        enabled_detectors: list[str],
    ) -> list[str]:
        return [
            name
            for name in enabled_detectors
            if name in REFINER_DETECTORS
        ]

    def _run_detectors(
        self,
        image: np.ndarray,
        proposal_detectors: list[str],
        refiner_detectors: list[str],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        detector_results: dict[str, Any] = {}

        proposal_results = self._run_proposal_detectors(
            image=image,
            proposal_detectors=proposal_detectors,
            context=context,
        )

        detector_results.update(proposal_results)

        proposal_fusion = self._build_stage_fusion(
            stage_name="proposal",
            detector_results=proposal_results,
            image_shape=image.shape[:2],
        )

        if proposal_fusion is not None:
            detector_results["proposal_fusion"] = proposal_fusion

        # Keep all proposal results for prompt creation.
        # GroundingDINO can have boxes but an empty mask.
        prompt_sources = dict(proposal_results)

        if proposal_fusion is not None:
            prompt_sources["proposal_fusion"] = proposal_fusion

        prompts = self._build_refiner_prompts(
            detector_results=prompt_sources,
            image_shape=image.shape[:2],
        )

        refiner_results = self._run_refiner_detectors(
            image=image,
            refiner_detectors=refiner_detectors,
            prompts=prompts,
            context=context,
            prompt_source_detectors=list(prompt_sources.keys()),
        )

        detector_results.update(refiner_results)

        refiner_fusion = self._build_stage_fusion(
            stage_name="refiner",
            detector_results=refiner_results,
            image_shape=image.shape[:2],
        )

        if refiner_fusion is not None:
            detector_results["refiner_fusion"] = refiner_fusion

        return detector_results

    def _run_proposal_detectors(
        self,
        image: np.ndarray,
        proposal_detectors: list[str],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        results: dict[str, Any] = {}

        for detector_name in proposal_detectors:
            detector = self._get_proposal_detector(detector_name)

            results[detector_name] = detector.detect(
                image,
                context={
                    **context,
                    "detector": detector_name,
                    "detector_stage": "proposal",
                },
            )

        return results

    def _run_refiner_detectors(
        self,
        image: np.ndarray,
        refiner_detectors: list[str],
        prompts: dict[str, Any],
        context: dict[str, Any],
        prompt_source_detectors: list[str],
    ) -> dict[str, Any]:
        results: dict[str, Any] = {}

        for detector_name in refiner_detectors:
            if detector_name == "sam2":
                results["sam2"] = self.sam2.detect(
                    image,
                    prompts=prompts,
                    context={
                        **context,
                        "detector": "sam2",
                        "detector_stage": "refiner",
                        "prompt_source_detectors": prompt_source_detectors,
                    },
                )
                continue

            if detector_name == "mobile_sam_2":
                results["mobile_sam_2"] = self.mobile_sam_2.detect(
                    image,
                    prompts=prompts,
                    context={
                        **context,
                        "detector": "mobile_sam_2",
                        "detector_stage": "refiner",
                        "prompt_source_detectors": prompt_source_detectors,
                    },
                )
                continue

            raise ValueError(f"Unsupported refiner detector: {detector_name}")

        return results

    def _get_proposal_detector(
        self,
        name: str,
    ) -> Any:
        if name == "opencv":
            return self.opencv

        if name == "grounding_dino":
            return self.grounding_dino

        if name == "yolo":
            return self.yolo

        if name == "fft":
            return self.fft

        if name == "anomaly":
            return self.anomaly

        if name == "paddle_ocr":
            return self.paddle_ocr

        if name == "easy_ocr":
            return self.easy_ocr

        raise ValueError(f"Unsupported proposal detector: {name}")

    def _build_stage_fusion(
        self,
        stage_name: str,
        detector_results: dict[str, Any],
        image_shape: tuple[int, int],
    ) -> Any | None:
        """
        Fuse proposal/refiner stage only if 2+ non-empty masks exist.

        Empty masks are ignored, so empty SAM2 cannot erase valid masks.
        """

        non_empty_results = self._non_empty_results(detector_results)

        if len(non_empty_results) < 2:
            return None

        if not self.config.enable_fusion or not self.config.fusion.enabled:
            return None

        fusion_result = self.fusion.detect(
            inputs=non_empty_results,
            image_shape=image_shape,
        )

        metadata = getattr(fusion_result, "metadata", {})

        if isinstance(metadata, dict):
            metadata["stage"] = stage_name
            metadata["stage_fusion"] = True
            metadata["source_detectors"] = list(non_empty_results.keys())
            metadata["ignored_empty_detectors"] = [
                name
                for name in detector_results
                if name not in non_empty_results
            ]

        return fusion_result

    def _build_refiner_prompts(
        self,
        detector_results: dict[str, Any],
        image_shape: tuple[int, int],
    ) -> dict[str, Any]:
        if not self.config.use_detector_boxes_as_refiner_prompts:
            return {}

        boxes: list[Any] = []
        points: list[list[int]] = []
        point_labels: list[int] = []

        for result in detector_results.values():
            boxes.extend(
                self._pad_prompt_boxes(
                    boxes=extract_boxes(result),
                    image_shape=image_shape,
                )
            )

            mask = getattr(result, "mask", None)

            if mask is not None and int(np.count_nonzero(mask)) > 0:
                mask_points = mask_to_prompt_points(
                    mask, max_points=self.config.max_prompt_points_per_mask
                )

                for point in mask_points:
                    points.append(point)
                    point_labels.append(1)

        prompts: dict[str, Any] = {}

        if boxes:
            prompts["boxes"] = boxes

        if points:
            prompts["points"] = np.asarray(points, dtype=np.float32)
            prompts["point_labels"] = np.asarray(point_labels, dtype=np.int32)

        return prompts

    def _pad_prompt_boxes(
        self,
        boxes: list[Any],
        image_shape: tuple[int, int],
    ) -> list[Any]:
        padding = max(0, int(self.config.refiner_prompt_box_padding))

        if padding <= 0:
            return boxes

        padded_boxes: list[Any] = []

        for box in boxes:
            try:
                padded_boxes.append(
                    pad_bbox(
                        bbox=tuple(map(int, box)),
                        image_shape=image_shape,
                        padding=padding,
                    )
                )
            except Exception:
                padded_boxes.append(box)

        return padded_boxes

    def _extract_boxes(
        self,
        result: Any,
    ) -> list[Any]:
        return extract_boxes(result)

    def _mask_to_prompt_points(
        self,
        mask: np.ndarray,
        max_points: int = 20,
    ) -> list[list[int]]:
        return mask_to_prompt_points(mask, max_points=max_points)

    def _build_final_mask(
        self,
        detector_results: dict[str, Any],
        image_shape: tuple[int, int],
        proposal_detectors: list[str],
        refiner_detectors: list[str],
    ) -> np.ndarray:
        if not detector_results:
            return np.zeros(image_shape, dtype=np.uint8)

        final_inputs = self._final_fusion_inputs(
            detector_results=detector_results,
            proposal_detectors=proposal_detectors,
            refiner_detectors=refiner_detectors,
        )

        if not final_inputs:
            return np.zeros(image_shape, dtype=np.uint8)

        if len(final_inputs) == 1:
            result = next(iter(final_inputs.values()))
            mask = getattr(result, "mask", None)

            if mask is None:
                return np.zeros(image_shape, dtype=np.uint8)

            return normalize_mask(mask, image_shape)

        if self.config.enable_fusion and self.config.fusion.enabled:
            fusion_result = self.fusion.detect(
                inputs=final_inputs,
                image_shape=image_shape,
            )

            metadata = getattr(fusion_result, "metadata", {})

            if isinstance(metadata, dict):
                metadata["stage"] = "final"
                metadata["stage_fusion"] = False
                metadata["source_detectors"] = list(final_inputs.keys())
                metadata["source_count"] = len(final_inputs)

            detector_results["fusion"] = fusion_result

            return normalize_mask(fusion_result.mask, image_shape)

        masks = [getattr(r, "mask", None) for r in final_inputs.values()]
        masks = [m for m in masks if m is not None]

        if not masks:
            return np.zeros(image_shape, dtype=np.uint8)

        return union_masks(masks, image_shape)

    def _final_fusion_inputs(
        self,
        detector_results: dict[str, Any],
        proposal_detectors: list[str],
        refiner_detectors: list[str],
    ) -> dict[str, Any]:
        """
        Prefer stage-fused results when available.

        Empty detector results are ignored.
        This means failed/empty SAM2 cannot erase valid proposal masks.
        """

        inputs: dict[str, Any] = {}

        if "proposal_fusion" in detector_results and has_non_empty_mask(
            detector_results["proposal_fusion"]
        ):
            inputs["proposal_fusion"] = detector_results["proposal_fusion"]
        else:
            for name in proposal_detectors:
                result = detector_results.get(name)

                if result is not None and has_non_empty_mask(result):
                    inputs[name] = result

        if "refiner_fusion" in detector_results and has_non_empty_mask(
            detector_results["refiner_fusion"]
        ):
            inputs["refiner_fusion"] = detector_results["refiner_fusion"]
        else:
            for name in refiner_detectors:
                result = detector_results.get(name)

                if result is not None and has_non_empty_mask(result):
                    inputs[name] = result

        return inputs

    def _has_non_empty_mask(
        self,
        result: Any,
    ) -> bool:
        return has_non_empty_mask(result)

    def _non_empty_results(
        self,
        detector_results: dict[str, Any],
    ) -> dict[str, Any]:
        return non_empty_results(detector_results)

    def _empty_result_names(
        self,
        detector_results: dict[str, Any],
    ) -> list[str]:
        return empty_result_names(detector_results)

    def _normalize_mask(
        self,
        mask: np.ndarray,
        image_shape: tuple[int, int],
    ) -> np.ndarray:
        return normalize_mask(mask, image_shape)

    def _union_masks(
        self,
        masks: list[np.ndarray],
        image_shape: tuple[int, int],
    ) -> np.ndarray:
        return union_masks(masks, image_shape)

    def _resolve_detector_mode(
        self,
        proposal_detectors: list[str],
        refiner_detectors: list[str],
    ) -> str:
        return resolve_detector_mode(proposal_detectors, refiner_detectors)

    @staticmethod
    def _collect_detections(
        detector_results: dict[str, Any],
    ) -> list[Any]:
        return collect_detections(detector_results)
