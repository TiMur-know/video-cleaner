# pipelines/image_pipeline.py

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import importlib
import numpy as np

from pipelines.preprocessing_pipeline import (
    PreprocessingPipeline,
    PreprocessingPipelineConfig,
)
from pipelines.detection_pipeline import (
    DetectionPipeline,
    DetectionPipelineConfig,
)
from pipelines.mask_processing_pipeline import (
    MaskProcessingPipeline,
    MaskProcessingPipelineConfig,
)
from pipelines.inpainting_pipeline import (
    InpaintingPipeline,
    InpaintingPipelineConfig,
)
from pipelines.postprocessing_pipeline import (
    PostprocessingPipeline,
    PostprocessingPipelineConfig,
)
from pipelines.utils import is_cancelled


_MODULE_CACHE: dict[str, Any] = {}


def lazy_import(module_name: str) -> Any:
    if module_name not in _MODULE_CACHE:
        try:
            _MODULE_CACHE[module_name] = importlib.import_module(module_name)
        except ImportError as exc:
            raise ImportError(
                f"Missing optional dependency '{module_name}'."
            ) from exc

    return _MODULE_CACHE[module_name]


@dataclass(slots=True)
class ImagePipelineConfig:
    input_path: str = "data/inputs/input.png"
    output_path: str = "data/outputs/output.png"
    mask_output_path: str | None = "data/masks/mask.png"

    preprocessing: PreprocessingPipelineConfig = field(
        default_factory=PreprocessingPipelineConfig
    )
    detection: DetectionPipelineConfig = field(
        default_factory=DetectionPipelineConfig
    )
    mask_processing: MaskProcessingPipelineConfig = field(
        default_factory=MaskProcessingPipelineConfig
    )
    inpainting: InpaintingPipelineConfig = field(
        default_factory=InpaintingPipelineConfig
    )
    postprocessing: PostprocessingPipelineConfig = field(
        default_factory=PostprocessingPipelineConfig
    )

    save_mask: bool = True
    save_output: bool = True

    # cv2 loads images as BGR.
    color_order: str = "bgr"


@dataclass(slots=True)
class ImagePipelineResult:
    original_image: np.ndarray
    preprocessed_image: np.ndarray
    mask: np.ndarray
    inpainted_image: np.ndarray
    final_image: np.ndarray

    output_path: str | None = None
    mask_output_path: str | None = None

    preprocessing_metadata: dict[str, Any] = field(default_factory=dict)
    detection_metadata: dict[str, Any] = field(default_factory=dict)
    mask_processing_metadata: dict[str, Any] = field(default_factory=dict)
    inpainting_metadata: dict[str, Any] = field(default_factory=dict)
    postprocessing_metadata: dict[str, Any] = field(default_factory=dict)

    cancelled: bool = False
    cancelled_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ImagePipeline:
    """
    Full image watermark-removal pipeline.

    Flow:
        load image
            ↓
        preprocessing
            ↓
        detection
            ↓
        mask_processing
            ↓
        inpainting
            ↓
        postprocessing
            ↓
        save final image
    """

    name = "image_pipeline"

    def __init__(
        self,
        config: ImagePipelineConfig | None = None,
        yolo_model: Any | None = None,
        sam2_model: Any | None = None,
        easy_ocr_reader: Any | None = None,
        paddle_ocr_model: Any | None = None,
        lama_model: Any | None = None,
        stable_diffusion_pipe: Any | None = None,
        sdxl_pipe: Any | None = None,
        flux_pipe: Any | None = None,
    ) -> None:
        self.config = config or ImagePipelineConfig()

        self.preprocessing_pipeline = PreprocessingPipeline(
            self.config.preprocessing
        )

        self.detection_pipeline = DetectionPipeline(
            self.config.detection,
            yolo_model=yolo_model,
            sam2_model=sam2_model,
            easy_ocr_reader=easy_ocr_reader,
            paddle_ocr_model=paddle_ocr_model,
        )

        self.mask_processing_pipeline = MaskProcessingPipeline(
            self.config.mask_processing
        )

        self.inpainting_pipeline = InpaintingPipeline(
            self.config.inpainting,
            lama_model=lama_model,
            stable_diffusion_pipe=stable_diffusion_pipe,
            sdxl_pipe=sdxl_pipe,
            flux_pipe=flux_pipe,
        )

        self.postprocessing_pipeline = PostprocessingPipeline(
            self.config.postprocessing
        )

    def run(
        self,
        input_path: str | None = None,
        output_path: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> ImagePipelineResult:
        context = context or {}

        input_path = input_path or self.config.input_path
        output_path = output_path or self.config.output_path

        original_image = self._load_image(input_path)

        pipeline_context = {
            **context,
            "pipeline": self.name,
            "input_type": "image",
            "input_path": input_path,
            "output_path": output_path,
            "color_order": self.config.color_order,
        }

        if is_cancelled(pipeline_context):
            return self._cancelled_result(
                original_image=original_image,
                stage="before_preprocessing",
                output_path=output_path,
            )

        preprocessing_result = self.preprocessing_pipeline.run(
            original_image,
            context={
                **pipeline_context,
                "stage": "preprocessing",
            },
        )

        preprocessed_image = preprocessing_result.main_image

        if is_cancelled(pipeline_context):
            return self._cancelled_result(
                original_image=original_image,
                preprocessed_image=preprocessed_image,
                stage="after_preprocessing",
                output_path=output_path,
                preprocessing_metadata=preprocessing_result.metadata,
            )

        detection_result = self.detection_pipeline.run(
            preprocessed_image,
            context={
                **pipeline_context,
                "stage": "detection",
            },
        )

        if is_cancelled(pipeline_context):
            return self._cancelled_result(
                original_image=original_image,
                preprocessed_image=preprocessed_image,
                mask=detection_result.mask,
                stage="after_detection",
                output_path=output_path,
                preprocessing_metadata=preprocessing_result.metadata,
                detection_metadata=detection_result.metadata,
            )

        mask_processing_result = self.mask_processing_pipeline.run_image(
            image=preprocessed_image,
            rough_mask=detection_result.mask,
            boxes=getattr(detection_result, "boxes", None),
            detection_results=[detection_result],
            support_masks=[],
            context={
                **pipeline_context,
                "stage": "mask_processing",
            },
        )

        mask = mask_processing_result.mask

        if is_cancelled(pipeline_context):
            return self._cancelled_result(
                original_image=original_image,
                preprocessed_image=preprocessed_image,
                mask=mask,
                stage="after_mask_processing",
                output_path=output_path,
                preprocessing_metadata=preprocessing_result.metadata,
                detection_metadata=detection_result.metadata,
                mask_processing_metadata=mask_processing_result.metadata,
            )

        inpainting_result = self.inpainting_pipeline.run_image(
            image=original_image,
            mask=mask,
            context={
                **pipeline_context,
                "stage": "inpainting",
            },
        )

        if is_cancelled(pipeline_context) or getattr(inpainting_result, "cancelled", False):
            return self._cancelled_result(
                original_image=original_image,
                preprocessed_image=preprocessed_image,
                mask=mask,
                inpainted_image=inpainting_result.image,
                stage=getattr(inpainting_result, "cancelled_at", None) or "after_inpainting",
                output_path=output_path,
                preprocessing_metadata=preprocessing_result.metadata,
                detection_metadata=detection_result.metadata,
                mask_processing_metadata=mask_processing_result.metadata,
                inpainting_metadata=inpainting_result.metadata,
            )

        if inpainting_result.image is None:
            raise RuntimeError("Inpainting pipeline returned image=None")

        inpainted_image = inpainting_result.image

        postprocessing_result = self.postprocessing_pipeline.run_image(
            image=inpainted_image,
            mask=mask,
            original_image=original_image,
            context={
                **pipeline_context,
                "stage": "postprocessing",
            },
        )

        if is_cancelled(pipeline_context) or getattr(postprocessing_result, "cancelled", False):
            return self._cancelled_result(
                original_image=original_image,
                preprocessed_image=preprocessed_image,
                mask=mask,
                inpainted_image=inpainted_image,
                final_image=postprocessing_result.image,
                stage=getattr(postprocessing_result, "cancelled_at", None) or "after_postprocessing",
                output_path=output_path,
                preprocessing_metadata=preprocessing_result.metadata,
                detection_metadata=detection_result.metadata,
                mask_processing_metadata=mask_processing_result.metadata,
                inpainting_metadata=inpainting_result.metadata,
                postprocessing_metadata=postprocessing_result.metadata,
            )

        if postprocessing_result.image is None:
            raise RuntimeError("Postprocessing pipeline returned image=None")

        final_image = postprocessing_result.image

        saved_output_path = None
        saved_mask_path = None

        if self.config.save_output:
            saved_output_path = self._save_image(
                final_image,
                output_path,
            )

        if self.config.save_mask and self.config.mask_output_path is not None:
            saved_mask_path = self._save_image(
                mask,
                self.config.mask_output_path,
            )

        metadata = {
            "pipeline": self.name,
            "input_path": input_path,
            "output_path": saved_output_path,
            "mask_output_path": saved_mask_path,
            "original_shape": original_image.shape,
            "preprocessed_shape": preprocessed_image.shape,
            "final_shape": final_image.shape,
            "mask_area": int(np.count_nonzero(mask)),
            "mask_processing": mask_processing_result.metadata,
        }

        return ImagePipelineResult(
            original_image=original_image,
            preprocessed_image=preprocessed_image,
            mask=mask,
            inpainted_image=inpainted_image,
            final_image=final_image,
            output_path=saved_output_path,
            mask_output_path=saved_mask_path,
            preprocessing_metadata=preprocessing_result.metadata,
            detection_metadata=detection_result.metadata,
            mask_processing_metadata=mask_processing_result.metadata,
            inpainting_metadata=inpainting_result.metadata,
            postprocessing_metadata=postprocessing_result.metadata,
            cancelled=False,
            metadata=metadata,
        )

    def _cancelled_result(
        self,
        original_image: np.ndarray,
        stage: str,
        output_path: str,
        preprocessed_image: np.ndarray | None = None,
        mask: np.ndarray | None = None,
        inpainted_image: np.ndarray | None = None,
        final_image: np.ndarray | None = None,
        preprocessing_metadata: dict[str, Any] | None = None,
        detection_metadata: dict[str, Any] | None = None,
        mask_processing_metadata: dict[str, Any] | None = None,
        inpainting_metadata: dict[str, Any] | None = None,
        postprocessing_metadata: dict[str, Any] | None = None,
    ) -> ImagePipelineResult:
        preprocessed_image = preprocessed_image if preprocessed_image is not None else original_image

        if mask is None:
            mask = np.zeros(original_image.shape[:2], dtype=np.uint8)

        inpainted_image = inpainted_image if inpainted_image is not None else original_image
        final_image = final_image if final_image is not None else inpainted_image

        metadata = {
            "pipeline": self.name,
            "cancelled": True,
            "cancelled_at": stage,
            "output_path": output_path,
            "original_shape": original_image.shape,
            "preprocessed_shape": preprocessed_image.shape,
            "final_shape": final_image.shape,
            "mask_area": int(np.count_nonzero(mask)),
            "mask_processing": mask_processing_metadata or {},
        }

        return ImagePipelineResult(
            original_image=original_image,
            preprocessed_image=preprocessed_image,
            mask=mask,
            inpainted_image=inpainted_image,
            final_image=final_image,
            output_path=None,
            mask_output_path=None,
            preprocessing_metadata=preprocessing_metadata or {},
            detection_metadata=detection_metadata or {},
            mask_processing_metadata=mask_processing_metadata or {},
            inpainting_metadata=inpainting_metadata or {},
            postprocessing_metadata=postprocessing_metadata or {},
            cancelled=True,
            cancelled_at=stage,
            metadata=metadata,
        )

    def _load_image(self, path: str) -> np.ndarray:
        cv2 = lazy_import("cv2")

        image_path = Path(path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

        if image is None:
            raise RuntimeError(f"Could not read image: {path}")

        return image

    def _save_image(
        self,
        image: np.ndarray,
        path: str,
    ) -> str:
        cv2 = lazy_import("cv2")

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        image_to_save = self._prepare_image_for_save(image)

        success = cv2.imwrite(
            str(output_path),
            image_to_save,
        )

        if not success:
            raise RuntimeError(f"Failed to save image: {output_path}")

        return str(output_path)

    @staticmethod
    def _prepare_image_for_save(image: np.ndarray) -> np.ndarray:
        if image.dtype == np.uint8:
            return image

        if np.issubdtype(image.dtype, np.floating):
            if image.max() <= 1.0:
                return np.clip(image * 255.0, 0, 255).astype(np.uint8)

            return np.clip(image, 0, 255).astype(np.uint8)

        return np.clip(image, 0, 255).astype(np.uint8)


def run_image_pipeline(
    input_path: str,
    output_path: str = "data/outputs/output.png",
) -> ImagePipelineResult:
    pipeline = ImagePipeline(
        ImagePipelineConfig(
            input_path=input_path,
            output_path=output_path,
        )
    )

    return pipeline.run()
