# pipelines/postprocessing_pipeline.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from postprocessing.artifact_removal import (
    ArtifactRemovalConfig,
    ArtifactRemovalPostprocessor,
)
from postprocessing.color_matching import (
    ColorMatchingConfig,
    ColorMatchingPostprocessor,
)
from postprocessing.seam_blending import (
    SeamBlendingConfig,
    SeamBlendingPostprocessor,
)
from postprocessing.sharpening import (
    SharpeningConfig,
    SharpeningPostprocessor,
)
from postprocessing.temporal_smoothing import (
    TemporalSmoothingConfig,
    TemporalSmoothingPostprocessor,
)
from pipelines.utils import is_cancelled, get_optional_item


@dataclass(slots=True)
class PostprocessingPipelineConfig:
    enabled: bool = True

    enable_color_matching: bool = True
    enable_seam_blending: bool = True
    enable_artifact_removal: bool = True
    enable_sharpening: bool = True
    enable_temporal_smoothing: bool = True

    color_matching: ColorMatchingConfig = field(default_factory=ColorMatchingConfig)
    seam_blending: SeamBlendingConfig = field(default_factory=SeamBlendingConfig)
    artifact_removal: ArtifactRemovalConfig = field(default_factory=ArtifactRemovalConfig)
    sharpening: SharpeningConfig = field(default_factory=SharpeningConfig)
    temporal_smoothing: TemporalSmoothingConfig = field(
        default_factory=TemporalSmoothingConfig
    )


@dataclass(slots=True)
class PostprocessingPipelineResult:
    image: np.ndarray | None = None
    frames: list[np.ndarray] | None = None

    cancelled: bool = False
    cancelled_at: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)


class PostprocessingPipeline:
    """
    Final cleanup pipeline.

    Image flow:
        inpainted image
            ↓
        color matching
            ↓
        seam blending
            ↓
        artifact removal
            ↓
        sharpening
            ↓
        final image

    Video flow:
        inpainted frames
            ↓
        per-frame postprocessing
            ↓
        temporal smoothing
            ↓
        final frames

    Cancel support:
        Pass:

            context={
                "cancel_token": {"stop": False}
            }

        UI can set:

            cancel_token["stop"] = True

        This pipeline checks the flag between frames and stages.
    """

    name = "postprocessing_pipeline"

    def __init__(self, config: PostprocessingPipelineConfig | None = None) -> None:
        self.config = config or PostprocessingPipelineConfig()

        self.color_matching = ColorMatchingPostprocessor(
            self.config.color_matching
        )
        self.seam_blending = SeamBlendingPostprocessor(
            self.config.seam_blending
        )
        self.artifact_removal = ArtifactRemovalPostprocessor(
            self.config.artifact_removal
        )
        self.sharpening = SharpeningPostprocessor(
            self.config.sharpening
        )
        self.temporal_smoothing = TemporalSmoothingPostprocessor(
            self.config.temporal_smoothing
        )

    def run_image(
        self,
        image: np.ndarray,
        mask: np.ndarray | None = None,
        original_image: np.ndarray | None = None,
        context: dict[str, Any] | None = None,
    ) -> PostprocessingPipelineResult:
        context = context or {}

        if not self.config.enabled:
            return PostprocessingPipelineResult(
                image=image,
                metadata={
                    "pipeline": self.name,
                    "enabled": False,
                    "input_type": "image",
                },
            )

        if is_cancelled(context):
            return self._cancelled_image_result(
                image=image,
                stage="before_postprocessing",
            )

        output = image.copy()
        applied_steps: list[str] = []

        if self.config.enable_color_matching:
            if is_cancelled(context):
                return self._cancelled_image_result(output, "before_color_matching")

            output = self.color_matching(
                output,
                reference_image=original_image,
                mask=mask,
                context={
                    **context,
                    "stage": "color_matching",
                },
            )
            applied_steps.append("color_matching")

        if self.config.enable_seam_blending:
            if is_cancelled(context):
                return self._cancelled_image_result(output, "before_seam_blending")

            output = self.seam_blending(
                output,
                mask=mask,
                original_image=original_image,
                context={
                    **context,
                    "stage": "seam_blending",
                },
            )
            applied_steps.append("seam_blending")

        if self.config.enable_artifact_removal:
            if is_cancelled(context):
                return self._cancelled_image_result(output, "before_artifact_removal")

            output = self.artifact_removal(
                output,
                mask=mask,
                context={
                    **context,
                    "stage": "artifact_removal",
                },
            )
            applied_steps.append("artifact_removal")

        if self.config.enable_sharpening:
            if is_cancelled(context):
                return self._cancelled_image_result(output, "before_sharpening")

            output = self.sharpening(
                output,
                mask=mask,
                context={
                    **context,
                    "stage": "sharpening",
                },
            )
            applied_steps.append("sharpening")

        return PostprocessingPipelineResult(
            image=output,
            cancelled=False,
            metadata={
                "pipeline": self.name,
                "input_type": "image",
                "enabled": True,
                "cancelled": False,
                "applied_steps": applied_steps,
                "input_shape": image.shape,
                "output_shape": output.shape,
                "has_mask": mask is not None,
                "has_original_image": original_image is not None,
            },
        )

    def run_video(
        self,
        frames: list[np.ndarray],
        masks: list[np.ndarray] | None = None,
        original_frames: list[np.ndarray] | None = None,
        context: dict[str, Any] | None = None,
    ) -> PostprocessingPipelineResult:
        context = context or {}

        if not self.config.enabled:
            return PostprocessingPipelineResult(
                frames=frames,
                metadata={
                    "pipeline": self.name,
                    "enabled": False,
                    "input_type": "video",
                    "frame_count": len(frames),
                },
            )

        if is_cancelled(context):
            return self._cancelled_video_result(
                frames=[],
                stage="before_postprocessing",
            )

        processed_frames: list[np.ndarray] = []
        frame_metadata: list[dict[str, Any]] = []

        for index, frame in enumerate(frames):
            if is_cancelled(context):
                return self._cancelled_video_result(
                    frames=processed_frames,
                    stage=f"frame_{index}",
                    metadata={
                        "processed_frames": len(processed_frames),
                        "total_frames": len(frames),
                    },
                )

            mask = self._get_optional_item(masks, index)
            original_frame = self._get_optional_item(original_frames, index)

            image_result = self.run_image(
                image=frame,
                mask=mask,
                original_image=original_frame,
                context={
                    **context,
                    "stage": "postprocessing_frame",
                    "frame_index": index,
                },
            )

            if image_result.image is None:
                raise RuntimeError(
                    f"Postprocessing returned image=None for frame {index}"
                )

            processed_frames.append(image_result.image)
            frame_metadata.append(image_result.metadata)

            if image_result.cancelled:
                return self._cancelled_video_result(
                    frames=processed_frames,
                    stage=f"frame_{index}",
                    metadata={
                        "processed_frames": len(processed_frames),
                        "total_frames": len(frames),
                        "frame_metadata": frame_metadata,
                    },
                )

        if self.config.enable_temporal_smoothing:
            if is_cancelled(context):
                return self._cancelled_video_result(
                    frames=processed_frames,
                    stage="before_temporal_smoothing",
                    metadata={
                        "processed_frames": len(processed_frames),
                        "total_frames": len(frames),
                    },
                )

            processed_frames = self.temporal_smoothing(
                processed_frames,
                masks=masks,
                context={
                    **context,
                    "stage": "temporal_smoothing",
                },
            )

        return PostprocessingPipelineResult(
            frames=processed_frames,
            cancelled=False,
            metadata={
                "pipeline": self.name,
                "input_type": "video",
                "enabled": True,
                "cancelled": False,
                "input_frame_count": len(frames),
                "output_frame_count": len(processed_frames),
                "temporal_smoothing": self.config.enable_temporal_smoothing,
                "frame_metadata": frame_metadata,
            },
        )

    @staticmethod
    # Use shared helper pipelines.utils.get_optional_item
    def _get_optional_item(
        items: list[np.ndarray] | None,
        index: int,
    ) -> np.ndarray | None:
        return get_optional_item(items, index)

    @staticmethod
    # Cancellation checks delegated to pipelines.utils.is_cancelled

    def _cancelled_image_result(
        self,
        image: np.ndarray,
        stage: str,
    ) -> PostprocessingPipelineResult:
        return PostprocessingPipelineResult(
            image=image,
            cancelled=True,
            cancelled_at=stage,
            metadata={
                "pipeline": self.name,
                "input_type": "image",
                "cancelled": True,
                "cancelled_at": stage,
                "output_shape": image.shape,
            },
        )

    def _cancelled_video_result(
        self,
        frames: list[np.ndarray],
        stage: str,
        metadata: dict[str, Any] | None = None,
    ) -> PostprocessingPipelineResult:
        metadata = metadata or {}

        metadata.update(
            {
                "pipeline": self.name,
                "input_type": "video",
                "cancelled": True,
                "cancelled_at": stage,
                "output_frame_count": len(frames),
            }
        )

        return PostprocessingPipelineResult(
            frames=frames,
            cancelled=True,
            cancelled_at=stage,
            metadata=metadata,
        )


def run_postprocessing_image(
    image: np.ndarray,
    mask: np.ndarray | None = None,
    original_image: np.ndarray | None = None,
) -> PostprocessingPipelineResult:
    pipeline = PostprocessingPipeline()
    return pipeline.run_image(
        image=image,
        mask=mask,
        original_image=original_image,
    )


def run_postprocessing_video(
    frames: list[np.ndarray],
    masks: list[np.ndarray] | None = None,
    original_frames: list[np.ndarray] | None = None,
) -> PostprocessingPipelineResult:
    pipeline = PostprocessingPipeline()
    return pipeline.run_video(
        frames=frames,
        masks=masks,
        original_frames=original_frames,
    )