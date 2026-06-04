# pipelines/video_pipeline.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from utils.frame_extract import (
    FrameExtractionConfig,
    FrameExtractionResult,
    FrameExtractor,
)
from utils.video_io_utils import ensure_dir, read_frame, save_frame
from utils.video_rebuilder import (
    VideoRebuildConfig,
    VideoRebuildResult,
    VideoRebuilder,
)

from pipelines.preprocessing_pipeline import (
    PreprocessingPipeline,
    PreprocessingPipelineConfig,
)
from pipelines.detection_pipeline import (
    DetectionPipeline,
    DetectionPipelineConfig,
)
from pipelines.inpainting_pipeline import (
    InpaintingPipeline,
    InpaintingPipelineConfig,
)
from pipelines.postprocessing_pipeline import (
    PostprocessingPipeline,
    PostprocessingPipelineConfig,
)

from detectors.utils import resize_mask_nearest, to_mask_uint8

from trackers.optical_flow_tracker import (
    OpticalFlowTracker,
    OpticalFlowTrackerConfig,
)
from trackers.kalman_tracker import (
    KalmanTracker,
    KalmanTrackerConfig,
)
from trackers.xmem_tracker import (
    XMemTracker,
    XMemTrackerConfig,
)
from trackers.cotracker import (
    CoTracker,
    CoTrackerConfig,
)
from pipelines.mask_processing_pipeline import (
    MaskProcessingPipeline,
    MaskProcessingPipelineConfig,
)
from pipelines.utils import is_cancelled


TrackerMode = Literal[
    "none",
    "optical_flow",
    "kalman",
    "xmem",
    "cotracker",
]


@dataclass(slots=True)
class VideoTrackingConfig:
    enabled: bool = True

    mode: TrackerMode = "optical_flow"

    # Run full detection every N frames.
    # Between detection frames, tracker predicts mask.
    detection_interval: int = 5

    optical_flow: OpticalFlowTrackerConfig = field(
        default_factory=OpticalFlowTrackerConfig
    )
    kalman: KalmanTrackerConfig = field(default_factory=KalmanTrackerConfig)
    xmem: XMemTrackerConfig = field(default_factory=XMemTrackerConfig)
    cotracker: CoTrackerConfig = field(default_factory=CoTrackerConfig)


@dataclass(slots=True)
class VideoPipelineConfig:
    input_path: str = "data/inputs/input.mp4"
    output_path: str = "data/outputs/output.mp4"

    frame_dir: str = "data/temp/frames"
    processed_frame_dir: str = "data/temp/processed_frames"

    extraction: FrameExtractionConfig = field(default_factory=FrameExtractionConfig)
    preprocessing: PreprocessingPipelineConfig = field(
        default_factory=PreprocessingPipelineConfig
    )
    detection: DetectionPipelineConfig = field(default_factory=DetectionPipelineConfig)
    mask_processing: MaskProcessingPipelineConfig = field(
        default_factory=MaskProcessingPipelineConfig
    )
    tracking: VideoTrackingConfig = field(default_factory=VideoTrackingConfig)
    inpainting: InpaintingPipelineConfig = field(default_factory=InpaintingPipelineConfig)
    postprocessing: PostprocessingPipelineConfig = field(
        default_factory=PostprocessingPipelineConfig
    )
    rebuild: VideoRebuildConfig = field(default_factory=VideoRebuildConfig)

    save_processed_frames: bool = True
    processed_frame_extension: str = "png"

    # Good for first implementation. Later you can stream to reduce RAM use.
    keep_frames_in_memory: bool = True


@dataclass(slots=True)
class VideoPipelineResult:
    output_path: str
    mask_paths: list[str] = field(default_factory=list)
    processed_frame_paths: list[str] = field(default_factory=list)

    extraction_result: FrameExtractionResult | None = None
    rebuild_result: VideoRebuildResult | None = None

    cancelled: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class VideoPipeline:
    """
    Full video watermark-removal pipeline.

    Flow:
        input video
            ↓
        frame extraction
            ↓
        preprocessing per frame
            ↓
        detection per frame
            ↓
        mask_processing per frame
            ↓
        tracking + inpainting per frame
            ↓
        postprocessing per frame
            ↓
        video rebuilding

    Cancel support:
        Pass this in context:

            context={
                "cancel_token": {"stop": False}
            }

        UI can set:

            cancel_token["stop"] = True

        This pipeline checks that flag between long stages and loops.
    """

    name = "video_pipeline"

    def __init__(
        self,
        config: VideoPipelineConfig | None = None,
        yolo_model: Any | None = None,
        sam2_model: Any | None = None,
        easy_ocr_reader: Any | None = None,
        paddle_ocr_model: Any | None = None,
        lama_model: Any | None = None,
        stable_diffusion_pipe: Any | None = None,
        sdxl_pipe: Any | None = None,
        flux_pipe: Any | None = None,
        xmem_model: Any | None = None,
        cotracker_model: Any | None = None,
    ) -> None:
        self.config = config or VideoPipelineConfig()

        self.config.extraction.output_dir = self.config.frame_dir
        self.config.rebuild.output_path = self.config.output_path

        self.extractor = FrameExtractor(self.config.extraction)

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

        self.rebuilder = VideoRebuilder(self.config.rebuild)

        self.optical_flow_tracker = OpticalFlowTracker(
            self.config.tracking.optical_flow
        )
        self.kalman_tracker = KalmanTracker(
            self.config.tracking.kalman
        )
        self.xmem_tracker = XMemTracker(
            self.config.tracking.xmem,
            model=xmem_model,
        )
        self.cotracker = CoTracker(
            self.config.tracking.cotracker,
            model=cotracker_model,
        )

    def run(
        self,
        input_path: str | None = None,
        output_path: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> VideoPipelineResult:
        context = context or {}

        input_path = input_path or self.config.input_path
        output_path = output_path or self.config.output_path

        self.config.rebuild.output_path = output_path
        self.config.rebuild.source_video_path = input_path

        pipeline_context = {
            **context,
            "pipeline": self.name,
            "input_type": "video",
            "input_path": input_path,
            "output_path": output_path,
        }

        if is_cancelled(pipeline_context):
            return self._cancelled_result(
                output_path=output_path,
                stage="before_frame_extraction",
                metadata={
                    "input_path": input_path,
                    "output_path": output_path,
                },
            )

        extraction = self.extractor.extract(
            input_path,
            context={
                **pipeline_context,
                "stage": "frame_extraction",
            },
        )

        if is_cancelled(pipeline_context):
            return self._cancelled_result(
                output_path=output_path,
                stage="after_frame_extraction",
                extraction=extraction,
                metadata={
                    "input_path": input_path,
                    "output_path": output_path,
                    "extraction": extraction.metadata,
                },
            )

        original_frames = self._load_extracted_frames(
            extraction=extraction,
            context=pipeline_context,
        )

        if not original_frames:
            raise RuntimeError("No frames were extracted from the video.")

        pipeline_context = {
            **pipeline_context,
            "fps": extraction.fps,
            "metadata": extraction.metadata,
        }

        if is_cancelled(pipeline_context):
            return self._cancelled_result(
                output_path=output_path,
                stage="after_frame_loading",
                extraction=extraction,
                metadata={
                    "input_path": input_path,
                    "output_path": output_path,
                    "extraction": extraction.metadata,
                },
            )

        masks = self._build_masks(
            frames=original_frames,
            context=pipeline_context,
        )

        if is_cancelled(pipeline_context):
            return self._cancelled_result(
                output_path=output_path,
                stage="after_mask_building",
                extraction=extraction,
                metadata={
                    "input_path": input_path,
                    "output_path": output_path,
                    "fps": extraction.fps,
                    "width": extraction.width,
                    "height": extraction.height,
                    "total_frames": extraction.total_frames,
                    "processed_masks": len(masks),
                    "extraction": extraction.metadata,
                },
            )

        if len(masks) < len(original_frames):
            original_frames = original_frames[: len(masks)]

        inpainting_result = self.inpainting_pipeline.run_video(
            frames=original_frames,
            masks=masks,
            context={
                **pipeline_context,
                "stage": "inpainting",
            },
        )

        if is_cancelled(pipeline_context):
            return self._cancelled_result(
                output_path=output_path,
                stage="after_inpainting",
                extraction=extraction,
                metadata={
                    "input_path": input_path,
                    "output_path": output_path,
                    "fps": extraction.fps,
                    "width": extraction.width,
                    "height": extraction.height,
                    "total_frames": extraction.total_frames,
                    "processed_masks": len(masks),
                    "inpainting": inpainting_result.metadata,
                    "extraction": extraction.metadata,
                },
            )

        if inpainting_result.frames is None:
            raise RuntimeError("Inpainting pipeline returned frames=None")

        inpainted_frames = inpainting_result.frames

        if len(inpainted_frames) < len(masks):
            masks = masks[: len(inpainted_frames)]
            original_frames = original_frames[: len(inpainted_frames)]

        postprocessing_result = self.postprocessing_pipeline.run_video(
            frames=inpainted_frames,
            masks=masks,
            original_frames=original_frames,
            context={
                **pipeline_context,
                "stage": "postprocessing",
            },
        )

        if is_cancelled(pipeline_context):
            return self._cancelled_result(
                output_path=output_path,
                stage="after_postprocessing",
                extraction=extraction,
                metadata={
                    "input_path": input_path,
                    "output_path": output_path,
                    "fps": extraction.fps,
                    "width": extraction.width,
                    "height": extraction.height,
                    "total_frames": extraction.total_frames,
                    "processed_masks": len(masks),
                    "inpainting": inpainting_result.metadata,
                    "postprocessing": postprocessing_result.metadata,
                    "extraction": extraction.metadata,
                },
            )

        if postprocessing_result.frames is None:
            raise RuntimeError("Postprocessing pipeline returned frames=None")

        final_frames = postprocessing_result.frames

        processed_frame_paths = self._save_processed_frames(
            frames=final_frames,
            context=pipeline_context,
        )

        if is_cancelled(pipeline_context):
            return self._cancelled_result(
                output_path=output_path,
                stage="after_saving_processed_frames",
                extraction=extraction,
                processed_frame_paths=processed_frame_paths,
                metadata={
                    "input_path": input_path,
                    "output_path": output_path,
                    "fps": extraction.fps,
                    "width": extraction.width,
                    "height": extraction.height,
                    "total_frames": extraction.total_frames,
                    "processed_frames": len(final_frames),
                    "processed_frame_paths": len(processed_frame_paths),
                    "extraction": extraction.metadata,
                    "inpainting": inpainting_result.metadata,
                    "postprocessing": postprocessing_result.metadata,
                },
            )

        self.config.rebuild.fps = extraction.fps
        self.config.rebuild.width = extraction.width
        self.config.rebuild.height = extraction.height

        rebuild_result = self.rebuilder.rebuild(
            processed_frame_paths,
            context={
                **pipeline_context,
                "stage": "video_rebuild",
                "fps": extraction.fps,
            },
        )

        metadata = {
            "pipeline": self.name,
            "cancelled": False,
            "input_path": input_path,
            "output_path": rebuild_result.output_path,
            "fps": extraction.fps,
            "width": extraction.width,
            "height": extraction.height,
            "total_frames": extraction.total_frames,
            "processed_frames": len(final_frames),
            "tracking_mode": self.config.tracking.mode,
            "mask_processing_enabled": self.config.mask_processing.enabled,
            "mask_area_total": int(sum(np.count_nonzero(mask) for mask in masks)),
            "extraction": extraction.metadata,
            "inpainting": inpainting_result.metadata,
            "postprocessing": postprocessing_result.metadata,
            "rebuild": rebuild_result.metadata,
        }

        return VideoPipelineResult(
            output_path=rebuild_result.output_path,
            processed_frame_paths=processed_frame_paths,
            extraction_result=extraction,
            rebuild_result=rebuild_result,
            cancelled=False,
            metadata=metadata,
        )

    def _load_extracted_frames(
        self,
        extraction: FrameExtractionResult,
        context: dict[str, Any],
    ) -> list[np.ndarray]:
        frames: list[np.ndarray] = []

        for item in extraction.frames:
            if is_cancelled(context):
                break

            if item.image is not None:
                frames.append(item.image)
                continue

            if item.path is None:
                raise RuntimeError(
                    f"Extracted frame {item.index} has neither image nor path."
                )

            frames.append(read_frame(item.path))

        return frames

    def _build_masks(
        self,
        frames: list[np.ndarray],
        context: dict[str, Any],
    ) -> list[np.ndarray]:
        if is_cancelled(context):
            return []

        if not self.config.tracking.enabled:
            return self._detect_all_masks(frames, context)

        mode = self.config.tracking.mode

        if mode == "none":
            return self._detect_all_masks(frames, context)

        if mode in ("xmem", "cotracker"):
            return self._track_sequence_with_model(frames, context)

        return self._detect_and_track_stream(frames, context)

    def _detect_all_masks(
        self,
        frames: list[np.ndarray],
        context: dict[str, Any],
    ) -> list[np.ndarray]:
        masks: list[np.ndarray] = []

        for index, frame in enumerate(frames):
            if is_cancelled(context):
                break

            mask = self._detect_single_mask(
                frame=frame,
                frame_index=index,
                context=context,
            )
            masks.append(mask)

        return masks

    def _detect_and_track_stream(
        self,
        frames: list[np.ndarray],
        context: dict[str, Any],
    ) -> list[np.ndarray]:
        masks: list[np.ndarray] = []

        previous_frame: np.ndarray | None = None
        previous_mask: np.ndarray | None = None

        interval = max(1, self.config.tracking.detection_interval)

        for index, frame in enumerate(frames):
            if is_cancelled(context):
                break

            should_detect = (
                index == 0
                or previous_mask is None
                or index % interval == 0
            )

            if should_detect:
                mask = self._detect_single_mask(
                    frame=frame,
                    frame_index=index,
                    context=context,
                )

                if self.config.tracking.mode == "kalman":
                    mask = self.kalman_tracker.track(
                        detected_mask=mask,
                        output_shape=frame.shape[:2],
                    )

            else:
                mask = self._track_single_mask(
                    previous_frame=previous_frame,
                    current_frame=frame,
                    previous_mask=previous_mask,
                )

            masks.append(mask)

            previous_frame = frame
            previous_mask = mask

        return masks

    def _track_single_mask(
        self,
        previous_frame: np.ndarray | None,
        current_frame: np.ndarray,
        previous_mask: np.ndarray | None,
    ) -> np.ndarray:
        if previous_mask is None:
            return np.zeros(current_frame.shape[:2], dtype=np.uint8)

        mode = self.config.tracking.mode

        if mode == "optical_flow":
            if previous_frame is None:
                return previous_mask

            return self.optical_flow_tracker.track(
                previous_frame=previous_frame,
                current_frame=current_frame,
                previous_mask=previous_mask,
            )

        if mode == "kalman":
            return self.kalman_tracker.track(
                detected_mask=None,
                output_shape=current_frame.shape[:2],
            )

        return previous_mask

    def _track_sequence_with_model(
        self,
        frames: list[np.ndarray],
        context: dict[str, Any],
    ) -> list[np.ndarray]:
        """
        Sequence trackers need an initial mask.

        For XMem and CoTracker:
            detect first frame
            then track the mask through the whole sequence

        Note:
            Full interruption inside model tracker depends on the tracker implementation.
            This checks cancel before and after the model call.
        """

        if not frames:
            return []

        if is_cancelled(context):
            return []

        initial_mask = self._detect_single_mask(
            frame=frames[0],
            frame_index=0,
            context=context,
        )

        if is_cancelled(context):
            return [initial_mask]

        mode = self.config.tracking.mode

        if mode == "xmem":
            masks = self.xmem_tracker.track_sequence(
                frames=frames,
                initial_mask=initial_mask,
            )
            return masks if not is_cancelled(context) else masks[:1]

        if mode == "cotracker":
            masks = self.cotracker.track_sequence(
                frames=frames,
                initial_mask=initial_mask,
            )
            return masks if not is_cancelled(context) else masks[:1]

        raise ValueError(f"Unsupported sequence tracker mode: {mode}")

    def _detect_single_mask(
        self,
        frame: np.ndarray,
        frame_index: int,
        context: dict[str, Any],
    ) -> np.ndarray:
        if is_cancelled(context):
            return np.zeros(frame.shape[:2], dtype=np.uint8)

        preprocessing_result = self.preprocessing_pipeline.run(
            frame,
            context={
                **context,
                "stage": "preprocessing",
                "frame_index": frame_index,
            },
        )

        if is_cancelled(context):
            return np.zeros(frame.shape[:2], dtype=np.uint8)

        detection_result = self.detection_pipeline.run(
            preprocessing_result.main_image,
            context={
                **context,
                "stage": "detection",
                "frame_index": frame_index,
            },
        )

        if is_cancelled(context):
            return np.zeros(frame.shape[:2], dtype=np.uint8)

        mask_processing_result = self.mask_processing_pipeline.run_image(
            image=preprocessing_result.main_image,
            rough_mask=detection_result.mask,
            boxes=getattr(detection_result, "boxes", None),
            detection_results=[detection_result],
            support_masks=[],
            context={
                **context,
                "stage": "mask_processing",
                "frame_index": frame_index,
            },
        )

        return self._ensure_mask_shape(
            mask=mask_processing_result.mask,
            image_shape=frame.shape[:2],
        )

    def _ensure_mask_shape(
        self,
        mask: np.ndarray,
        image_shape: tuple[int, int],
    ) -> np.ndarray:
        mask_uint8 = to_mask_uint8(mask)

        if mask_uint8.shape[:2] == image_shape:
            return mask_uint8

        return resize_mask_nearest(
            mask_uint8,
            image_shape=image_shape,
        )

    def _save_processed_frames(
        self,
        frames: list[np.ndarray],
        context: dict[str, Any],
    ) -> list[str]:
        if not self.config.save_processed_frames:
            return frames  # type: ignore[return-value]

        output_dir = ensure_dir(self.config.processed_frame_dir)

        extension = self.config.processed_frame_extension.lower().lstrip(".")

        paths: list[str] = []

        for index, frame in enumerate(frames):
            if is_cancelled(context):
                break

            path = output_dir / f"frame_{index:08d}.{extension}"
            saved_path = save_frame(frame, path)
            paths.append(saved_path)

        return paths

    # Cancellation checks delegated to pipelines.utils.is_cancelled

    def _cancelled_result(
        self,
        output_path: str,
        stage: str,
        extraction: FrameExtractionResult | None = None,
        rebuild: VideoRebuildResult | None = None,
        processed_frame_paths: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> VideoPipelineResult:
        metadata = metadata or {}

        metadata.update(
            {
                "pipeline": self.name,
                "cancelled": True,
                "cancelled_at": stage,
                "output_path": output_path,
            }
        )

        return VideoPipelineResult(
            output_path=output_path,
            processed_frame_paths=processed_frame_paths or [],
            extraction_result=extraction,
            rebuild_result=rebuild,
            cancelled=True,
            metadata=metadata,
        )


def run_video_pipeline(
    input_path: str,
    output_path: str = "data/outputs/output.mp4",
) -> VideoPipelineResult:
    pipeline = VideoPipeline(
        VideoPipelineConfig(
            input_path=input_path,
            output_path=output_path,
        )
    )

    return pipeline.run()
