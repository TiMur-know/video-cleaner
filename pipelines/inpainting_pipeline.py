# pipelines/inpainting_pipeline.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from inpainters.opencv_inpainter import (
    OpenCVInpainter,
    OpenCVInpainterConfig,
)
from inpainters.lama_inpainter import (
    LaMaInpainter,
    LaMaInpainterConfig,
)
from inpainters.sdxl_inpainter import (
    SDXLInpainter,
    SDXLInpainterConfig,
)
from inpainters.stable_diffusion_inpainter import (
    StableDiffusionInpainter,
    StableDiffusionInpainterConfig,
)
from inpainters.flux_inpainter import (
    FluxInpainter,
    FluxInpainterConfig,
)
from pipelines.utils import is_cancelled


InpainterMode = Literal[
    "auto",
    "opencv",
    "lama",
    "stable_diffusion",
    "sdxl",
    "flux",
]


@dataclass(slots=True)
class InpaintingPipelineConfig:
    enabled: bool = True

    inpainter: InpainterMode = "auto"

    enable_opencv: bool = True
    enable_lama: bool = True
    enable_stable_diffusion: bool = False
    enable_sdxl: bool = False
    enable_flux: bool = False

    # If chosen inpainter fails, try the next available backend.
    fallback_enabled: bool = True

    opencv: OpenCVInpainterConfig = field(default_factory=OpenCVInpainterConfig)
    lama: LaMaInpainterConfig = field(default_factory=LaMaInpainterConfig)
    stable_diffusion: StableDiffusionInpainterConfig = field(
        default_factory=StableDiffusionInpainterConfig
    )
    sdxl: SDXLInpainterConfig = field(default_factory=SDXLInpainterConfig)
    flux: FluxInpainterConfig = field(default_factory=FluxInpainterConfig)


@dataclass(slots=True)
class InpaintingPipelineResult:
    image: np.ndarray | None = None
    frames: list[np.ndarray] | None = None
    mask: np.ndarray | None = None
    masks: list[np.ndarray] | None = None

    cancelled: bool = False
    cancelled_at: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)


class InpaintingPipeline:
    """
    Inpainting pipeline.

    Image flow:
        image + mask
            ↓
        selected inpainter
            ↓
        inpainted image

    Video flow:
        frames + masks
            ↓
        per-frame inpainting
            ↓
        inpainted frames

    Cancel support:
        Pass:

            context={
                "cancel_token": {"stop": False}
            }

        UI can set:

            cancel_token["stop"] = True

        This pipeline checks the flag between frames and before each backend call.
    """

    name = "inpainting_pipeline"

    def __init__(
        self,
        config: InpaintingPipelineConfig | None = None,
        lama_model: Any | None = None,
        stable_diffusion_pipe: Any | None = None,
        sdxl_pipe: Any | None = None,
        flux_pipe: Any | None = None,
    ) -> None:
        self.config = config or InpaintingPipelineConfig()

        self.opencv_inpainter = OpenCVInpainter(self.config.opencv)

        self.lama_inpainter = LaMaInpainter(
            self.config.lama,
            model=lama_model,
        )

        self.sdxl_inpainter = SDXLInpainter(
            self.config.sdxl,
            pipe=sdxl_pipe,
        )

        self.stable_diffusion_inpainter = StableDiffusionInpainter(
            self.config.stable_diffusion,
            pipe=stable_diffusion_pipe,
        )

        self.flux_inpainter = FluxInpainter(
            self.config.flux,
            pipe=flux_pipe,
        )

    def run_image(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        context: dict[str, Any] | None = None,
    ) -> InpaintingPipelineResult:
        context = context or {}

        if not self.config.enabled:
            return InpaintingPipelineResult(
                image=image,
                mask=mask,
                cancelled=False,
                metadata={
                    "pipeline": self.name,
                    "enabled": False,
                    "input_type": "image",
                    "selected_inpainter": None,
                    "used_inpainter": None,
                    "fallback_used": False,
                },
            )

        if is_cancelled(context):
            return self._cancelled_image_result(
                image=image,
                mask=mask,
                stage="before_inpainting",
            )

        if mask is None:
            raise ValueError("mask is required for inpainting.")

        if mask.shape[:2] != image.shape[:2]:
            raise ValueError(
                f"Mask shape {mask.shape[:2]} does not match image shape {image.shape[:2]}"
            )

        selected_backends = self._select_backends()
        errors: list[str] = []

        for backend_name in selected_backends:
            if is_cancelled(context):
                return self._cancelled_image_result(
                    image=image,
                    mask=mask,
                    stage=f"before_{backend_name}",
                    errors=errors,
                )

            try:
                result = self._run_backend(
                    backend_name=backend_name,
                    image=image,
                    mask=mask,
                    context={
                        **context,
                        "stage": "inpainting",
                        "inpainter": backend_name,
                    },
                )

                if is_cancelled(context):
                    return self._cancelled_image_result(
                        image=result.image,
                        mask=result.mask,
                        stage=f"after_{backend_name}",
                        errors=errors,
                    )

                return InpaintingPipelineResult(
                    image=result.image,
                    mask=result.mask,
                    cancelled=False,
                    metadata={
                        "pipeline": self.name,
                        "enabled": True,
                        "cancelled": False,
                        "input_type": "image",
                        "selected_inpainter": self.config.inpainter,
                        "used_inpainter": backend_name,
                        "fallback_used": backend_name != selected_backends[0],
                        "errors": errors,
                        "backend_metadata": getattr(result, "metadata", {}),
                        "input_shape": image.shape,
                        "output_shape": result.image.shape,
                        "mask_shape": mask.shape,
                        "mask_area": int(np.count_nonzero(mask)),
                    },
                )

            except Exception as exc:
                errors.append(f"{backend_name}: {type(exc).__name__}: {exc}")

                if not self.config.fallback_enabled:
                    raise

                continue

        raise RuntimeError(
            "All inpainting backends failed:\n" + "\n".join(errors)
        )

    def run_video(
        self,
        frames: list[np.ndarray],
        masks: list[np.ndarray],
        context: dict[str, Any] | None = None,
    ) -> InpaintingPipelineResult:
        context = context or {}

        if not self.config.enabled:
            return InpaintingPipelineResult(
                frames=frames,
                masks=masks,
                cancelled=False,
                metadata={
                    "pipeline": self.name,
                    "enabled": False,
                    "input_type": "video",
                    "frame_count": len(frames),
                    "mask_count": len(masks),
                    "selected_inpainter": None,
                    "used_inpainter": None,
                },
            )

        if is_cancelled(context):
            return self._cancelled_video_result(
                frames=[],
                masks=[],
                stage="before_inpainting",
            )

        if len(frames) != len(masks):
            min_len = min(len(frames), len(masks))
            frames = frames[:min_len]
            masks = masks[:min_len]

        output_frames: list[np.ndarray] = []
        output_masks: list[np.ndarray] = []
        frame_metadata: list[dict[str, Any]] = []

        for index, (frame, mask) in enumerate(zip(frames, masks)):
            if is_cancelled(context):
                return self._cancelled_video_result(
                    frames=output_frames,
                    masks=output_masks,
                    stage=f"frame_{index}",
                    metadata={
                        "processed_frames": len(output_frames),
                        "total_frames": len(frames),
                        "frame_metadata": frame_metadata,
                    },
                )

            image_result = self.run_image(
                image=frame,
                mask=mask,
                context={
                    **context,
                    "stage": "inpainting_frame",
                    "frame_index": index,
                },
            )

            if image_result.image is None:
                raise RuntimeError(
                    f"Inpainting returned image=None for frame {index}"
                )

            output_frames.append(image_result.image)
            output_masks.append(mask)
            frame_metadata.append(image_result.metadata)

            if image_result.cancelled:
                return self._cancelled_video_result(
                    frames=output_frames,
                    masks=output_masks,
                    stage=f"frame_{index}",
                    metadata={
                        "processed_frames": len(output_frames),
                        "total_frames": len(frames),
                        "frame_metadata": frame_metadata,
                    },
                )

        return InpaintingPipelineResult(
            frames=output_frames,
            masks=output_masks,
            cancelled=False,
            metadata={
                "pipeline": self.name,
                "enabled": True,
                "cancelled": False,
                "input_type": "video",
                "input_frame_count": len(frames),
                "output_frame_count": len(output_frames),
                "selected_inpainter": self.config.inpainter,
                "frame_metadata": frame_metadata,
            },
        )

    def _select_backends(self) -> list[str]:
        """
        Select backend order.

        auto:
            tries best available order:
                lama → opencv → sdxl → flux

        manual:
            tries selected backend first.
            If fallback_enabled=True, tries remaining enabled backends after it.
        """

        enabled: list[str] = []

        if self.config.enable_lama:
            enabled.append("lama")

        if self.config.enable_opencv:
            enabled.append("opencv")

        if self.config.enable_stable_diffusion:
            enabled.append("stable_diffusion")

        if self.config.enable_sdxl:
            enabled.append("sdxl")

        if self.config.enable_flux:
            enabled.append("flux")

        if self.config.inpainter == "auto":
            return enabled

        selected = self.config.inpainter

        if selected not in {"opencv", "lama", "stable_diffusion", "sdxl", "flux"}:
            raise ValueError(f"Unsupported inpainter: {selected}")

        if selected not in enabled:
            enabled.insert(0, selected)

        ordered = [selected]

        if self.config.fallback_enabled:
            ordered.extend(
                backend
                for backend in enabled
                if backend != selected
            )

        return ordered

    def _run_backend(
        self,
        backend_name: str,
        image: np.ndarray,
        mask: np.ndarray,
        context: dict[str, Any],
    ) -> Any:
        if backend_name == "opencv":
            return self.opencv_inpainter.inpaint(
                image=image,
                mask=mask,
                context=context,
            )

        if backend_name == "lama":
            return self.lama_inpainter.inpaint(
                image=image,
                mask=mask,
                context=context,
            )

        if backend_name == "sdxl":
            return self.sdxl_inpainter.inpaint(
                image=image,
                mask=mask,
                context=context,
            )

        if backend_name == "stable_diffusion":
            return self.stable_diffusion_inpainter.inpaint(
                image=image,
                mask=mask,
                context=context,
            )

        if backend_name == "flux":
            return self.flux_inpainter.inpaint(
                image=image,
                mask=mask,
                context=context,
            )

        raise ValueError(f"Unsupported backend: {backend_name}")

    # Cancellation checks delegated to pipelines.utils.is_cancelled

    def _cancelled_image_result(
        self,
        image: np.ndarray | None,
        mask: np.ndarray | None,
        stage: str,
        errors: list[str] | None = None,
    ) -> InpaintingPipelineResult:
        return InpaintingPipelineResult(
            image=image,
            mask=mask,
            cancelled=True,
            cancelled_at=stage,
            metadata={
                "pipeline": self.name,
                "input_type": "image",
                "cancelled": True,
                "cancelled_at": stage,
                "errors": errors or [],
                "output_shape": image.shape if image is not None else None,
                "mask_shape": mask.shape if mask is not None else None,
            },
        )

    def _cancelled_video_result(
        self,
        frames: list[np.ndarray],
        masks: list[np.ndarray],
        stage: str,
        metadata: dict[str, Any] | None = None,
    ) -> InpaintingPipelineResult:
        metadata = metadata or {}

        metadata.update(
            {
                "pipeline": self.name,
                "input_type": "video",
                "cancelled": True,
                "cancelled_at": stage,
                "output_frame_count": len(frames),
                "mask_count": len(masks),
            }
        )

        return InpaintingPipelineResult(
            frames=frames,
            masks=masks,
            cancelled=True,
            cancelled_at=stage,
            metadata=metadata,
        )


def run_inpainting_image(
    image: np.ndarray,
    mask: np.ndarray,
) -> InpaintingPipelineResult:
    pipeline = InpaintingPipeline()
    return pipeline.run_image(
        image=image,
        mask=mask,
    )


def run_inpainting_video(
    frames: list[np.ndarray],
    masks: list[np.ndarray],
) -> InpaintingPipelineResult:
    pipeline = InpaintingPipeline()
    return pipeline.run_video(
        frames=frames,
        masks=masks,
    )
