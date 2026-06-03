# inpainters/flux_inpainter.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from inpainters.utils import (
    image_to_pil_rgb,
    lazy_import,
    make_generator,
    mask_to_pil,
    pil_to_numpy,
    resize_pil_to_multiple,
    resize_pil_to_size,
    resolve_device,
    resolve_torch_dtype,
    to_mask_uint8,
    validate_image,
    validate_mask,
)


ColorOrder = Literal["bgr", "rgb"]


@dataclass(slots=True)
class FluxInpaintResult:
    image: np.ndarray
    mask: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FluxInpainterConfig:
    enabled: bool = True

    model_id: str = "black-forest-labs/FLUX.1-Fill-dev"
    device: str = "auto"

    input_color_order: ColorOrder = "bgr"
    output_color_order: ColorOrder = "bgr"

    prompt: str = (
        "clean natural background, realistic texture, seamless repair, "
        "no watermark, no text, no logo"
    )
    negative_prompt: str = (
        "watermark, text, logo, artifacts, distortion, blurry, low quality"
    )

    num_inference_steps: int = 40
    guidance_scale: float = 30.0
    seed: int | None = 42

    # FLUX Fill works with dimensions divisible by 8.
    resize_to_multiple_of: int = 8

    dilate_mask_iterations: int = 1
    mask_kernel_size: int = 5

    enable_model_cpu_offload: bool = False
    enable_attention_slicing: bool = True

    torch_dtype: str = "bfloat16"


class FluxInpainter:
    """
    FLUX inpainting adapter using Diffusers.

    Best for:
        - high-quality watermark removal
        - larger or complex regions
        - natural background reconstruction

    Install:
        pip install diffusers transformers accelerate torch pillow

    Notes:
        - FLUX Fill does not use `strength` like SDXL inpainting.
        - You can pass a preloaded pipe with FluxInpainter(pipe=...).
    """

    name = "flux"

    def __init__(
        self,
        config: FluxInpainterConfig | None = None,
        pipe: Any | None = None,
    ) -> None:
        self.config = config or FluxInpainterConfig()
        self.pipe = pipe

    def __call__(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        context: dict[str, Any] | None = None,
    ) -> FluxInpaintResult:
        return self.inpaint(image=image, mask=mask, context=context)

    def inpaint(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        context: dict[str, Any] | None = None,
    ) -> FluxInpaintResult:
        mask_uint8 = to_mask_uint8(mask)

        if not self.config.enabled:
            return FluxInpaintResult(
                image=image,
                mask=mask_uint8,
                metadata={
                    "enabled": False,
                    "inpainter": self.name,
                },
            )

        validate_image(image, self.name)
        validate_mask(mask, image.shape[:2], self.name)

        if self.pipe is None:
            self.pipe = self._load_pipeline()

        pil_image = image_to_pil_rgb(
            image,
            input_color_order=self.config.input_color_order,
        )

        pil_mask = mask_to_pil(
            mask,
            dilate_iterations=self.config.dilate_mask_iterations,
            kernel_size=self.config.mask_kernel_size,
        )

        original_size = pil_image.size

        pil_image = resize_pil_to_multiple(
            pil_image,
            multiple=self.config.resize_to_multiple_of,
        )
        pil_mask = resize_pil_to_size(pil_mask, pil_image.size)

        generator = make_generator(
            seed=self.config.seed,
            device=self.config.device,
        )

        kwargs: dict[str, Any] = {
            "prompt": self.config.prompt,
            "image": pil_image,
            "mask_image": pil_mask,
            "height": pil_image.height,
            "width": pil_image.width,
            "num_inference_steps": self.config.num_inference_steps,
            "guidance_scale": self.config.guidance_scale,
            "generator": generator,
        }

        if self.config.negative_prompt:
            kwargs["negative_prompt"] = self.config.negative_prompt

        result = self.pipe(**kwargs)

        output_pil = result.images[0]

        if output_pil.size != original_size:
            output_pil = resize_pil_to_size(output_pil, original_size)

        output = pil_to_numpy(
            output_pil,
            output_color_order=self.config.output_color_order,
        )

        metadata = {
            "inpainter": self.name,
            "model_id": self.config.model_id,
            "device": resolve_device(self.config.device),
            "num_inference_steps": self.config.num_inference_steps,
            "guidance_scale": self.config.guidance_scale,
            "mask_area": int(np.count_nonzero(mask_uint8)),
        }

        return FluxInpaintResult(
            image=output,
            mask=mask_uint8,
            metadata=metadata,
        )

    def _load_pipeline(self) -> Any:
        torch = lazy_import("torch")
        diffusers = lazy_import("diffusers")

        pipeline_cls = self._get_flux_pipeline_class(diffusers)
        dtype = resolve_torch_dtype(torch, self.config.torch_dtype)

        pipe = pipeline_cls.from_pretrained(
            self.config.model_id,
            torch_dtype=dtype,
        )

        if (
            self.config.enable_attention_slicing
            and hasattr(pipe, "enable_attention_slicing")
        ):
            pipe.enable_attention_slicing()

        if (
            self.config.enable_model_cpu_offload
            and hasattr(pipe, "enable_model_cpu_offload")
        ):
            pipe.enable_model_cpu_offload()
            return pipe

        return pipe.to(resolve_device(self.config.device))

    @staticmethod
    def _get_flux_pipeline_class(diffusers: Any) -> Any:
        """
        Diffusers may expose different FLUX inpainting class names depending
        on version.
        """
        if hasattr(diffusers, "FluxFillPipeline"):
            return diffusers.FluxFillPipeline

        if hasattr(diffusers, "FluxInpaintPipeline"):
            return diffusers.FluxInpaintPipeline

        raise ImportError(
            "Could not find FluxFillPipeline or FluxInpaintPipeline in diffusers. "
            "Update diffusers or pass a preloaded pipe to FluxInpainter(pipe=...)."
        )


def flux_inpaint(
    image: np.ndarray,
    mask: np.ndarray,
    prompt: str | None = None,
    device: str = "auto",
) -> np.ndarray:
    config = FluxInpainterConfig(device=device)

    if prompt is not None:
        config.prompt = prompt

    inpainter = FluxInpainter(config)
    return inpainter.inpaint(image, mask).image