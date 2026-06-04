# inpainters/stable_diffusion_inpainter.py

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
class StableDiffusionInpaintResult:
    image: np.ndarray
    mask: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StableDiffusionInpainterConfig:
    enabled: bool = True

    model_id: str = "runwayml/stable-diffusion-inpainting"
    device: str = "auto"

    input_color_order: ColorOrder = "bgr"
    output_color_order: ColorOrder = "bgr"

    prompt: str = (
        "clean natural image, realistic background, seamless texture, "
        "no watermark, no text, no logo"
    )
    negative_prompt: str = (
        "watermark, text, logo, blurry, distorted, artifacts, low quality"
    )

    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    strength: float = 0.85

    seed: int | None = 42

    # Stable Diffusion inpainting expects dimensions divisible by 8.
    resize_to_multiple_of: int = 8

    dilate_mask_iterations: int = 1
    mask_kernel_size: int = 5

    enable_model_cpu_offload: bool = False
    enable_attention_slicing: bool = True

    torch_dtype: str = "float16"


class StableDiffusionInpainter:
    """
    Stable Diffusion inpainting adapter.

    Compared with SDXL this backend is usually lighter and faster, but can be
    less detailed on large or complex repairs.

    Install:
        pip install diffusers transformers accelerate torch pillow
    """

    name = "stable_diffusion"

    def __init__(
        self,
        config: StableDiffusionInpainterConfig | None = None,
        pipe: Any | None = None,
    ) -> None:
        self.config = config or StableDiffusionInpainterConfig()
        self.pipe = pipe

    def __call__(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        context: dict[str, Any] | None = None,
    ) -> StableDiffusionInpaintResult:
        return self.inpaint(image=image, mask=mask, context=context)

    def inpaint(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        context: dict[str, Any] | None = None,
    ) -> StableDiffusionInpaintResult:
        mask_uint8 = to_mask_uint8(mask)

        if not self.config.enabled:
            return StableDiffusionInpaintResult(
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

        result = self.pipe(
            prompt=self.config.prompt,
            negative_prompt=self.config.negative_prompt,
            image=pil_image,
            mask_image=pil_mask,
            num_inference_steps=self.config.num_inference_steps,
            guidance_scale=self.config.guidance_scale,
            strength=self.config.strength,
            generator=generator,
        )

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
            "strength": self.config.strength,
            "mask_area": int(np.count_nonzero(mask_uint8)),
        }

        return StableDiffusionInpaintResult(
            image=output,
            mask=mask_uint8,
            metadata=metadata,
        )

    def _load_pipeline(self) -> Any:
        torch = lazy_import("torch")
        diffusers = lazy_import("diffusers")

        pipeline_cls = getattr(diffusers, "StableDiffusionInpaintPipeline")
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


def stable_diffusion_inpaint(
    image: np.ndarray,
    mask: np.ndarray,
    prompt: str | None = None,
    device: str = "auto",
) -> np.ndarray:
    config = StableDiffusionInpainterConfig(device=device)

    if prompt is not None:
        config.prompt = prompt

    inpainter = StableDiffusionInpainter(config)
    return inpainter.inpaint(image, mask).image
