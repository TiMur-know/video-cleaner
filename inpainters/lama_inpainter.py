# inpainters/lama_inpainter.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import numpy as np

from inpainters.utils import (
    image_to_pil_rgb,
    lazy_import,
    mask_to_pil,
    model_output_to_numpy,
    resize_numpy_to_shape,
    resolve_device,
    to_mask_uint8,
    validate_image,
    validate_mask,
)


ColorOrder = Literal["bgr", "rgb"]


@dataclass(slots=True)
class LaMaInpaintResult:
    image: np.ndarray
    mask: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LaMaInpainterConfig:
    enabled: bool = True

    device: str = "auto"

    input_color_order: ColorOrder = "bgr"
    output_color_order: ColorOrder = "bgr"

    # Some wrappers need a checkpoint/model path, some download automatically.
    model_path: str | None = None

    dilate_mask_iterations: int = 1
    mask_kernel_size: int = 5

    # Inject your preferred LaMa loader here.
    model_factory: Callable[["LaMaInpainterConfig"], Any] | None = None


class LaMaInpainter:
    """
    LaMa inpainting adapter.

    Best for:
        - fast object/watermark removal
        - larger masks than OpenCV can handle
        - natural non-generative background completion

    Supported model styles:
        model.inpaint(image, mask)
        model.predict(image, mask)
        model(image, mask)
    """

    name = "lama"

    def __init__(
        self,
        config: LaMaInpainterConfig | None = None,
        model: Any | None = None,
    ) -> None:
        self.config = config or LaMaInpainterConfig()
        self.model = model

    def __call__(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        context: dict[str, Any] | None = None,
    ) -> LaMaInpaintResult:
        return self.inpaint(image=image, mask=mask, context=context)

    def inpaint(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        context: dict[str, Any] | None = None,
    ) -> LaMaInpaintResult:
        mask_uint8 = to_mask_uint8(mask)

        if not self.config.enabled:
            return LaMaInpaintResult(
                image=image,
                mask=mask_uint8,
                metadata={
                    "enabled": False,
                    "inpainter": self.name,
                },
            )

        validate_image(image, self.name)
        validate_mask(mask, image.shape[:2], self.name)

        if self.model is None:
            self.model = self._load_model()

        pil_image = image_to_pil_rgb(
            image,
            input_color_order=self.config.input_color_order,
        )

        pil_mask = mask_to_pil(
            mask,
            dilate_iterations=self.config.dilate_mask_iterations,
            kernel_size=self.config.mask_kernel_size,
        )

        raw_output = self._run_model(pil_image, pil_mask)

        output = model_output_to_numpy(
            raw_output,
            output_color_order=self.config.output_color_order,
        )

        if output.shape[:2] != image.shape[:2]:
            output = resize_numpy_to_shape(
                output,
                image_shape=image.shape[:2],
            )

        metadata = {
            "inpainter": self.name,
            "device": resolve_device(self.config.device),
            "model_path": self.config.model_path,
            "mask_area": int(np.count_nonzero(mask_uint8)),
        }

        return LaMaInpaintResult(
            image=output,
            mask=mask_uint8,
            metadata=metadata,
        )

    def _load_model(self) -> Any:
        if self.config.model_factory is not None:
            return self.config.model_factory(self.config)

        # Common simple LaMa wrapper.
        try:
            module = lazy_import("simple_lama_inpainting")
            if hasattr(module, "SimpleLama"):
                return module.SimpleLama()
        except ImportError:
            pass

        # Alternative wrapper/module name.
        try:
            module = lazy_import("simple_lama")
            if hasattr(module, "SimpleLama"):
                return module.SimpleLama()
        except ImportError:
            pass

        raise ImportError(
            "No LaMa backend found. Install a LaMa wrapper or pass a model. "
            "Example: LaMaInpainter(model=your_lama_model) or "
            "LaMaInpainterConfig(model_factory=...)."
        )

    def _run_model(self, pil_image: Any, pil_mask: Any) -> Any:
        if hasattr(self.model, "inpaint"):
            return self.model.inpaint(pil_image, pil_mask)

        if hasattr(self.model, "predict"):
            return self.model.predict(pil_image, pil_mask)

        if callable(self.model):
            return self.model(pil_image, pil_mask)

        raise TypeError(
            "Unsupported LaMa model interface. Expected inpaint(image, mask), "
            "predict(image, mask), or callable model(image, mask)."
        )


def lama_inpaint(
    image: np.ndarray,
    mask: np.ndarray,
    model: Any | None = None,
) -> np.ndarray:
    inpainter = LaMaInpainter(model=model)
    return inpainter.inpaint(image, mask).image