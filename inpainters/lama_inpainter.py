# inpainters/lama_inpainter.py

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from core.logger import log_event
from inpainters.utils import (
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

    model_path: str = "models/lama/big-lama.pt"

    # LaMa expects H/W divisible by 8.
    modulo: int = 8

    # Expand mask slightly before inpainting.
    dilate_mask_iterations: int = 1
    mask_kernel_size: int = 5

    check_model_path: bool = True
    log_events: bool = True


class LaMaInpainter:
    """
    Simple local big-lama.pt inpainter.

    No simple-lama-inpainting.
    No iopaint.

    Expected model:
        torch.jit.load("models/lama/big-lama.pt")

    Expected forward:
        model(image_tensor, mask_tensor)

    Tensor format:
        image: Bx3xHxW, RGB float32, 0..1
        mask:  Bx1xHxW, float32, 0 or 1
    """

    name = "lama"

    def __init__(
        self,
        config: LaMaInpainterConfig | None = None,
        model: Any | None = None,
    ) -> None:
        self.config = config or LaMaInpainterConfig()
        self.model = model
        self._loaded = model is not None

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
        context = context or {}

        mask_uint8 = to_mask_uint8(mask)

        self._log(
            "input",
            {
                "image_shape": image.shape if image is not None else None,
                "mask_shape": mask_uint8.shape,
                "mask_area": int(np.count_nonzero(mask_uint8)),
                "enabled": self.config.enabled,
                "model_path": self.config.model_path,
                "model_loaded": self.model is not None,
                "context_keys": list(context.keys()),
            },
        )

        if not self.config.enabled:
            return self._result(
                image=image,
                mask=mask_uint8,
                enabled=False,
                reason="disabled",
            )

        validate_image(image, self.name)
        validate_mask(mask_uint8, image.shape[:2], self.name)

        if self.model is None:
            self.model = self._load_model()
            self._loaded = True

        output = self._run_lama(
            image=image,
            mask=mask_uint8,
        )

        result = self._result(
            image=output,
            mask=mask_uint8,
            enabled=True,
            reason=None,
        )

        self._log_output(result)
        return result

    def _load_model(self) -> Any:
        model_path = Path(self.config.model_path).expanduser()

        if self.config.check_model_path and not model_path.exists():
            raise FileNotFoundError(
                f"LaMa model file not found: {model_path}. "
                "Expected: models/lama/big-lama.pt"
            )

        torch = self._torch()
        device = self._device()

        self._log(
            "load",
            {
                "model_path": str(model_path.resolve()),
                "device": str(device),
            },
        )

        model = torch.jit.load(
            str(model_path.resolve()),
            map_location=device,
        )
        model.eval()
        model.to(device)

        return model

    def _run_lama(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        original_height, original_width = image.shape[:2]

        image_tensor = self._image_to_tensor(image)
        mask_tensor = self._mask_to_tensor(mask)

        torch = self._torch()

        with torch.inference_mode():
            output = self.model(image_tensor, mask_tensor)

        output_rgb = self._tensor_to_image(output)

        # Remove padding.
        output_rgb = output_rgb[:original_height, :original_width]

        if self.config.output_color_order == "bgr":
            return output_rgb[:, :, ::-1]

        return output_rgb

    def _image_to_tensor(self, image: np.ndarray) -> Any:
        torch = self._torch()

        image_rgb = self._to_rgb(image)
        image_rgb = self._pad_to_modulo(
            image_rgb.astype(np.float32) / 255.0,
            value_mode="reflect",
        )

        # HWC -> BCHW
        tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0)
        return tensor.float().to(self._device())

    def _mask_to_tensor(self, mask: np.ndarray) -> Any:
        torch = self._torch()

        mask_uint8 = to_mask_uint8(mask)

        if self.config.dilate_mask_iterations > 0:
            cv2 = self._cv2()

            kernel_size = max(1, int(self.config.mask_kernel_size))
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

            mask_uint8 = cv2.dilate(
                mask_uint8,
                kernel,
                iterations=self.config.dilate_mask_iterations,
            )

        mask_float = mask_uint8.astype(np.float32)

        if mask_float.max() > 1.0:
            mask_float /= 255.0

        mask_float = (mask_float > 0.5).astype(np.float32)

        mask_float = self._pad_to_modulo(
            mask_float,
            value_mode="constant",
        )

        # HW -> B1HW
        tensor = torch.from_numpy(mask_float).unsqueeze(0).unsqueeze(0)
        return tensor.float().to(self._device())

    def _tensor_to_image(self, output: Any) -> np.ndarray:
        if isinstance(output, dict):
            output = self._extract_tensor_from_dict(output)

        if isinstance(output, (list, tuple)):
            output = output[0]

        # BCHW -> CHW
        if output.ndim == 4:
            output = output[0]

        # CHW -> HWC
        output = output.permute(1, 2, 0).detach().cpu().numpy()

        if output.max() <= 1.5:
            output *= 255.0

        return np.clip(output, 0, 255).astype(np.uint8)

    def _extract_tensor_from_dict(self, output: dict[str, Any]) -> Any:
        for key in (
            "inpainted",
            "predicted_image",
            "output",
            "out",
            "image",
        ):
            if key in output:
                return output[key]

        raise RuntimeError(
            f"Unsupported LaMa output dict keys: {list(output.keys())}"
        )

    def _to_rgb(self, image: np.ndarray) -> np.ndarray:
        if image.ndim != 3:
            raise ValueError(f"Expected HxWxC image, got {image.shape}")

        if image.shape[2] == 4:
            image = image[:, :, :3]

        if image.shape[2] != 3:
            raise ValueError(f"Expected 3-channel image, got {image.shape}")

        if self.config.input_color_order == "bgr":
            return image[:, :, ::-1]

        return image

    def _pad_to_modulo(
        self,
        array: np.ndarray,
        value_mode: Literal["reflect", "constant"],
    ) -> np.ndarray:
        modulo = max(1, int(self.config.modulo))

        height, width = array.shape[:2]

        pad_height = (modulo - height % modulo) % modulo
        pad_width = (modulo - width % modulo) % modulo

        if pad_height == 0 and pad_width == 0:
            return array

        if array.ndim == 2:
            padding = ((0, pad_height), (0, pad_width))
        else:
            padding = ((0, pad_height), (0, pad_width), (0, 0))

        if value_mode == "constant":
            return np.pad(array, padding, mode="constant", constant_values=0)

        return np.pad(array, padding, mode="reflect")

    def _result(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        enabled: bool,
        reason: str | None,
    ) -> LaMaInpaintResult:
        return LaMaInpaintResult(
            image=image,
            mask=mask,
            metadata={
                "enabled": enabled,
                "reason": reason,
                "inpainter": self.name,
                "backend": "torchscript_big_lama",
                "device": resolve_device(self.config.device),
                "model_path": self.config.model_path,
                "model_loaded": self._loaded,
                "mask_area": int(np.count_nonzero(mask)),
                "image_shape": image.shape if image is not None else None,
            },
        )

    def _log_output(self, result: LaMaInpaintResult) -> None:
        self._log(
            "output",
            {
                "enabled": result.metadata.get("enabled"),
                "reason": result.metadata.get("reason"),
                "image_shape": result.image.shape if result.image is not None else None,
                "mask_shape": result.mask.shape if result.mask is not None else None,
                "mask_area": result.metadata.get("mask_area"),
                "model_loaded": result.metadata.get("model_loaded"),
                "device": result.metadata.get("device"),
            },
        )

    def _device(self) -> Any:
        torch = self._torch()
        return torch.device(resolve_device(self.config.device))

    @staticmethod
    def _torch() -> Any:
        import torch

        return torch

    @staticmethod
    def _cv2() -> Any:
        import cv2

        return cv2

    def _log(
        self,
        event: str,
        payload: dict[str, Any],
    ) -> None:
        if self.config.log_events:
            log_event(self.name, event, payload)


def lama_inpaint(
    image: np.ndarray,
    mask: np.ndarray,
    model: Any | None = None,
) -> np.ndarray:
    inpainter = LaMaInpainter(model=model)
    return inpainter.inpaint(image, mask).image
