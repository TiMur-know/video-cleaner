# postprocessing/color_matching.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from core.logger import log_postprocessing_event
from postprocessing.utils import (
    blend_masked,
    image_to_3_channels,
    lazy_import,
    prepare_image_pair_for_color_matching,
    restore_alpha_channel,
    to_mask_bool,
)


ColorMatchingMethod = Literal["mean_std", "lab_mean_std"]
ColorOrder = Literal["bgr", "rgb"]


@dataclass(slots=True)
class ColorMatchingConfig:
    enabled: bool = True

    method: ColorMatchingMethod = "mean_std"

    input_color_order: ColorOrder = "bgr"

    # Match only repaired mask region.
    masked_only: bool = True

    # Avoid extreme correction.
    max_shift: float = 40.0
    max_scale: float = 2.0

    eps: float = 1e-6

    # Debug logs.
    log_events: bool = True

    # If True, return RGBA when reference image was RGBA.
    restore_reference_alpha: bool = True


class ColorMatchingPostprocessor:
    """
    Match repaired image colors to original/reference image.

    Important:
        Color matching works on 3-channel color data.
        If reference image is RGBA, alpha is temporarily removed,
        then optionally restored after matching.
    """

    name = "color_matching"

    def __init__(self, config: ColorMatchingConfig | None = None) -> None:
        self.config = config or ColorMatchingConfig()

    def __call__(
        self,
        image: np.ndarray,
        reference_image: np.ndarray,
        mask: np.ndarray | None = None,
        context: dict[str, Any] | None = None,
    ) -> np.ndarray:
        return self.apply(
            image=image,
            reference_image=reference_image,
            mask=mask,
            context=context,
        )

    def apply(
        self,
        image: np.ndarray,
        reference_image: np.ndarray,
        mask: np.ndarray | None = None,
        context: dict[str, Any] | None = None,
    ) -> np.ndarray:
        context = context or {}

        self._log_input(
            image=image,
            reference_image=reference_image,
            mask=mask,
            context=context,
        )

        if not self.config.enabled:
            self._log_output(
                output=image,
                reason="disabled",
            )
            return image

        image_3, reference_3, reference_alpha = prepare_image_pair_for_color_matching(
            image=image,
            reference_image=reference_image,
        )

        if self.config.method == "mean_std":
            corrected = self._mean_std_match(
                image=image_3,
                reference=reference_3,
                mask=mask,
            )

        elif self.config.method == "lab_mean_std":
            corrected = self._lab_mean_std_match(
                image=image_3,
                reference=reference_3,
                mask=mask,
            )

        else:
            raise ValueError(f"Unsupported color matching method: {self.config.method}")

        if mask is not None and self.config.masked_only:
            output_3 = blend_masked(
                original=image_3,
                processed=corrected,
                mask=mask,
            )
        else:
            output_3 = corrected

        if self.config.restore_reference_alpha:
            output = restore_alpha_channel(
                image=output_3,
                alpha=reference_alpha,
            )
        else:
            output = output_3

        self._log_output(
            output=output,
            reason=None,
        )

        return output

    def _mean_std_match(
        self,
        image: np.ndarray,
        reference: np.ndarray,
        mask: np.ndarray | None,
    ) -> np.ndarray:
        image_float = image.astype(np.float32)
        reference_float = reference.astype(np.float32)

        source_pixels = self._select_source_pixels(image_float, mask)
        reference_pixels = self._select_reference_pixels(reference_float, mask)

        source_mean = source_pixels.mean(axis=0)
        source_std = source_pixels.std(axis=0) + self.config.eps

        reference_mean = reference_pixels.mean(axis=0)
        reference_std = reference_pixels.std(axis=0) + self.config.eps

        scale = reference_std / source_std
        scale = np.clip(
            scale,
            1.0 / self.config.max_scale,
            self.config.max_scale,
        )

        shift = reference_mean - source_mean
        shift = np.clip(
            shift,
            -self.config.max_shift,
            self.config.max_shift,
        )

        corrected = (image_float - source_mean) * scale + source_mean + shift

        return np.clip(corrected, 0, 255).astype(image.dtype)

    def _lab_mean_std_match(
        self,
        image: np.ndarray,
        reference: np.ndarray,
        mask: np.ndarray | None,
    ) -> np.ndarray:
        cv2 = lazy_import("cv2")

        image = image_to_3_channels(image)
        reference = image_to_3_channels(reference)

        if self.config.input_color_order == "bgr":
            image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            reference_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB)
            back_code = cv2.COLOR_LAB2BGR
        else:
            image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            reference_lab = cv2.cvtColor(reference, cv2.COLOR_RGB2LAB)
            back_code = cv2.COLOR_LAB2RGB

        corrected_lab = self._mean_std_match(
            image=image_lab,
            reference=reference_lab,
            mask=mask,
        )

        return cv2.cvtColor(corrected_lab, back_code)

    @staticmethod
    def _flatten_pixels(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image.reshape(-1, 1)

        return image.reshape(-1, image.shape[-1])

    def _select_source_pixels(
        self,
        image: np.ndarray,
        mask: np.ndarray | None,
    ) -> np.ndarray:
        if mask is None:
            return self._flatten_pixels(image)

        mask_bool = to_mask_bool(mask)

        if image.ndim == 2:
            pixels = image[mask_bool].reshape(-1, 1)
        else:
            pixels = image[mask_bool]

        if len(pixels) == 0:
            return self._flatten_pixels(image)

        return pixels

    def _select_reference_pixels(
        self,
        reference: np.ndarray,
        mask: np.ndarray | None,
    ) -> np.ndarray:
        if mask is None:
            return self._flatten_pixels(reference)

        mask_bool = to_mask_bool(mask)

        # Use surrounding area outside repaired region as reference.
        inverse_mask = ~mask_bool

        if reference.ndim == 2:
            pixels = reference[inverse_mask].reshape(-1, 1)
        else:
            pixels = reference[inverse_mask]

        if len(pixels) == 0:
            return self._flatten_pixels(reference)

        return pixels

    def _log_input(
        self,
        image: np.ndarray,
        reference_image: np.ndarray,
        mask: np.ndarray | None,
        context: dict[str, Any],
    ) -> None:
        if not self.config.log_events:
            return

        log_postprocessing_event(
            self.name,
            "input",
            {
                "image": image,
                "reference_image": reference_image,
                "mask": mask,
                "method": self.config.method,
                "masked_only": self.config.masked_only,
                "restore_reference_alpha": self.config.restore_reference_alpha,
                "enabled": self.config.enabled,
                "context_keys": list(context.keys()),
            },
        )

    def _log_output(
        self,
        output: np.ndarray,
        reason: str | None,
    ) -> None:
        if not self.config.log_events:
            return

        log_postprocessing_event(
            self.name,
            "output",
            {
                "output": output,
                "reason": reason,
                "enabled": self.config.enabled,
            },
        )


def color_match(
    image: np.ndarray,
    reference_image: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    processor = ColorMatchingPostprocessor()
    return processor(image, reference_image=reference_image, mask=mask)
