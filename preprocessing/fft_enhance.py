# preprocessing/fft_enhance.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from preprocessing.utils import restore_dtype, to_float01, validate_image


FFTMode = Literal["high_pass", "band_pass", "notch_suppress"]


@dataclass(slots=True)
class FFTEnhanceConfig:
    enabled: bool = True

    # "high_pass" is the safest first option.
    mode: FFTMode = "high_pass"

    # Removes low-frequency background variation.
    high_pass_radius: int = 20

    # Used only for band_pass mode.
    low_radius: int = 10
    high_radius: int = 80

    # Used to blend FFT result with original image.
    strength: float = 0.7

    # If True, normalize enhanced image to full [0, 1] range.
    normalize: bool = True


class FFTEnhancePreprocessor:
    """
    FFT-based frequency enhancement.

    Purpose:
        Highlight periodic, repeated, or faint watermark patterns.

    Useful before:
        FFTDetector, anomaly detector, OCR detector.

    Notes:
        - This is NumPy-only.
        - No OpenCV dependency.
        - Works on grayscale and color images.
    """

    name = "fft_enhance"

    def __init__(self, config: FFTEnhanceConfig | None = None) -> None:
        self.config = config or FFTEnhanceConfig()

    def __call__(
        self,
        image: np.ndarray,
        context: dict[str, Any] | None = None,
    ) -> np.ndarray:
        return self.apply(image, context=context)

    def apply(
        self,
        image: np.ndarray,
        context: dict[str, Any] | None = None,
    ) -> np.ndarray:
        if not self.config.enabled:
            return image

        validate_image(image, self.name)

        if image.ndim == 3 and image.shape[-1] == 4:
            color = image[..., :3]
            alpha = image[..., 3:]
            output = self.apply(color, context=context)
            return np.concatenate([output, alpha], axis=-1)

        original_dtype = image.dtype
        image_float = to_float01(image)

        if image_float.ndim == 2:
            output = self._process_channel(image_float)
        else:
            channels = [
                self._process_channel(image_float[..., idx])
                for idx in range(image_float.shape[-1])
            ]
            output = np.stack(channels, axis=-1)

        return restore_dtype(output, original_dtype)

    def _process_channel(self, channel: np.ndarray) -> np.ndarray:
        spectrum = np.fft.fft2(channel)
        spectrum_shifted = np.fft.fftshift(spectrum)

        mask = self._build_mask(channel.shape)

        filtered_spectrum = spectrum_shifted * mask

        filtered = np.fft.ifftshift(filtered_spectrum)
        enhanced = np.fft.ifft2(filtered)
        enhanced = np.abs(enhanced)

        if self.config.normalize:
            enhanced = self._normalize01(enhanced)

        blended = (
            (1.0 - self.config.strength) * channel
            + self.config.strength * enhanced
        )

        return np.clip(blended, 0.0, 1.0)

    def _build_mask(self, shape: tuple[int, int]) -> np.ndarray:
        height, width = shape

        y = np.arange(height) - height // 2
        x = np.arange(width) - width // 2

        xx, yy = np.meshgrid(x, y)
        radius = np.sqrt(xx**2 + yy**2)

        if self.config.mode == "high_pass":
            return self._high_pass_mask(radius)

        if self.config.mode == "band_pass":
            return self._band_pass_mask(radius)

        if self.config.mode == "notch_suppress":
            return self._notch_suppress_mask(radius)

        raise ValueError(f"Unsupported FFT enhance mode: {self.config.mode}")

    def _high_pass_mask(self, radius: np.ndarray) -> np.ndarray:
        """
        Keeps high frequencies and suppresses low-frequency background.

        Good for faint text/logos and semi-transparent watermarks.
        """
        mask = np.ones_like(radius, dtype=np.float32)
        mask[radius < self.config.high_pass_radius] = 0.0
        return mask

    def _band_pass_mask(self, radius: np.ndarray) -> np.ndarray:
        """
        Keeps only a frequency band.

        Useful when watermark texture lies in a known frequency range.
        """
        mask = np.zeros_like(radius, dtype=np.float32)

        keep = (
            (radius >= self.config.low_radius)
            & (radius <= self.config.high_radius)
        )

        mask[keep] = 1.0
        return mask

    def _notch_suppress_mask(self, radius: np.ndarray) -> np.ndarray:
        """
        Soft suppression of very low frequencies.

        Less aggressive than hard high-pass filtering.
        """
        radius_safe = np.maximum(radius, 1.0)

        mask = 1.0 - np.exp(
            -(radius_safe**2) / (2.0 * self.config.high_pass_radius**2)
        )

        return mask.astype(np.float32)

    @staticmethod
    def _normalize01(channel: np.ndarray) -> np.ndarray:
        min_value = float(np.min(channel))
        max_value = float(np.max(channel))

        if max_value - min_value < 1e-8:
            return np.zeros_like(channel, dtype=np.float32)

        return ((channel - min_value) / (max_value - min_value)).astype(np.float32)


def fft_enhance(
    image: np.ndarray,
    mode: FFTMode = "high_pass",
    strength: float = 0.7,
    high_pass_radius: int = 20,
) -> np.ndarray:
    processor = FFTEnhancePreprocessor(
        FFTEnhanceConfig(
            mode=mode,
            strength=strength,
            high_pass_radius=high_pass_radius,
        )
    )

    return processor(image)