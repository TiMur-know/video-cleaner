# pipelines/preprocessing_pipeline.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from preprocessing.resize import ResizeConfig, ResizePreprocessor
from preprocessing.denoise import DenoiseConfig, DenoisePreprocessor
from preprocessing.gamma import GammaConfig, GammaPreprocessor
from preprocessing.clahe import CLAHEConfig, CLAHEPreprocessor
from preprocessing.edge_enhance import EdgeEnhanceConfig, EdgeEnhancePreprocessor
from preprocessing.fft_enhance import FFTEnhanceConfig, FFTEnhancePreprocessor
from preprocessing.patchify import PatchifyConfig, PatchifyPreprocessor


@dataclass(slots=True)
class PreprocessingPipelineConfig:
    """
    Config for the whole preprocessing pipeline.

    Recommended order:
        resize
        denoise
        gamma
        clahe
        edge_enhance

    Special branches:
        fft_enhance is usually used only for FFT detector input.
        patchify is usually used only before patch-based detectors.
    """

    resize: ResizeConfig = field(default_factory=ResizeConfig)
    denoise: DenoiseConfig = field(default_factory=DenoiseConfig)
    gamma: GammaConfig = field(default_factory=GammaConfig)
    clahe: CLAHEConfig = field(default_factory=CLAHEConfig)
    edge_enhance: EdgeEnhanceConfig = field(default_factory=EdgeEnhanceConfig)
    fft_enhance: FFTEnhanceConfig = field(default_factory=FFTEnhanceConfig)
    patchify: PatchifyConfig = field(default_factory=PatchifyConfig)

    # Main detector path.
    enable_main_path: bool = True

    # FFT-specialized path.
    enable_fft_path: bool = False

    # Patch-specialized path.
    enable_patch_path: bool = False


@dataclass(slots=True)
class PreprocessingResult:
    """
    Output of the preprocessing pipeline.

    main_image:
        Image used by normal detectors:
        OCR, YOLO, SAM2, anomaly detector.

    fft_image:
        Optional image used by FFTDetector.

    patches:
        Optional patches used by patch-based detectors.

    metadata:
        Useful info for downstream pipelines.
    """

    main_image: np.ndarray
    fft_image: np.ndarray | None = None
    patches: list[Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class PreprocessingPipeline:
    """
    Main preprocessing orchestrator.

    This pipeline prepares inputs for detectors.
    It does not perform detection itself.
    """

    def __init__(
        self,
        config: PreprocessingPipelineConfig | None = None,
    ) -> None:
        self.config = config or PreprocessingPipelineConfig()

        self.resize = ResizePreprocessor(self.config.resize)
        self.denoise = DenoisePreprocessor(self.config.denoise)
        self.gamma = GammaPreprocessor(self.config.gamma)
        self.clahe = CLAHEPreprocessor(self.config.clahe)
        self.edge_enhance = EdgeEnhancePreprocessor(self.config.edge_enhance)
        self.fft_enhance = FFTEnhancePreprocessor(self.config.fft_enhance)
        self.patchify = PatchifyPreprocessor(self.config.patchify)

    def run(
        self,
        image: np.ndarray,
        context: dict[str, Any] | None = None,
    ) -> PreprocessingResult:
        """
        Run preprocessing.

        Parameters
        ----------
        image:
            Input image or video frame.

        context:
            Optional extra info, for example:
                {
                    "input_type": "image",
                    "frame_index": 12,
                    "source": "video"
                }

        Returns
        -------
        PreprocessingResult
        """

        context = context or {}

        metadata: dict[str, Any] = {
            "original_shape": image.shape,
            "input_dtype": str(image.dtype),
        }

        main_image = image

        if self.config.enable_main_path:
            main_image = self._run_main_path(main_image, context)

        fft_image = None

        if self.config.enable_fft_path:
            fft_image = self._run_fft_path(main_image, context)

        patches = None

        if self.config.enable_patch_path:
            patches = self._run_patch_path(main_image, context)

        metadata["main_shape"] = main_image.shape
        metadata["main_dtype"] = str(main_image.dtype)

        if fft_image is not None:
            metadata["fft_shape"] = fft_image.shape
            metadata["fft_dtype"] = str(fft_image.dtype)

        if patches is not None:
            metadata["num_patches"] = len(patches)

        return PreprocessingResult(
            main_image=main_image,
            fft_image=fft_image,
            patches=patches,
            metadata=metadata,
        )

    def _run_main_path(
        self,
        image: np.ndarray,
        context: dict[str, Any],
    ) -> np.ndarray:
        """
        Main preprocessing path for most detectors.

        Used by:
            OCRDetector
            YOLODetector
            SAM2Detector
            AnomalyDetector
        """

        image = self.resize(image, context=context)
        image = self.denoise(image, context=context)
        image = self.gamma(image, context=context)
        image = self.clahe(image, context=context)
        image = self.edge_enhance(image, context=context)

        return image

    def _run_fft_path(
        self,
        image: np.ndarray,
        context: dict[str, Any],
    ) -> np.ndarray:
        """
        FFT-specific preprocessing path.

        Used by:
            FFTDetector

        Usually this should not replace the main image.
        It creates a separate frequency-enhanced version.
        """

        return self.fft_enhance(image, context=context)

    def _run_patch_path(
        self,
        image: np.ndarray,
        context: dict[str, Any],
    ) -> list[Any]:
        """
        Patch-specific preprocessing path.

        Used by:
            patch-based YOLO, SAM2, OCR, or anomaly detection.
        """

        patches = self.patchify(image, context=context)

        if isinstance(patches, np.ndarray):
            return [patches]

        return patches