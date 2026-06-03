# tests/module_checker.py

from __future__ import annotations

import argparse
import importlib
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass(slots=True)
class CheckResult:
    name: str
    passed: bool
    skipped: bool = False
    error: str | None = None


@dataclass(slots=True)
class CheckReport:
    results: list[CheckResult] = field(default_factory=list)

    def add_pass(self, name: str) -> None:
        self.results.append(CheckResult(name=name, passed=True))

    def add_fail(self, name: str, error: BaseException | str) -> None:
        self.results.append(
            CheckResult(
                name=name,
                passed=False,
                error=str(error),
            )
        )

    def add_skip(self, name: str, reason: str) -> None:
        self.results.append(
            CheckResult(
                name=name,
                passed=True,
                skipped=True,
                error=reason,
            )
        )

    @property
    def failures(self) -> list[CheckResult]:
        return [result for result in self.results if not result.passed]

    @property
    def skipped(self) -> list[CheckResult]:
        return [result for result in self.results if result.skipped]

    def print_summary(self, verbose: bool = False) -> None:
        passed = len(
            [item for item in self.results if item.passed and not item.skipped]
        )
        skipped = len(self.skipped)
        failed = len(self.failures)
        total = len(self.results)

        print("\n========== MODULE CHECK SUMMARY ==========")
        print(f"Total:   {total}")
        print(f"Passed:  {passed}")
        print(f"Skipped: {skipped}")
        print(f"Failed:  {failed}")

        if self.failures:
            print("\nFailures:")
            for item in self.failures:
                print(f"  x {item.name}")
                print(f"    {item.error}")

        if verbose and self.skipped:
            print("\nSkipped:")
            for item in self.skipped:
                print(f"  - {item.name}: {item.error}")

        if failed == 0:
            print("\nAll required module checks passed.")
        else:
            print("\nSome module checks failed.")


CORE_MODULES = [
    "core.enums",
    "core.types",
    "core.logger",
    "core.registry",
    "core.config",
]

UTIL_MODULES = [
    "utils.video_io_utils",
    "utils.frame_extract",
    "utils.video_rebuilder",
]

PREPROCESSING_MODULES = [
    "preprocessing.utils",
    "preprocessing.clahe",
    "preprocessing.denoise",
    "preprocessing.gamma",
    "preprocessing.resize",
    "preprocessing.edge_enhance",
    "preprocessing.fft_enhance",
    "preprocessing.patchify",
]

DETECTOR_MODULES = [
    "detectors.utils",
    "detectors.yolo_detector",
    "detectors.sam2_detector",
    "detectors.mobile_sam_2_detector",
    "detectors.easy_ocr_detector",
    "detectors.paddle_ocr_detector",
    "detectors.fft_detector",
    "detectors.anomaly_detector",
    "detectors.fusion_detector",
]

TRACKER_MODULES = [
    "trackers.optical_flow_tracker",
    "trackers.kalman_tracker",
    "trackers.cotracker",
    "trackers.xmem_tracker",
    "trackers.trajectory_manager",
]

INPAINTER_MODULES = [
    "inpainters.utils",
    "inpainters.opencv_inpainter",
    "inpainters.lama_inpainter",
    "inpainters.sdxl_inpainter",
    "inpainters.flux_inpainter",
]

POSTPROCESSING_MODULES = [
    "postprocessing.utils",
    "postprocessing.artifact_removal",
    "postprocessing.color_matching",
    "postprocessing.seam_blending",
    "postprocessing.sharpening",
    "postprocessing.temporal_smoothing",
]

PIPELINE_MODULES = [
    "pipelines.preprocessing_pipeline",
    "pipelines.detection_pipeline",
    "pipelines.inpainting_pipeline",
    "pipelines.postprocessing_pipeline",
    "pipelines.image_pipeline",
    "pipelines.video_pipeline",
]

UI_MODULES = [
    "ui.terminal_app",
    "ui.visual_app",
]

ROOT_MODULES = [
    "config",
    "control",
    "main",
]

ALL_MODULES = (
    CORE_MODULES
    + UTIL_MODULES
    + PREPROCESSING_MODULES
    + DETECTOR_MODULES
    + TRACKER_MODULES
    + INPAINTER_MODULES
    + POSTPROCESSING_MODULES
    + PIPELINE_MODULES
    + UI_MODULES
    + ROOT_MODULES
)


def import_module(name: str) -> Any:
    return importlib.import_module(name)


def check_imports(report: CheckReport, modules: list[str]) -> None:
    for module_name in modules:
        try:
            import_module(module_name)
            report.add_pass(f"import:{module_name}")
        except Exception as exc:
            report.add_fail(f"import:{module_name}", format_error(exc))


def check_class_instantiation(report: CheckReport) -> None:
    checks: list[tuple[str, str, str | None]] = [
        # Utils
        ("utils.frame_extract", "FrameExtractor", "FrameExtractionConfig"),
        ("utils.video_rebuilder", "VideoRebuilder", "VideoRebuildConfig"),

        # Preprocessing
        ("preprocessing.clahe", "CLAHEPreprocessor", "CLAHEConfig"),
        ("preprocessing.denoise", "DenoisePreprocessor", "DenoiseConfig"),
        ("preprocessing.gamma", "GammaPreprocessor", "GammaConfig"),
        ("preprocessing.resize", "ResizePreprocessor", "ResizeConfig"),
        ("preprocessing.edge_enhance", "EdgeEnhancePreprocessor", "EdgeEnhanceConfig"),
        ("preprocessing.fft_enhance", "FFTEnhancePreprocessor", "FFTEnhanceConfig"),
        ("preprocessing.patchify", "PatchifyPreprocessor", "PatchifyConfig"),

        # Detectors
        ("detectors.yolo_detector", "YOLODetector", "YOLODetectorConfig"),
        ("detectors.sam2_detector", "SAM2Detector", "SAM2DetectorConfig"),
        ("detectors.mobile_sam_2_detector", "MobileSAM2Detector", "MobileSAM2DetectorConfig"),
        ("detectors.easy_ocr_detector", "EasyOCRDetector", "EasyOCRDetectorConfig"),
        ("detectors.paddle_ocr_detector", "PaddleOCRDetector", "PaddleOCRDetectorConfig"),
        ("detectors.fft_detector", "FFTDetector", "FFTDetectorConfig"),
        ("detectors.anomaly_detector", "AnomalyDetector", "AnomalyDetectorConfig"),
        ("detectors.fusion_detector", "FusionDetector", "FusionDetectorConfig"),

        # Trackers
        ("trackers.optical_flow_tracker", "OpticalFlowTracker", "OpticalFlowTrackerConfig"),
        ("trackers.kalman_tracker", "KalmanTracker", "KalmanTrackerConfig"),
        ("trackers.cotracker", "CoTracker", "CoTrackerConfig"),
        ("trackers.xmem_tracker", "XMemTracker", "XMemTrackerConfig"),
        ("trackers.trajectory_manager", "TrajectoryManager", "TrajectoryManagerConfig"),

        # Inpainters
        ("inpainters.opencv_inpainter", "OpenCVInpainter", "OpenCVInpainterConfig"),
        ("inpainters.lama_inpainter", "LaMaInpainter", "LaMaInpainterConfig"),
        ("inpainters.sdxl_inpainter", "SDXLInpainter", "SDXLInpainterConfig"),
        ("inpainters.flux_inpainter", "FluxInpainter", "FluxInpainterConfig"),

        # Postprocessing
        ("postprocessing.artifact_removal", "ArtifactRemovalPostprocessor", "ArtifactRemovalConfig"),
        ("postprocessing.color_matching", "ColorMatchingPostprocessor", "ColorMatchingConfig"),
        ("postprocessing.seam_blending", "SeamBlendingPostprocessor", "SeamBlendingConfig"),
        ("postprocessing.sharpening", "SharpeningPostprocessor", "SharpeningConfig"),
        ("postprocessing.temporal_smoothing", "TemporalSmoothingPostprocessor", "TemporalSmoothingConfig"),

        # Pipelines
        ("pipelines.preprocessing_pipeline", "PreprocessingPipeline", "PreprocessingPipelineConfig"),
        ("pipelines.detection_pipeline", "DetectionPipeline", "DetectionPipelineConfig"),
        ("pipelines.inpainting_pipeline", "InpaintingPipeline", "InpaintingPipelineConfig"),
        ("pipelines.postprocessing_pipeline", "PostprocessingPipeline", "PostprocessingPipelineConfig"),
        ("pipelines.image_pipeline", "ImagePipeline", "ImagePipelineConfig"),
        ("pipelines.video_pipeline", "VideoPipeline", "VideoPipelineConfig"),

        # UI
        ("ui.terminal_app", "TerminalUI", None),
    ]

    for module_name, class_name, config_name in checks:
        check_name = f"instantiate:{module_name}.{class_name}"

        try:
            module = import_module(module_name)
            cls = getattr(module, class_name)

            if config_name is None:
                cls()
            else:
                config_cls = getattr(module, config_name)
                config = config_cls()
                cls(config)

            report.add_pass(check_name)

        except Exception as exc:
            report.add_fail(check_name, format_error(exc))


def check_disabled_smoke_tests(report: CheckReport) -> None:
    image = np.zeros((64, 64, 3), dtype=np.uint8)

    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[20:40, 20:40] = 255

    tests: list[tuple[str, Callable[[], None]]] = [
        ("smoke:preprocessing", lambda: smoke_preprocessing(image)),
        ("smoke:detectors_disabled", lambda: smoke_detectors_disabled(image)),
        ("smoke:fusion_disabled", lambda: smoke_fusion_disabled(image)),
        ("smoke:trackers_disabled", lambda: smoke_trackers_disabled(image, mask)),
        ("smoke:inpainters_disabled", lambda: smoke_inpainters_disabled(image, mask)),
        ("smoke:postprocessing_disabled", lambda: smoke_postprocessing_disabled(image, mask)),
        ("smoke:utils", smoke_utils),
        ("smoke:core_config", smoke_core_config),
        ("smoke:core_registry", smoke_core_registry),
        ("smoke:terminal_app", smoke_terminal_app),
    ]

    for name, test in tests:
        try:
            test()
            report.add_pass(name)
        except Exception as exc:
            report.add_fail(name, format_error(exc))


def smoke_preprocessing(image: np.ndarray) -> None:
    modules = [
        ("preprocessing.clahe", "CLAHEConfig", "CLAHEPreprocessor"),
        ("preprocessing.denoise", "DenoiseConfig", "DenoisePreprocessor"),
        ("preprocessing.gamma", "GammaConfig", "GammaPreprocessor"),
        ("preprocessing.resize", "ResizeConfig", "ResizePreprocessor"),
        ("preprocessing.edge_enhance", "EdgeEnhanceConfig", "EdgeEnhancePreprocessor"),
        ("preprocessing.fft_enhance", "FFTEnhanceConfig", "FFTEnhancePreprocessor"),
        ("preprocessing.patchify", "PatchifyConfig", "PatchifyPreprocessor"),
    ]

    for module_name, config_name, class_name in modules:
        module = import_module(module_name)
        config_cls = getattr(module, config_name)
        cls = getattr(module, class_name)

        processor = cls(config_cls(enabled=False))
        output = processor(image)

        if output is not image and not isinstance(output, np.ndarray):
            raise AssertionError(
                f"{class_name} returned unsupported output: {type(output)}"
            )


def smoke_detectors_disabled(image: np.ndarray) -> None:
    modules = [
        ("detectors.yolo_detector", "YOLODetectorConfig", "YOLODetector"),
        ("detectors.sam2_detector", "SAM2DetectorConfig", "SAM2Detector"),
        ("detectors.easy_ocr_detector", "EasyOCRDetectorConfig", "EasyOCRDetector"),
        ("detectors.paddle_ocr_detector", "PaddleOCRDetectorConfig", "PaddleOCRDetector"),
        ("detectors.fft_detector", "FFTDetectorConfig", "FFTDetector"),
        ("detectors.anomaly_detector", "AnomalyDetectorConfig", "AnomalyDetector"),
    ]

    for module_name, config_name, class_name in modules:
        module = import_module(module_name)
        config_cls = getattr(module, config_name)
        cls = getattr(module, class_name)

        detector = cls(config_cls(enabled=False))
        result = detector.detect(image)

        assert hasattr(result, "mask"), f"{class_name} result missing mask"
        assert result.mask.shape == image.shape[:2], f"{class_name} mask shape mismatch"


def smoke_fusion_disabled(image: np.ndarray) -> None:
    module = import_module("detectors.fusion_detector")

    config = module.FusionDetectorConfig(enabled=False)
    detector = module.FusionDetector(config)

    result = detector.detect(
        inputs=[],
        image_shape=image.shape[:2],
    )

    assert result.mask.shape == image.shape[:2]


def smoke_trackers_disabled(image: np.ndarray, mask: np.ndarray) -> None:
    optical_module = import_module("trackers.optical_flow_tracker")
    optical = optical_module.OpticalFlowTracker(
        optical_module.OpticalFlowTrackerConfig(enabled=False)
    )
    optical_result = optical.track(image, image, mask)
    assert optical_result.shape == mask.shape

    kalman_module = import_module("trackers.kalman_tracker")
    kalman = kalman_module.KalmanTracker(
        kalman_module.KalmanTrackerConfig(enabled=False)
    )
    kalman_result = kalman.track(mask)
    assert kalman_result.shape == mask.shape

    xmem_module = import_module("trackers.xmem_tracker")
    xmem = xmem_module.XMemTracker(
        xmem_module.XMemTrackerConfig(enabled=False)
    )
    xmem.initialize(image, mask)
    xmem_result = xmem.track(image)
    assert xmem_result.shape == mask.shape

    cotracker_module = import_module("trackers.cotracker")
    cotracker = cotracker_module.CoTracker(
        cotracker_module.CoTrackerConfig(enabled=False)
    )
    cotracker_result = cotracker.track_sequence([image, image], mask)
    assert len(cotracker_result) == 2


def smoke_inpainters_disabled(image: np.ndarray, mask: np.ndarray) -> None:
    modules = [
        ("inpainters.opencv_inpainter", "OpenCVInpainterConfig", "OpenCVInpainter"),
        ("inpainters.lama_inpainter", "LaMaInpainterConfig", "LaMaInpainter"),
        ("inpainters.sdxl_inpainter", "SDXLInpainterConfig", "SDXLInpainter"),
        ("inpainters.flux_inpainter", "FluxInpainterConfig", "FluxInpainter"),
    ]

    for module_name, config_name, class_name in modules:
        module = import_module(module_name)
        config_cls = getattr(module, config_name)
        cls = getattr(module, class_name)

        inpainter = cls(config_cls(enabled=False))
        result = inpainter.inpaint(image, mask)

        assert hasattr(result, "image"), f"{class_name} result missing image"
        assert hasattr(result, "mask"), f"{class_name} result missing mask"
        assert result.image.shape == image.shape


def smoke_postprocessing_disabled(image: np.ndarray, mask: np.ndarray) -> None:
    artifact_module = import_module("postprocessing.artifact_removal")
    artifact = artifact_module.ArtifactRemovalPostprocessor(
        artifact_module.ArtifactRemovalConfig(enabled=False)
    )
    assert artifact(image, mask=mask).shape == image.shape

    color_module = import_module("postprocessing.color_matching")
    color = color_module.ColorMatchingPostprocessor(
        color_module.ColorMatchingConfig(enabled=False)
    )
    assert color(image, reference_image=image, mask=mask).shape == image.shape

    seam_module = import_module("postprocessing.seam_blending")
    seam = seam_module.SeamBlendingPostprocessor(
        seam_module.SeamBlendingConfig(enabled=False)
    )
    assert seam(image, mask=mask, original_image=image).shape == image.shape

    sharpen_module = import_module("postprocessing.sharpening")
    sharpen = sharpen_module.SharpeningPostprocessor(
        sharpen_module.SharpeningConfig(enabled=False)
    )
    assert sharpen(image, mask=mask).shape == image.shape

    temporal_module = import_module("postprocessing.temporal_smoothing")
    temporal = temporal_module.TemporalSmoothingPostprocessor(
        temporal_module.TemporalSmoothingConfig(enabled=False)
    )
    frames = temporal([image, image], masks=[mask, mask])
    assert len(frames) == 2


def smoke_utils() -> None:
    frame_module = import_module("utils.frame_extract")
    rebuild_module = import_module("utils.video_rebuilder")
    io_module = import_module("utils.video_io_utils")

    extractor_config = frame_module.FrameExtractionConfig(enabled=False)
    extractor = frame_module.FrameExtractor(extractor_config)

    result = extractor.extract("not_used.mp4")
    assert result.extracted_count == 0

    rebuilder_config = rebuild_module.VideoRebuildConfig(enabled=False)
    rebuilder = rebuild_module.VideoRebuilder(rebuilder_config)

    try:
        rebuilder.rebuild([])
    except RuntimeError:
        pass

    assert hasattr(io_module, "read_frame")
    assert hasattr(io_module, "save_frame")
    assert hasattr(io_module, "prepare_video_frame")


def smoke_core_config() -> None:
    module = import_module("core.config")
    config = module.default_config()
    data = module.to_dict(config)

    assert isinstance(data, dict)
    assert "runtime" in data


def smoke_core_registry() -> None:
    module = import_module("core.registry")

    registry = module.Registry("test")

    class Demo:
        pass

    registry.register("demo", Demo)
    assert registry.get("demo") is Demo
    assert registry.build("demo").__class__ is Demo


def smoke_terminal_app() -> None:
    module = import_module("ui.terminal_app")

    ui = module.TerminalUI(enabled=False)

    ui.info("disabled")
    ui.success("disabled")
    ui.warning("disabled")
    ui.error("disabled")

    program = module.TerminalProgram(
        ui=ui,
        non_interactive=True,
    )

    assert program.ui is ui
    assert program.non_interactive is True


def check_heavy_smoke_tests(report: CheckReport) -> None:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[:, :, 0] = 50
    image[:, :, 1] = 100
    image[:, :, 2] = 150

    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[20:40, 20:40] = 255

    tests: list[tuple[str, Callable[[], None]]] = [
        ("heavy:opencv_inpainter", lambda: heavy_opencv_inpainter(image, mask)),
        ("heavy:fft_detector", lambda: heavy_fft_detector(image)),
        ("heavy:anomaly_detector", lambda: heavy_anomaly_detector(image)),
        ("heavy:postprocessing", lambda: heavy_postprocessing(image, mask)),
        ("heavy:utils_video_frame_prepare", lambda: heavy_utils_video_frame_prepare(image)),
    ]

    for name, test in tests:
        try:
            test()
            report.add_pass(name)
        except ImportError as exc:
            report.add_skip(name, str(exc))
        except Exception as exc:
            report.add_fail(name, format_error(exc))


def heavy_opencv_inpainter(image: np.ndarray, mask: np.ndarray) -> None:
    module = import_module("inpainters.opencv_inpainter")

    inpainter = module.OpenCVInpainter(
        module.OpenCVInpainterConfig(enabled=True)
    )

    result = inpainter.inpaint(image, mask)
    assert result.image.shape == image.shape


def heavy_fft_detector(image: np.ndarray) -> None:
    module = import_module("detectors.fft_detector")

    detector = module.FFTDetector(
        module.FFTDetectorConfig(enabled=True)
    )

    result = detector.detect(image)
    assert result.mask.shape == image.shape[:2]


def heavy_anomaly_detector(image: np.ndarray) -> None:
    module = import_module("detectors.anomaly_detector")

    detector = module.AnomalyDetector(
        module.AnomalyDetectorConfig(enabled=True)
    )

    result = detector.detect(image)
    assert result.mask.shape == image.shape[:2]


def heavy_postprocessing(image: np.ndarray, mask: np.ndarray) -> None:
    artifact_module = import_module("postprocessing.artifact_removal")
    artifact = artifact_module.ArtifactRemovalPostprocessor(
        artifact_module.ArtifactRemovalConfig(enabled=True)
    )
    assert artifact(image, mask=mask).shape == image.shape

    seam_module = import_module("postprocessing.seam_blending")
    seam = seam_module.SeamBlendingPostprocessor(
        seam_module.SeamBlendingConfig(enabled=True)
    )
    assert seam(image, mask=mask, original_image=image).shape == image.shape

    sharpen_module = import_module("postprocessing.sharpening")
    sharpen = sharpen_module.SharpeningPostprocessor(
        sharpen_module.SharpeningConfig(enabled=True)
    )
    assert sharpen(image, mask=mask).shape == image.shape


def heavy_utils_video_frame_prepare(image: np.ndarray) -> None:
    module = import_module("utils.video_io_utils")

    prepared = module.prepare_video_frame(
        image,
        width=image.shape[1],
        height=image.shape[0],
        resize=True,
    )

    assert prepared.shape == image.shape
    assert prepared.dtype == np.uint8


def format_error(exc: BaseException) -> str:
    return "".join(
        traceback.format_exception_only(type(exc), exc)
    ).strip()


def run_module_checks(
    include_heavy: bool = False,
    verbose: bool = False,
) -> int:
    report = CheckReport()

    print("Checking imports...")
    check_imports(report, ALL_MODULES)

    print("Checking class construction...")
    check_class_instantiation(report)

    print("Running lightweight smoke tests...")
    check_disabled_smoke_tests(report)

    if include_heavy:
        print("Running optional heavy smoke tests...")
        check_heavy_smoke_tests(report)

    report.print_summary(verbose=verbose)

    return 1 if report.failures else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether watermwark modules import and run smoke tests."
    )

    parser.add_argument(
        "--heavy",
        action="store_true",
        help="Run optional OpenCV-based smoke tests.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show skipped checks.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raise SystemExit(
        run_module_checks(
            include_heavy=args.heavy,
            verbose=args.verbose,
        )
    )


if __name__ == "__main__":
    main()