# tests/package_checker.py

from __future__ import annotations

import importlib
import shutil
from dataclasses import dataclass, field


@dataclass(slots=True)
class PackageCheckResult:
    name: str
    import_name: str
    required: bool
    installed: bool
    note: str = ""


@dataclass(slots=True)
class PackageCheckReport:
    results: list[PackageCheckResult] = field(default_factory=list)

    @property
    def missing_required(self) -> list[PackageCheckResult]:
        return [
            result
            for result in self.results
            if result.required and not result.installed
        ]

    @property
    def missing_optional(self) -> list[PackageCheckResult]:
        return [
            result
            for result in self.results
            if not result.required and not result.installed
        ]

    def print_summary(self, verbose: bool = False) -> None:
        installed = len([item for item in self.results if item.installed])
        missing_required = len(self.missing_required)
        missing_optional = len(self.missing_optional)
        total = len(self.results)

        print("\n========== PACKAGE CHECK SUMMARY ==========")
        print(f"Total:            {total}")
        print(f"Installed:        {installed}")
        print(f"Missing required: {missing_required}")
        print(f"Missing optional: {missing_optional}")

        if self.missing_required:
            print("\nMissing required packages:")
            for item in self.missing_required:
                print(f"  x {item.name}")
                print(f"    import: {item.import_name}")

                if item.note:
                    print(f"    {item.note}")

        if self.missing_optional:
            print("\nMissing optional packages:")
            for item in self.missing_optional:
                print(f"  - {item.name}")
                print(f"    import: {item.import_name}")

                if item.note:
                    print(f"    {item.note}")

        if verbose:
            self.print_installed_packages()

        if missing_required == 0:
            print("\nAll required packages are installed.")
        else:
            print("\nSome required packages are missing.")

    def print_installed_packages(self) -> None:
        installed_items = [
            item
            for item in self.results
            if item.installed
        ]

        if not installed_items:
            return

        print("\nInstalled packages:")
        for item in installed_items:
            kind = "required" if item.required else "optional"
            print(f"  + {item.name} ({kind})")


REQUIRED_PACKAGES: list[tuple[str, str, str]] = [
    (
        "numpy",
        "numpy",
        "Install with: pip install numpy",
    ),
]


RECOMMENDED_PACKAGES: list[tuple[str, str, str]] = [
    (
        "opencv-python",
        "cv2",
        "Install with: pip install opencv-python",
    ),
    (
        "pillow",
        "PIL.Image",
        "Install with: pip install pillow",
    ),
    (
        "rich",
        "rich",
        "Install with: pip install rich",
    ),
    (
        "gradio",
        "gradio",
        "Install with: pip install gradio",
    ),
    (
        "moviepy",
        "moviepy",
        "Needed when video rebuild audio_backend='moviepy'. "
        "Install with: pip install moviepy",
    ),
]


OPTIONAL_MODEL_PACKAGES: list[tuple[str, str, str]] = [
    (
        "torch",
        "torch",
        "Needed for SDXL, FLUX, SAM2, XMem, and CoTracker.",
    ),
    (
        "diffusers",
        "diffusers",
        "Needed for SDXL and FLUX inpainters.",
    ),
    (
        "transformers",
        "transformers",
        "Often needed by diffusers pipelines.",
    ),
    (
        "accelerate",
        "accelerate",
        "Recommended for diffusers model loading.",
    ),
    (
        "ultralytics",
        "ultralytics",
        "Needed for YOLO detector.",
    ),
    (
        "easyocr",
        "easyocr",
        "Needed for EasyOCR detector.",
    ),
    (
        "paddleocr",
        "paddleocr",
        "Needed for PaddleOCR detector.",
    ),
    (
        "mobile-sam-2",
        "mobile_sam_2",
        "Optional. Needed only if you use the MobileSAM2 detector. Package name may differ depending on your source.",
    ),
    (
        "paddlepaddle",
        "paddle",
        "Needed for PaddleOCR runtime.",
    ),
    (
        "pytesseract",
        "pytesseract",
        "Only needed if you add or use Tesseract OCR.",
    ),
    (
        "simple-lama-inpainting",
        "simple_lama_inpainting",
        "Optional LaMa wrapper.",
    ),
]


def check_packages(
    include_optional: bool = True,
    rich_required: bool = False,
    gradio_required: bool = False,
    moviepy_required: bool = False,
) -> PackageCheckReport:
    """
    Check Python dependencies.

    Required:
        numpy

    Recommended:
        opencv-python
        pillow
        rich
        gradio
        moviepy

    Optional model packages:
        torch
        diffusers
        transformers
        accelerate
        ultralytics
        OCR packages
        LaMa wrapper

    System executable:
        ffmpeg, only needed when audio_backend='ffmpeg'
    """

    report = PackageCheckReport()

    for package_name, import_name, note in REQUIRED_PACKAGES:
        report.results.append(
            check_import(
                package_name=package_name,
                import_name=import_name,
                required=True,
                note=note,
            )
        )

    for package_name, import_name, note in RECOMMENDED_PACKAGES:
        required = False

        if package_name == "rich" and rich_required:
            required = True

        if package_name == "gradio" and gradio_required:
            required = True

        if package_name == "moviepy" and moviepy_required:
            required = True

        report.results.append(
            check_import(
                package_name=package_name,
                import_name=import_name,
                required=required,
                note=note,
            )
        )

    if include_optional:
        for package_name, import_name, note in OPTIONAL_MODEL_PACKAGES:
            report.results.append(
                check_import(
                    package_name=package_name,
                    import_name=import_name,
                    required=False,
                    note=note,
                )
            )

    report.results.append(check_ffmpeg())

    return report


def check_import(
    package_name: str,
    import_name: str,
    required: bool,
    note: str,
) -> PackageCheckResult:
    try:
        importlib.import_module(import_name)
        installed = True
    except Exception:
        installed = False

    return PackageCheckResult(
        name=package_name,
        import_name=import_name,
        required=required,
        installed=installed,
        note=note,
    )


def check_ffmpeg() -> PackageCheckResult:
    installed = shutil.which("ffmpeg") is not None

    return PackageCheckResult(
        name="ffmpeg",
        import_name="ffmpeg",
        required=False,
        installed=installed,
        note="Needed only when video rebuild audio_backend='ffmpeg'.",
    )


def run_package_checks(
    include_optional: bool = True,
    verbose: bool = False,
    rich_required: bool = False,
    gradio_required: bool = False,
    moviepy_required: bool = False,
) -> int:
    report = check_packages(
        include_optional=include_optional,
        rich_required=rich_required,
        gradio_required=gradio_required,
        moviepy_required=moviepy_required,
    )

    report.print_summary(verbose=verbose)

    return 1 if report.missing_required else 0