# utils/package_manager.py

from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Literal


ProgramMode = Literal["visual", "terminal", "run"]
AudioBackend = Literal["none", "moviepy", "ffmpeg"]


@dataclass(slots=True)
class PackageSpec:
    name: str
    import_name: str
    install_name: str
    pip_installable: bool = True
    note: str = ""


BASE_PACKAGES = [
    PackageSpec(
        name="numpy",
        import_name="numpy",
        install_name="numpy",
        note="Required by almost every module.",
    ),
    PackageSpec(
        name="opencv-python",
        import_name="cv2",
        install_name="opencv-python",
        note="Required for image/video reading, writing, masks, and OpenCV inpainting.",
    ),
    PackageSpec(
        name="pillow",
        import_name="PIL.Image",
        install_name="pillow",
        note="Required for PIL image conversion and generative inpainters.",
    ),
]

UI_PACKAGES = [
    PackageSpec(
        name="rich",
        import_name="rich",
        install_name="rich",
        note="Recommended for better terminal UI.",
    ),
    PackageSpec(
        name="gradio",
        import_name="gradio",
        install_name="gradio",
        note="Required for visual browser UI.",
    ),
]

AUDIO_PACKAGES = [
    PackageSpec(
        name="moviepy",
        import_name="moviepy",
        install_name="moviepy",
        note="Required when audio_backend='moviepy'.",
    ),
    PackageSpec(
        name="ffmpeg",
        import_name="ffmpeg",
        install_name="ffmpeg",
        pip_installable=False,
        note="Required only when audio_backend='ffmpeg'. Install from your OS package manager.",
    ),
]

MODEL_PACKAGES = [
    PackageSpec(
        name="torch",
        import_name="torch",
        install_name="torch",
        note="Required for SDXL, FLUX, SAM2, XMem, CoTracker.",
    ),
    PackageSpec(
        name="diffusers",
        import_name="diffusers",
        install_name="diffusers",
        note="Required for SDXL and FLUX inpainters.",
    ),
    PackageSpec(
        name="transformers",
        import_name="transformers",
        install_name="transformers",
        note="Often required by diffusers models.",
    ),
    PackageSpec(
        name="accelerate",
        import_name="accelerate",
        install_name="accelerate",
        note="Recommended for diffusers model loading.",
    ),
    PackageSpec(
        name="ultralytics",
        import_name="ultralytics",
        install_name="ultralytics",
        note="Required for YOLO detector.",
    ),
    PackageSpec(
        name="easyocr",
        import_name="easyocr",
        install_name="easyocr",
        note="Required for EasyOCR detector.",
    ),
    PackageSpec(
        name="paddleocr",
        import_name="paddleocr",
        install_name="paddleocr",
        note="Required for PaddleOCR detector.",
    ),
    PackageSpec(
        name="paddlepaddle",
        import_name="paddle",
        install_name="paddlepaddle",
        note="Required by PaddleOCR runtime.",
    ),
    PackageSpec(
        name="simple-lama-inpainting",
        import_name="simple_lama_inpainting",
        install_name="simple-lama-inpainting",
        note="Optional LaMa wrapper.",
    ),
]


def specs_for_context(
    program: ProgramMode,
    inpainter: str | None = None,
    ocr: str | None = None,
    audio_backend: AudioBackend | None = None,
    include_model_packages: bool = False,
) -> list[PackageSpec]:
    specs: list[PackageSpec] = []
    specs.extend(BASE_PACKAGES)

    if program == "terminal":
        specs.append(get_spec("rich"))

    if program == "visual":
        specs.append(get_spec("gradio"))

    if audio_backend == "moviepy":
        specs.append(get_spec("moviepy"))

    if audio_backend == "ffmpeg":
        specs.append(get_spec("ffmpeg"))

    if include_model_packages:
        specs.extend(model_specs_for_selection(inpainter=inpainter, ocr=ocr))

    return dedupe_specs(specs)


def model_specs_for_selection(
    inpainter: str | None = None,
    ocr: str | None = None,
) -> list[PackageSpec]:
    specs: list[PackageSpec] = []

    if inpainter in {"sdxl", "flux"}:
        specs.extend(
            [
                get_spec("torch"),
                get_spec("diffusers"),
                get_spec("transformers"),
                get_spec("accelerate"),
            ]
        )

    if inpainter == "lama":
        specs.append(get_spec("simple-lama-inpainting"))

    if ocr in {"easy", "both"}:
        specs.append(get_spec("easyocr"))

    if ocr in {"paddle", "both"}:
        specs.extend(
            [
                get_spec("paddleocr"),
                get_spec("paddlepaddle"),
            ]
        )

    return dedupe_specs(specs)


def get_spec(name: str) -> PackageSpec:
    all_specs = BASE_PACKAGES + UI_PACKAGES + AUDIO_PACKAGES + MODEL_PACKAGES

    for spec in all_specs:
        if spec.name == name:
            return spec

    raise KeyError(f"Unknown package spec: {name}")


def dedupe_specs(specs: list[PackageSpec]) -> list[PackageSpec]:
    seen: set[str] = set()
    result: list[PackageSpec] = []

    for spec in specs:
        if spec.name in seen:
            continue

        seen.add(spec.name)
        result.append(spec)

    return result


def missing_packages(specs: list[PackageSpec]) -> list[PackageSpec]:
    return [
        spec
        for spec in specs
        if not is_installed(spec)
    ]


def is_installed(spec: PackageSpec) -> bool:
    if spec.name == "ffmpeg":
        return shutil.which("ffmpeg") is not None

    try:
        importlib.import_module(spec.import_name)
        return True
    except Exception:
        return False


def format_missing_packages(missing: list[PackageSpec]) -> str:
    lines = []

    for spec in missing:
        lines.append(f"- {spec.name}")
        lines.append(f"  import: {spec.import_name}")

        if spec.pip_installable:
            lines.append(f"  install: pip install {spec.install_name}")
        else:
            lines.append("  install: system package, not pip")

        if spec.note:
            lines.append(f"  note: {spec.note}")

    return "\n".join(lines)


def install_packages(
    packages: list[PackageSpec],
) -> None:
    pip_packages = [
        spec.install_name
        for spec in packages
        if spec.pip_installable
    ]

    non_pip_packages = [
        spec
        for spec in packages
        if not spec.pip_installable
    ]

    if non_pip_packages:
        print("\nThese packages cannot be installed with pip:")
        print(format_missing_packages(non_pip_packages))

    if not pip_packages:
        return

    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        *pip_packages,
    ]

    print("\nInstalling packages:")
    print(" ".join(command))

    subprocess.check_call(command)


def ask_to_install_missing(
    missing: list[PackageSpec],
    assume_yes: bool = False,
) -> bool:
    if not missing:
        return True

    print("\nMissing packages:")
    print(format_missing_packages(missing))

    if assume_yes:
        install_packages(missing)
        return True

    answer = input("\nInstall missing pip packages now? [y/N]: ").strip().lower()

    if answer not in {"y", "yes"}:
        return False

    install_packages(missing)
    return True


def ensure_packages_or_prompt(
    program: ProgramMode,
    inpainter: str | None = None,
    ocr: str | None = None,
    audio_backend: AudioBackend | None = None,
    include_model_packages: bool = False,
    assume_yes: bool = False,
) -> bool:
    specs = specs_for_context(
        program=program,
        inpainter=inpainter,
        ocr=ocr,
        audio_backend=audio_backend,
        include_model_packages=include_model_packages,
    )

    missing = missing_packages(specs)

    if not missing:
        return True

    return ask_to_install_missing(
        missing=missing,
        assume_yes=assume_yes,
    )


def missing_for_visual_ui() -> list[PackageSpec]:
    specs = specs_for_context(
        program="visual",
        audio_backend="moviepy",
        include_model_packages=False,
    )

    return missing_packages(specs)


def install_missing_for_visual_ui() -> str:
    missing = missing_for_visual_ui()

    if not missing:
        return "All visual UI packages are installed."

    install_packages(missing)

    still_missing = missing_for_visual_ui()

    if still_missing:
        return "Some packages are still missing:\n" + format_missing_packages(still_missing)

    return "Installed missing visual UI packages. Restart the app if imports still fail."