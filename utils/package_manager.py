# utils/package_manager.py

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
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


@dataclass(slots=True)
class ModelSpec:
    name: str
    kind: Literal["local", "hf"]
    identifier: str
    local_path: str | None = None
    installable: bool = False
    expected_size_bytes: int | None = None
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
        note="Required for Stable Diffusion, SDXL, FLUX, SAM2, XMem, CoTracker.",
    ),
    PackageSpec(
        name="diffusers",
        import_name="diffusers",
        install_name="diffusers",
        note="Required for Stable Diffusion, SDXL, and FLUX inpainters.",
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
        name="huggingface_hub",
        import_name="huggingface_hub",
        install_name="huggingface_hub",
        note="Required for downloading and removing cached Hugging Face models.",
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

MODEL_SPECS = [
    ModelSpec(
        name="GroundingDINO base",
        kind="hf",
        identifier="IDEA-Research/grounding-dino-base",
        local_path="models/grounding-dino-base",
        installable=True,
        expected_size_bytes=700 * 1024 * 1024,
        note="Used by the GroundingDINO proposal detector.",
    ),
    ModelSpec(
        name="SDXL inpainting",
        kind="hf",
        identifier="diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        local_path="models/stable-diffusion-xl-1.0-inpainting-0.1",
        installable=True,
        expected_size_bytes=13 * 1024 * 1024 * 1024,
        note="Used by the SDXL inpainter.",
    ),
    ModelSpec(
        name="Stable Diffusion inpainting",
        kind="hf",
        identifier="runwayml/stable-diffusion-inpainting",
        local_path="models/stable-diffusion-inpainting",
        installable=True,
        expected_size_bytes=5 * 1024 * 1024 * 1024,
        note="Used by the classic Stable Diffusion inpainter.",
    ),
    ModelSpec(
        name="Flux Fill",
        kind="hf",
        identifier="black-forest-labs/FLUX.1-Fill-dev",
        local_path="models/FLUX.1-Fill-dev",
        installable=True,
        expected_size_bytes=45 * 1024 * 1024 * 1024,
        note="Used by the Flux inpainter. Access may require Hugging Face login.",
    ),
    ModelSpec(
        name="YOLO watermark checkpoint",
        kind="local",
        identifier="models/yolo/watermarks_s_yolov8_v1.pt",
        expected_size_bytes=25 * 1024 * 1024,
        note="Local YOLO detector checkpoint.",
    ),
    ModelSpec(
        name="SAM2 checkpoint",
        kind="local",
        identifier="models/sam2/sam2_b.pt",
        expected_size_bytes=375 * 1024 * 1024,
        note="Local SAM2 refiner checkpoint.",
    ),
    ModelSpec(
        name="SAM2 config",
        kind="local",
        identifier="models/sam2/sam2_hiera_b+.yaml",
        expected_size_bytes=10 * 1024,
        note="Local SAM2 model config file.",
    ),
    ModelSpec(
        name="MobileSAM2 checkpoint",
        kind="local",
        identifier="models/mobile_sam/mobile_sam.pt",
        expected_size_bytes=40 * 1024 * 1024,
        note="Local MobileSAM2 checkpoint.",
    ),
    ModelSpec(
        name="LaMa checkpoint",
        kind="local",
        identifier="models/lama/big-lama.pt",
        expected_size_bytes=200 * 1024 * 1024,
        note="Local LaMa TorchScript checkpoint.",
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

    if inpainter in {"stable_diffusion", "sdxl", "flux"}:
        specs.extend(
            [
                get_spec("torch"),
                get_spec("diffusers"),
                get_spec("transformers"),
                get_spec("accelerate"),
                get_spec("huggingface_hub"),
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


def all_package_specs() -> list[PackageSpec]:
    return dedupe_specs(BASE_PACKAGES + UI_PACKAGES + AUDIO_PACKAGES + MODEL_PACKAGES)


def get_model_spec(name: str) -> ModelSpec:
    for spec in MODEL_SPECS:
        if spec.name == name:
            return spec

    raise KeyError(f"Unknown model spec: {name}")


def package_choices() -> list[str]:
    return [spec.name for spec in all_package_specs()]


def model_choices() -> list[str]:
    return [spec.name for spec in MODEL_SPECS]


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
        return importlib.util.find_spec(spec.import_name) is not None
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


def package_status() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for spec in all_package_specs():
        installed = is_installed(spec)
        rows.append(
            {
                "name": spec.name,
                "installed": installed,
                "source": "pip" if spec.pip_installable else "system",
                "import": spec.import_name,
                "install": spec.install_name if spec.pip_installable else None,
                "note": spec.note,
            }
        )

    return rows


def model_status() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for spec in MODEL_SPECS:
        installed = is_model_installed(spec)
        installed_size_bytes = model_installed_size_bytes(spec) if installed else None
        expected_size_bytes = spec.expected_size_bytes
        rows.append(
            {
                "name": spec.name,
                "installed": installed,
                "kind": spec.kind,
                "identifier": spec.identifier,
                "local_path": model_local_path(spec),
                "installable": spec.installable,
                "installed_size": format_size(installed_size_bytes),
                "installed_size_bytes": installed_size_bytes,
                "expected_size": format_size(expected_size_bytes),
                "expected_size_bytes": expected_size_bytes,
                "shown_size": format_size(installed_size_bytes or expected_size_bytes),
                "shown_size_type": "installed" if installed_size_bytes is not None else "estimated",
                "note": spec.note,
            }
        )

    return rows


def full_environment_status() -> dict[str, Any]:
    return {
        "packages": package_status(),
        "models": model_status(),
    }


def installed_package_names() -> list[str]:
    return [
        row["name"]
        for row in package_status()
        if row["installed"]
    ]


def installed_model_names() -> list[str]:
    return [
        row["name"]
        for row in model_status()
        if row["installed"]
    ]


def format_installed_package_names() -> str:
    return format_name_list(
        installed_package_names(),
        empty_text="No tracked runtime packages are installed.",
    )


def format_installed_model_names() -> str:
    rows = [
        row
        for row in model_status()
        if row["installed"]
    ]

    if not rows:
        return "No tracked models or checkpoints are installed."

    return "\n".join(
        (
            f"- {row['name']} ({row['installed_size'] or 'size unknown'})"
            f"\n  path: {row['local_path']}"
        )
        for row in rows
    )


def format_model_size_summary() -> str:
    lines: list[str] = []

    for row in model_status():
        state = "installed" if row["installed"] else "missing"
        size = row["installed_size"] if row["installed"] else row["expected_size"]
        size_type = "actual" if row["installed_size"] else "estimated"

        if not size:
            size = "size unknown"
            size_type = "unknown"

        path = row.get("local_path")
        path_text = f", path {path}" if path else ""
        lines.append(f"- {row['name']}: {state}, {size_type} size {size}{path_text}")

    return "\n".join(lines)


def format_name_list(
    names: list[str],
    empty_text: str,
) -> str:
    if not names:
        return empty_text

    return "\n".join(f"- {name}" for name in names)


def is_model_installed(spec: ModelSpec) -> bool:
    if spec.kind == "local":
        return Path(spec.identifier).expanduser().exists()

    local_path = model_local_path(spec)
    return local_path is not None and Path(local_path).expanduser().exists()


def model_installed_size_bytes(spec: ModelSpec) -> int | None:
    if spec.kind == "local":
        return path_size_bytes(Path(spec.identifier).expanduser())

    local_path = model_local_path(spec)
    if local_path is None:
        return None

    return path_size_bytes(Path(local_path).expanduser())


def model_local_path(spec: ModelSpec) -> str | None:
    if spec.kind == "local":
        return spec.identifier

    return spec.local_path


def path_size_bytes(path: Path) -> int | None:
    if not path.exists():
        return None

    try:
        if path.is_file():
            return path.stat().st_size

        return sum(
            item.stat().st_size
            for item in path.rglob("*")
            if item.is_file()
        )
    except Exception:
        return None


def hf_model_cached(model_id: str) -> bool:
    try:
        from huggingface_hub import scan_cache_dir
    except Exception:
        return False

    try:
        cache_info = scan_cache_dir()
    except Exception:
        return False

    return any(
        repo.repo_id == model_id
        for repo in cache_info.repos
    )


def hf_model_cached_size_bytes(model_id: str) -> int | None:
    try:
        from huggingface_hub import scan_cache_dir
    except Exception:
        return None

    try:
        cache_info = scan_cache_dir()
    except Exception:
        return None

    sizes: list[int] = []

    for repo in cache_info.repos:
        if repo.repo_id != model_id:
            continue

        size = getattr(repo, "size_on_disk", None)

        if isinstance(size, int):
            sizes.append(size)

    if not sizes:
        return None

    return sum(sizes)


def format_size(size_bytes: int | None) -> str | None:
    if size_bytes is None:
        return None

    value = float(size_bytes)

    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024 or unit == "TB":
            if unit == "B":
                return f"{int(value)} {unit}"

            return f"{value:.1f} {unit}"

        value /= 1024

    return f"{size_bytes} B"


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


def uninstall_packages(
    packages: list[PackageSpec],
) -> None:
    pip_packages = [
        spec.install_name
        for spec in packages
        if spec.pip_installable
    ]

    if not pip_packages:
        return

    command = [
        sys.executable,
        "-m",
        "pip",
        "uninstall",
        "-y",
        *pip_packages,
    ]

    print("\nUninstalling packages:")
    print(" ".join(command))

    subprocess.check_call(command)


def install_models(
    models: list[ModelSpec],
) -> str:
    lines: list[str] = []

    for spec in models:
        if spec.kind == "local":
            lines.append(
                f"{spec.name}: local file is not auto-downloadable. Expected: {spec.identifier}"
            )
            continue

        try:
            from huggingface_hub import snapshot_download
        except Exception:
            lines.append(
                f"{spec.name}: install huggingface_hub first, then retry."
            )
            continue

        try:
            local_path = model_local_path(spec)

            if local_path is None:
                lines.append(f"{spec.name}: no local model path configured.")
                continue

            Path(local_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id=spec.identifier,
                local_dir=local_path,
            )
            lines.append(f"{spec.name}: downloaded to {local_path}.")
        except Exception as exc:
            lines.append(f"{spec.name}: download failed: {exc}")

    return "\n".join(lines) if lines else "No models selected."


def uninstall_models(
    models: list[ModelSpec],
) -> str:
    lines: list[str] = []

    for spec in models:
        if spec.kind == "local":
            lines.append(uninstall_local_model(spec))
            continue

        lines.append(uninstall_hf_model(spec))

    return "\n".join(lines) if lines else "No models selected."


def uninstall_local_model(spec: ModelSpec) -> str:
    path = Path(spec.identifier).expanduser()

    if not path.exists():
        return f"{spec.name}: already missing."

    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    except Exception as exc:
        return f"{spec.name}: uninstall failed: {exc}"

    return f"{spec.name}: removed {path}."


def uninstall_hf_model(spec: ModelSpec) -> str:
    local_path = model_local_path(spec)

    if local_path is None:
        return f"{spec.name}: no local model path configured."

    path = Path(local_path).expanduser()

    if not path.exists():
        return f"{spec.name}: already missing from {path}."

    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    except Exception as exc:
        return f"{spec.name}: uninstall failed: {exc}"

    return f"{spec.name}: removed {path}."


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


def install_packages_by_name(names: list[str] | None) -> str:
    specs = [
        get_spec(name)
        for name in names or []
    ]

    if not specs:
        return "No packages selected."

    try:
        install_packages(specs)
    except Exception as exc:
        return f"Install failed: {exc}"

    return "Install finished.\n\n" + format_package_status_for_names(names or [])


def uninstall_packages_by_name(names: list[str] | None) -> str:
    specs = [
        get_spec(name)
        for name in names or []
    ]

    if not specs:
        return "No packages selected."

    try:
        uninstall_packages(specs)
    except Exception as exc:
        return f"Uninstall failed: {exc}"

    return "Uninstall finished.\n\n" + format_package_status_for_names(names or [])


def install_models_by_name(names: list[str] | None) -> str:
    specs = [
        get_model_spec(name)
        for name in names or []
    ]

    return install_models(specs)


def uninstall_models_by_name(names: list[str] | None) -> str:
    specs = [
        get_model_spec(name)
        for name in names or []
    ]

    return uninstall_models(specs)


def format_package_status_for_names(names: list[str]) -> str:
    rows = {
        row["name"]: row
        for row in package_status()
    }

    lines: list[str] = []

    for name in names:
        row = rows.get(name)

        if row is None:
            continue

        state = "installed" if row["installed"] else "missing"
        lines.append(f"- {name}: {state}")

    return "\n".join(lines)
