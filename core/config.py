# core/config.py

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal
import json

from pipelines.image_pipeline import ImagePipelineConfig
from pipelines.video_pipeline import VideoPipelineConfig
from pipelines.mask_processing_pipeline import MaskProcessingPipelineConfig


RunMode = Literal["auto", "image", "video"]


@dataclass(slots=True)
class PathConfig:
    input_dir: str = "data/inputs"
    output_dir: str = "data/outputs"
    mask_dir: str = "data/masks"
    cache_dir: str = "data/cache"
    temp_dir: str = "data/temp"
    log_dir: str = "data/logs"
    model_dir: str = "models"


@dataclass(slots=True)
class RuntimeConfig:
    project_name: str = "watermwark"
    mode: RunMode = "auto"

    log_level: str = "INFO"
    log_file: str | None = "data/logs/watermwark.log"

    create_dirs: bool = True
    seed: int = 42


@dataclass(slots=True)
class ModuleConfig:
    """
    Optional loose config section for custom module overrides.

    Detailed config still lives inside:
        image: ImagePipelineConfig
        video: VideoPipelineConfig
    """

    preprocessing: dict[str, Any] = field(default_factory=dict)
    detectors: dict[str, Any] = field(default_factory=dict)
    trackers: dict[str, Any] = field(default_factory=dict)
    inpainters: dict[str, Any] = field(default_factory=dict)
    postprocessing: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AppConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    image: ImagePipelineConfig = field(default_factory=ImagePipelineConfig)
    video: VideoPipelineConfig = field(default_factory=VideoPipelineConfig)

    modules: ModuleConfig = field(default_factory=ModuleConfig)


def default_config() -> AppConfig:
    return AppConfig()


def load_config(path: str | None = None) -> AppConfig:
    """
    Load config from JSON.

    If path is None, returns default config.
    """

    config = default_config()

    if path is None:
        if config.runtime.create_dirs:
            ensure_project_dirs(config)
        return config

    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with config_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    update_dataclass(config, data)

    if config.runtime.create_dirs:
        ensure_project_dirs(config)

    return config


def save_config(config: AppConfig, path: str) -> None:
    """
    Save config to JSON.
    """

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(to_dict(config), file, indent=2)


def to_dict(config: AppConfig) -> dict[str, Any]:
    """
    Convert config to JSON-safe dictionary.
    """

    return make_json_safe(asdict(config))


def update_dataclass(instance: Any, values: dict[str, Any]) -> Any:
    """
    Recursively update a dataclass instance from a dictionary.

    Unknown keys are ignored.
    """

    if not is_dataclass(instance):
        return instance

    valid_fields = {item.name for item in fields(instance)}

    for key, value in values.items():
        if key not in valid_fields:
            continue

        current_value = getattr(instance, key)

        if is_dataclass(current_value) and isinstance(value, dict):
            update_dataclass(current_value, value)
        else:
            setattr(instance, key, value)

    return instance


def make_json_safe(value: Any) -> Any:
    """
    Convert dataclasses, enums, paths, tuples, and nested values to JSON-safe data.
    """

    if is_dataclass(value):
        return make_json_safe(asdict(value))

    if isinstance(value, dict):
        return {
            str(key): make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]

    if isinstance(value, Path):
        return str(value)

    if hasattr(value, "value"):
        return value.value

    return value


def ensure_project_dirs(config: AppConfig) -> None:
    """
    Create required project directories.
    """

    directories = [
        config.paths.input_dir,
        config.paths.output_dir,
        config.paths.mask_dir,
        config.paths.cache_dir,
        config.paths.temp_dir,
        config.paths.log_dir,
        config.paths.model_dir,
        config.image.output_path.rsplit("/", 1)[0],
        config.video.output_path.rsplit("/", 1)[0],
        config.video.frame_dir,
        config.video.processed_frame_dir,
    ]

    if config.image.mask_output_path is not None:
        directories.append(config.image.mask_output_path.rsplit("/", 1)[0])

    for directory in directories:
        if directory:
            Path(directory).mkdir(parents=True, exist_ok=True)


def infer_mode_from_path(path: str) -> Literal["image", "video"]:
    """
    Infer whether input path is image or video based on extension.
    """

    suffix = Path(path).suffix.lower()

    image_exts = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".bmp",
        ".tif",
        ".tiff",
    }

    video_exts = {
        ".mp4",
        ".mov",
        ".avi",
        ".mkv",
        ".webm",
        ".m4v",
    }

    if suffix in image_exts:
        return "image"

    if suffix in video_exts:
        return "video"

    raise ValueError(
        f"Cannot infer input type from extension '{suffix}'. "
        "Use --type image or --type video."
    )