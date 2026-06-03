# ui/terminal_app.py

from __future__ import annotations

import json
import traceback
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable

from config import DEFAULT_CONFIG_PATH
from controls.control import (
    apply_preset,
    set_audio_backend,
    set_device,
    set_inpainter,
    set_ocr,
    set_tracker,
)
from core.config import AppConfig, infer_mode_from_path, load_config
from core.logger import get_logger
from pipelines.image_pipeline import ImagePipeline
from pipelines.video_pipeline import VideoPipeline
from utils.package_manager import ensure_packages_or_prompt


class TerminalUI:
    """
    Low-level terminal UI.

    Uses rich if installed.
    Falls back to normal terminal input/print.
    """

    def __init__(
        self,
        enabled: bool = True,
        verbose: bool = False,
    ) -> None:
        self.enabled = enabled
        self.verbose = verbose
        self.rich_available = False
        self.console = None

        if self.enabled:
            self._setup_rich()

    def _setup_rich(self) -> None:
        try:
            from rich.console import Console

            self.console = Console()
            self.rich_available = True
        except Exception:
            self.console = None
            self.rich_available = False

    def title(self, text: str) -> None:
        if not self.enabled:
            return

        if self.rich_available:
            from rich.panel import Panel

            self.console.print(
                Panel.fit(
                    text,
                    title="watermwark",
                    border_style="cyan",
                )
            )
        else:
            print(f"\n=== {text} ===")

    def section(self, text: str) -> None:
        if not self.enabled:
            return

        if self.rich_available:
            self.console.rule(f"[bold cyan]{text}")
        else:
            print(f"\n--- {text} ---")

    def info(self, message: str) -> None:
        self._print(message, style="blue", prefix="info")

    def success(self, message: str) -> None:
        self._print(message, style="green", prefix="ok")

    def warning(self, message: str) -> None:
        self._print(message, style="yellow", prefix="warn")

    def error(self, message: str) -> None:
        self._print(message, style="red", prefix="error")

    def debug(self, message: str) -> None:
        if self.verbose:
            self._print(message, style="dim", prefix="debug")

    def _print(
        self,
        message: str,
        style: str = "white",
        prefix: str | None = None,
    ) -> None:
        if not self.enabled:
            return

        prefix_text = f"[{prefix}] " if prefix else ""

        if self.rich_available:
            self.console.print(f"{prefix_text}{message}", style=style)
        else:
            print(f"{prefix_text}{message}")

    def ask(
        self,
        message: str,
        default: str | None = None,
    ) -> str:
        if not self.enabled:
            if default is not None:
                return default

            return input(f"{message}: ").strip()

        if self.rich_available:
            from rich.prompt import Prompt

            return Prompt.ask(message, default=default)

        if default is not None:
            value = input(f"{message} [{default}]: ").strip()
            return value or default

        return input(f"{message}: ").strip()

    def ask_choice(
        self,
        message: str,
        choices: list[str],
        default: str,
    ) -> str:
        if self.rich_available:
            from rich.prompt import Prompt

            return Prompt.ask(
                message,
                choices=choices,
                default=default,
            )

        choices_text = "/".join(choices)
        value = input(f"{message} ({choices_text}) [{default}]: ").strip()
        value = value or default

        if value not in choices:
            self.warning(f"Invalid choice '{value}', using '{default}'")
            return default

        return value

    def ask_path(
        self,
        message: str,
        default: str | None = None,
        must_exist: bool = False,
    ) -> str:
        while True:
            value = self.ask(message, default=default)
            value = value.strip().strip('"').strip("'")

            if not must_exist:
                return value

            if Path(value).exists():
                return value

            self.warning(f"Path does not exist: {value}")

    def print_json(
        self,
        data: Any,
        title: str | None = None,
    ) -> None:
        if not self.enabled:
            return

        safe_data = self._json_safe(data)
        text = json.dumps(safe_data, indent=2)

        if self.rich_available:
            from rich.syntax import Syntax

            if title:
                self.section(title)

            self.console.print(
                Syntax(
                    text,
                    "json",
                    theme="monokai",
                    line_numbers=False,
                )
            )
        else:
            if title:
                print(f"\n--- {title} ---")
            print(text)

    def print_table(
        self,
        title: str,
        rows: dict[str, Any],
    ) -> None:
        if not self.enabled:
            return

        if self.rich_available:
            from rich.table import Table

            table = Table(title=title)
            table.add_column("Setting", style="cyan")
            table.add_column("Value", style="white")

            for key, value in rows.items():
                table.add_row(str(key), str(value))

            self.console.print(table)
        else:
            print(f"\n--- {title} ---")
            for key, value in rows.items():
                print(f"{key}: {value}")

    def run_with_status(
        self,
        message: str,
        func: Callable[[], Any],
    ) -> Any:
        if not self.enabled:
            return func()

        if self.rich_available:
            with self.console.status(message, spinner="dots"):
                return func()

        print(message)
        return func()

    def print_exception(self, exc: BaseException) -> None:
        if self.rich_available:
            self.console.print_exception(show_locals=self.verbose)
            return

        if self.verbose:
            traceback.print_exception(type(exc), exc, exc.__traceback__)
        else:
            self.error(f"{type(exc).__name__}: {exc}")

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if is_dataclass(value):
            return TerminalUI._json_safe(asdict(value))

        if isinstance(value, dict):
            return {
                str(key): TerminalUI._json_safe(item)
                for key, item in value.items()
            }

        if isinstance(value, (list, tuple)):
            return [TerminalUI._json_safe(item) for item in value]

        if isinstance(value, Path):
            return str(value)

        if hasattr(value, "value"):
            return value.value

        return value


class TerminalProgram:
    """
    High-level terminal workflow.

    Replaces old program_ui.py.
    """

    def __init__(
        self,
        ui: TerminalUI,
        non_interactive: bool = False,
    ) -> None:
        self.ui = ui
        self.non_interactive = non_interactive

    def load_config(
        self,
        config_path: str | None = None,
    ) -> tuple[AppConfig, str | None]:
        resolved_path = self.resolve_config_path(config_path)

        config = self.ui.run_with_status(
            "Loading configuration...",
            lambda: load_config(resolved_path),
        )

        return config, resolved_path

    def resolve_config_path(
        self,
        config_path: str | None = None,
    ) -> str | None:
        if config_path is not None:
            path = Path(config_path)

            if path.exists():
                return str(path)

            if self.non_interactive:
                raise FileNotFoundError(f"Config file not found: {path}")

            self.ui.warning(f"Config file not found: {path}")

            answer = self.ui.ask_choice(
                "Use default config instead?",
                choices=["yes", "no"],
                default="yes",
            )

            if answer == "yes":
                return None

            return self.ui.ask_path(
                "Enter config.json path",
                must_exist=True,
            )

        if DEFAULT_CONFIG_PATH.exists():
            return str(DEFAULT_CONFIG_PATH)

        if self.non_interactive:
            return None

        answer = self.ui.ask_choice(
            "No config.json found. Use defaults?",
            choices=["yes", "no"],
            default="yes",
        )

        if answer == "yes":
            return None

        return self.ui.ask_path(
            "Enter config.json path",
            must_exist=True,
        )

    def resolve_mode(
        self,
        config: AppConfig,
        input_path: str | None = None,
        mode_arg: str | None = None,
    ) -> str:
        if mode_arg is not None and mode_arg != "auto":
            config.runtime.mode = mode_arg
            return mode_arg

        if input_path is not None:
            mode = infer_mode_from_path(input_path)
            config.runtime.mode = mode
            return mode

        if config.runtime.mode != "auto":
            return config.runtime.mode

        if Path(config.image.input_path).exists():
            config.runtime.mode = "image"
            return "image"

        if Path(config.video.input_path).exists():
            config.runtime.mode = "video"
            return "video"

        if self.non_interactive:
            raise FileNotFoundError(
                "Could not infer mode. Provide --input or --type."
            )

        mode = self.ui.ask_choice(
            "What do you want to process?",
            choices=["image", "video"],
            default="image",
        )

        config.runtime.mode = mode
        return mode

    def ensure_paths(
        self,
        config: AppConfig,
        mode: str,
        input_path: str | None = None,
        output_path: str | None = None,
    ) -> None:
        if mode == "image":
            self.ensure_image_paths(
                config=config,
                input_path=input_path,
                output_path=output_path,
            )
            return

        if mode == "video":
            self.ensure_video_paths(
                config=config,
                input_path=input_path,
                output_path=output_path,
            )
            return

        raise ValueError(f"Unsupported mode: {mode}")

    def ensure_image_paths(
        self,
        config: AppConfig,
        input_path: str | None = None,
        output_path: str | None = None,
    ) -> None:
        if input_path is not None:
            config.image.input_path = input_path

        if output_path is not None:
            config.image.output_path = output_path

        image_path = Path(config.image.input_path)

        if not image_path.exists():
            if self.non_interactive:
                raise FileNotFoundError(f"Image input not found: {image_path}")

            self.ui.warning(f"Image input not found: {image_path}")

            config.image.input_path = self.ui.ask_path(
                "Enter image input path",
                default=str(image_path),
                must_exist=True,
            )

        if not config.image.output_path:
            config.image.output_path = self.ui.ask_path(
                "Enter image output path",
                default="data/outputs/output.png",
                must_exist=False,
            )

        if config.image.mask_output_path is None:
            config.image.mask_output_path = "data/masks/mask.png"

    def ensure_video_paths(
        self,
        config: AppConfig,
        input_path: str | None = None,
        output_path: str | None = None,
    ) -> None:
        if input_path is not None:
            config.video.input_path = input_path

        if output_path is not None:
            config.video.output_path = output_path

        video_path = Path(config.video.input_path)

        if not video_path.exists():
            if self.non_interactive:
                raise FileNotFoundError(f"Video input not found: {video_path}")

            self.ui.warning(f"Video input not found: {video_path}")

            config.video.input_path = self.ui.ask_path(
                "Enter video input path",
                default=str(video_path),
                must_exist=True,
            )

        if not config.video.output_path:
            config.video.output_path = self.ui.ask_path(
                "Enter video output path",
                default="data/outputs/output.mp4",
                must_exist=False,
            )

        config.video.rebuild.output_path = config.video.output_path
        config.video.rebuild.source_video_path = config.video.input_path

    def show_start(
        self,
        config: AppConfig,
        mode: str,
        config_path: str | None,
        preset: str,
    ) -> None:
        self.ui.title("watermwark pipeline")

        self.ui.print_table(
            "Run settings",
            {
                "mode": mode,
                "config": config_path or "defaults",
                "preset": preset,
                "input": self.get_input_path(config, mode),
                "output": self.get_output_path(config, mode),
                "inpainter": self.get_inpainter_name(config, mode),
                "tracker": config.video.tracking.mode if mode == "video" else "n/a",
                "audio_backend": config.video.rebuild.audio_backend if mode == "video" else "n/a",
                "copy_audio": config.video.rebuild.copy_audio if mode == "video" else "n/a",
            },
        )

    def show_image_result(self, result: Any) -> None:
        self.ui.success("Image pipeline finished.")

        self.ui.print_table(
            "Output",
            {
                "image": result.output_path,
                "mask": result.mask_output_path,
                "mask_area": result.metadata.get("mask_area"),
            },
        )

    def show_video_result(self, result: Any) -> None:
        rebuild_metadata = result.metadata.get("rebuild", {})

        self.ui.success("Video pipeline finished.")

        self.ui.print_table(
            "Output",
            {
                "video": result.output_path,
                "processed_frames": len(result.processed_frame_paths),
                "tracking": result.metadata.get("tracking_mode"),
                "audio_backend": rebuild_metadata.get("audio_backend"),
                "copy_audio": rebuild_metadata.get("copy_audio"),
                "mask_area_total": result.metadata.get("mask_area_total"),
            },
        )

    @staticmethod
    def get_input_path(config: AppConfig, mode: str) -> str:
        if mode == "image":
            return config.image.input_path

        if mode == "video":
            return config.video.input_path

        return "unknown"

    @staticmethod
    def get_output_path(config: AppConfig, mode: str) -> str:
        if mode == "image":
            return config.image.output_path

        if mode == "video":
            return config.video.output_path

        return "unknown"

    @staticmethod
    def get_inpainter_name(config: AppConfig, mode: str) -> str:
        if mode == "image":
            return config.image.inpainting.inpainter

        if mode == "video":
            return config.video.inpainting.inpainter

        return "unknown"


def add_terminal_ui_args(parser: Any) -> None:
    parser.add_argument(
        "--no-ui",
        action="store_true",
        help="Disable terminal UI formatting.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show verbose terminal output and full exceptions.",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress most terminal output.",
    )


def build_terminal_ui(args: Any) -> TerminalUI:
    return TerminalUI(
        enabled=not args.no_ui and not args.quiet,
        verbose=bool(args.verbose),
    )


def run_terminal_app(
    ui: TerminalUI,
    config_path: str | None = None,
    non_interactive: bool = False,
) -> None:
    program = TerminalProgram(
        ui=ui,
        non_interactive=non_interactive,
    )

    config, resolved_config_path = program.load_config(config_path)

    ui.title("watermwark terminal app")

    mode = ui.ask_choice(
        "What do you want to process?",
        choices=["image", "video"],
        default="image",
    )

    config.runtime.mode = mode

    program.ensure_paths(
        config=config,
        mode=mode,
    )

    preset = ui.ask_choice(
        "Choose quality preset",
        choices=["default", "fast", "balanced", "quality"],
        default="balanced",
    )

    apply_preset(config, preset)  # type: ignore[arg-type]

    inpainter = ui.ask_choice(
        "Choose inpainter",
        choices=["auto", "opencv", "lama", "sdxl", "flux"],
        default=program.get_inpainter_name(config, mode),
    )

    set_inpainter(config, inpainter)  # type: ignore[arg-type]

    tracker = "none"

    if mode == "video":
        tracker = ui.ask_choice(
            "Choose tracker",
            choices=["none", "optical_flow", "kalman", "xmem", "cotracker"],
            default=config.video.tracking.mode,
        )

        set_tracker(config, tracker)  # type: ignore[arg-type]

    ocr = ui.ask_choice(
        "Choose OCR",
        choices=["none", "paddle", "easy", "both"],
        default="paddle",
    )

    set_ocr(config, ocr)  # type: ignore[arg-type]

    device = ui.ask_choice(
        "Choose device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
    )

    set_device(config, device)  # type: ignore[arg-type]

    audio_backend: str | None = None

    if mode == "video":
        audio_backend = ui.ask_choice(
            "Choose audio backend",
            choices=["none", "moviepy", "ffmpeg"],
            default="moviepy",
        )

        set_audio_backend(config, audio_backend)  # type: ignore[arg-type]

    packages_ok = ensure_packages_or_prompt(
        program="terminal",
        inpainter=inpainter,
        ocr=ocr,
        audio_backend=audio_backend,  # type: ignore[arg-type]
        include_model_packages=True,
        assume_yes=False,
    )

    if not packages_ok:
        ui.warning("Missing packages were not installed. Pipeline was not started.")
        return

    program.show_start(
        config=config,
        mode=mode,
        config_path=resolved_config_path,
        preset=preset,
    )

    should_run = ui.ask_choice(
        "Run pipeline now?",
        choices=["yes", "no"],
        default="yes",
    )

    if should_run == "no":
        ui.warning("Pipeline was not started.")
        return

    logger = get_logger(
        name=config.runtime.project_name,
        level=config.runtime.log_level,
        log_file=config.runtime.log_file,
    )

    logger.info("Starting %s pipeline from terminal app", mode)

    if mode == "image":
        result = ui.run_with_status(
            "Running image pipeline...",
            lambda: run_image(config),
        )

        program.show_image_result(result)
        return

    if mode == "video":
        result = ui.run_with_status(
            "Running video pipeline...",
            lambda: run_video(config),
        )

        program.show_video_result(result)
        return

    raise ValueError(f"Unsupported mode: {mode}")


def run_image(config: AppConfig) -> Any:
    pipeline = ImagePipeline(config.image)

    return pipeline.run(
        input_path=config.image.input_path,
        output_path=config.image.output_path,
        context={
            "app": config.runtime.project_name,
            "mode": "image",
            "ui": "terminal",
        },
    )


def run_video(config: AppConfig) -> Any:
    config.video.rebuild.output_path = config.video.output_path
    config.video.rebuild.source_video_path = config.video.input_path

    pipeline = VideoPipeline(config.video)

    return pipeline.run(
        input_path=config.video.input_path,
        output_path=config.video.output_path,
        context={
            "app": config.runtime.project_name,
            "mode": "video",
            "ui": "terminal",
        },
    )