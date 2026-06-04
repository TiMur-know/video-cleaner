# main.py

from __future__ import annotations

import argparse
from typing import Any

from utils.env_loader import (
    env_bool,
    env_choice,
    env_int,
    env_str,
    load_env_file,
)
from utils.package_manager import ensure_packages_or_prompt


def preparse_args() -> argparse.Namespace:
    """
    Small early parser.

    Purpose:
        - load .env
        - decide program before importing heavy UI/pipeline modules
        - check/install missing packages before Gradio/Rich/model imports
    """

    load_env_file()

    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument(
        "--program",
        choices=["auto", "visual", "terminal", "run"],
        default=env_choice(
            "WATERMWARK_PROGRAM",
            ["auto", "visual", "terminal", "run"],
            "auto",
        ),
    )

    parser.add_argument(
        "--non-interactive",
        action="store_true",
        default=env_bool("WATERMWARK_NON_INTERACTIVE", False),
    )

    parser.add_argument(
        "--yes-install",
        action="store_true",
        default=env_bool("WATERMWARK_YES_INSTALL", False),
    )

    parser.add_argument(
        "--no-package-check",
        action="store_true",
        default=env_bool("WATERMWARK_NO_PACKAGE_CHECK", False),
    )

    parser.add_argument(
        "--audio-backend",
        choices=["none", "moviepy", "ffmpeg"],
        default=env_choice(
            "WATERMWARK_AUDIO_BACKEND",
            ["none", "moviepy", "ffmpeg"],
            "moviepy",
        ),
    )

    parser.add_argument(
        "--inpainter",
        choices=["auto", "opencv", "lama", "stable_diffusion", "sdxl", "flux"],
        default=env_choice(
            "WATERMWARK_INPAINTER",
            ["auto", "opencv", "lama", "stable_diffusion", "sdxl", "flux"],
            "auto",
        ),
    )

    parser.add_argument(
        "--ocr",
        choices=["none", "paddle", "easy", "both"],
        default=env_choice(
            "WATERMWARK_OCR",
            ["none", "paddle", "easy", "both"],
            "paddle",
        ),
    )

    args, _ = parser.parse_known_args()
    return args


def resolve_program_before_imports(args: argparse.Namespace) -> str:
    """
    Decide which program to open before importing heavier modules.
    """

    if args.program != "auto":
        return args.program

    if args.non_interactive:
        return "run"

    print("\n=== watermwark launcher ===")
    print("Choose program:")
    print("1. visual")
    print("2. terminal")
    print("3. run")

    answer = input("Program [visual]: ").strip().lower()

    if answer in {"2", "terminal"}:
        return "terminal"

    if answer in {"3", "run"}:
        return "run"

    return "visual"


def startup_package_check(
    program: str,
    args: argparse.Namespace,
) -> None:
    """
    Check packages before importing visual/terminal/pipeline modules.
    """

    if args.no_package_check:
        return

    audio_backend = args.audio_backend or "moviepy"

    ok = ensure_packages_or_prompt(
        program=program,  # type: ignore[arg-type]
        inpainter=args.inpainter,
        ocr=args.ocr,
        audio_backend=audio_backend,  # type: ignore[arg-type]
        include_model_packages=False,
        assume_yes=args.yes_install,
    )

    if not ok:
        raise SystemExit(
            "Missing packages were not installed. "
            "Install them or rerun with --yes-install."
        )


def parse_args() -> argparse.Namespace:
    """
    Full parser.

    .env values are used as defaults.
    CLI args override .env.
    """

    from controls.control import add_control_args
    from ui.terminal_app import add_terminal_ui_args

    load_env_file()

    parser = argparse.ArgumentParser(
        prog="watermwark",
        description="Watermark detection and removal pipeline.",
    )

    parser.add_argument(
        "--program",
        choices=["auto", "visual", "terminal", "run"],
        default=env_choice(
            "WATERMWARK_PROGRAM",
            ["auto", "visual", "terminal", "run"],
            "auto",
        ),
        help="Program mode: visual UI, terminal UI, or direct run.",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=env_str("WATERMWARK_CONFIG"),
        help="Path to JSON config file.",
    )

    parser.add_argument(
        "--input",
        type=str,
        default=env_str("WATERMWARK_INPUT"),
        help="Input image or video path.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=env_str("WATERMWARK_OUTPUT"),
        help="Output image or video path.",
    )

    parser.add_argument(
        "--type",
        choices=["auto", "image", "video"],
        default=env_choice(
            "WATERMWARK_TYPE",
            ["auto", "image", "video"],
            "auto",
        ),
        help="Input type.",
    )

    parser.add_argument(
        "--save-config",
        type=str,
        default=None,
        help="Save current resolved config to this path and exit.",
    )

    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print resolved config and exit.",
    )

    parser.add_argument(
        "--non-interactive",
        action="store_true",
        default=env_bool("WATERMWARK_NON_INTERACTIVE", False),
        help="Do not ask for missing paths.",
    )

    parser.add_argument(
        "--host",
        type=str,
        default=env_str("WATERMWARK_HOST", "127.0.0.1"),
        help="Visual UI host.",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=env_int("WATERMWARK_PORT", 7860),
        help="Visual UI port.",
    )

    parser.add_argument(
        "--yes-install",
        action="store_true",
        default=env_bool("WATERMWARK_YES_INSTALL", False),
        help="Automatically install missing pip packages.",
    )

    parser.add_argument(
        "--no-package-check",
        action="store_true",
        default=env_bool("WATERMWARK_NO_PACKAGE_CHECK", False),
        help="Skip startup package check.",
    )

    add_control_args(parser)
    add_terminal_ui_args(parser)

    return parser.parse_args()


def main() -> None:
    pre_args = preparse_args()

    program = resolve_program_before_imports(pre_args)

    startup_package_check(
        program=program,
        args=pre_args,
    )

    args = parse_args()

    if args.program == "auto":
        args.program = program

    from controls.control import (
        apply_controls,
        control_summary,
    )
    from core.config import (
        save_config,
        to_dict,
    )
    from core.logger import get_logger
    from ui.terminal_app import (
        TerminalProgram,
        build_terminal_ui,
        run_terminal_app,
    )
    from ui.visual_app import launch_visual_app

    ui = build_terminal_ui(args)

    try:
        if args.program == "visual":
            ui.info("Opening visual UI...")
            launch_visual_app(
                config_path=args.config,
                server_name=args.host,
                server_port=args.port,
            )
            return

        if args.program == "terminal":
            run_terminal_app(
                ui=ui,
                config_path=args.config,
                non_interactive=args.non_interactive,
            )
            return

        if args.program == "run":
            run_direct(
                args=args,
                ui=ui,
                TerminalProgram=TerminalProgram,
                apply_controls=apply_controls,
                control_summary=control_summary,
                save_config=save_config,
                to_dict=to_dict,
                get_logger=get_logger,
            )
            return

        raise ValueError(f"Unsupported program mode: {args.program}")

    except Exception as exc:
        ui.error("Program failed.")
        ui.print_exception(exc)
        raise SystemExit(1) from exc


def run_direct(
    args: argparse.Namespace,
    ui: Any,
    TerminalProgram: Any,
    apply_controls: Any,
    control_summary: Any,
    save_config: Any,
    to_dict: Any,
    get_logger: Any,
) -> None:
    terminal_program = TerminalProgram(
        ui=ui,
        non_interactive=args.non_interactive,
    )

    config, config_path = terminal_program.load_config(args.config)

    options = apply_controls(config, args)

    mode = terminal_program.resolve_mode(
        config=config,
        input_path=args.input,
        mode_arg=args.type,
    )

    terminal_program.ensure_paths(
        config=config,
        mode=mode,
        input_path=args.input,
        output_path=args.output,
    )

    logger = get_logger(
        name=config.runtime.project_name,
        level=config.runtime.log_level,
        log_file=config.runtime.log_file,
    )

    terminal_program.show_start(
        config=config,
        mode=mode,
        config_path=config_path,
        preset=options.preset,
    )

    if args.print_config:
        ui.print_json(to_dict(config), title="Resolved config")
        return

    if options.dry_run:
        ui.print_json(
            control_summary(config, mode, options),
            title="Dry run summary",
        )
        return

    if args.save_config is not None:
        save_config(config, args.save_config)
        ui.success(f"Saved config to {args.save_config}")
        logger.info("Saved config to %s", args.save_config)
        return

    logger.info("Starting %s pipeline", mode)

    if mode == "image":
        result = ui.run_with_status(
            "Running image pipeline...",
            lambda: run_image(config),
        )

        terminal_program.show_image_result(result)
        logger.info("Image pipeline finished: %s", result.output_path)
        return

    if mode == "video":
        result = ui.run_with_status(
            "Running video pipeline...",
            lambda: run_video(config),
        )

        terminal_program.show_video_result(result)
        logger.info("Video pipeline finished: %s", result.output_path)
        return

    raise ValueError(f"Unsupported mode: {mode}")


def run_image(config: Any) -> Any:
    from pipelines.image_pipeline import ImagePipeline

    pipeline = ImagePipeline(config.image)

    return pipeline.run(
        input_path=config.image.input_path,
        output_path=config.image.output_path,
        context={
            "app": config.runtime.project_name,
            "mode": "image",
            "ui": "direct",
        },
    )


def run_video(config: Any) -> Any:
    from pipelines.video_pipeline import VideoPipeline

    config.video.rebuild.output_path = config.video.output_path
    config.video.rebuild.source_video_path = config.video.input_path

    pipeline = VideoPipeline(config.video)

    return pipeline.run(
        input_path=config.video.input_path,
        output_path=config.video.output_path,
        context={
            "app": config.runtime.project_name,
            "mode": "video",
            "ui": "direct",
        },
    )


if __name__ == "__main__":
    main()
