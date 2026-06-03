# controls/io_control.py

from __future__ import annotations

from argparse import Namespace

from core.config import AppConfig, infer_mode_from_path


def apply_input_output_overrides(config: AppConfig, args: Namespace) -> None:
    if args.type is not None:
        config.runtime.mode = args.type

    if args.input is not None:
        mode = args.type or config.runtime.mode

        match mode:
            case "auto":
                mode = infer_mode_from_path(args.input)

            case "image" | "video":
                pass

            case _:
                raise ValueError(f"Unsupported input mode: {mode}")

        set_input_path_for_mode(
            config=config,
            mode=mode,
            input_path=args.input,
        )

    if args.output is not None:
        mode = args.type or config.runtime.mode

        match mode:
            case "auto" if args.input is not None:
                mode = infer_mode_from_path(args.input)

            case "auto":
                # Cannot safely infer output target without an input.
                return

            case "image" | "video":
                pass

            case _:
                raise ValueError(f"Unsupported output mode: {mode}")

        set_output_path_for_mode(
            config=config,
            mode=mode,
            output_path=args.output,
        )


def resolve_run_mode(config: AppConfig, args: Namespace) -> str:
    mode = args.type or config.runtime.mode

    match mode:
        case "auto":
            if args.input is not None:
                return infer_mode_from_path(args.input)

            return infer_mode_from_path(config.image.input_path)

        case "image" | "video":
            return mode

        case _:
            raise ValueError(f"Unsupported run mode: {mode}")


def set_input_path_for_mode(
    config: AppConfig,
    mode: str,
    input_path: str,
) -> None:
    match mode:
        case "image":
            config.image.input_path = input_path

        case "video":
            config.video.input_path = input_path

        case _:
            raise ValueError(f"Unsupported input mode: {mode}")


def set_output_path_for_mode(
    config: AppConfig,
    mode: str,
    output_path: str,
) -> None:
    match mode:
        case "image":
            config.image.output_path = output_path

        case "video":
            config.video.output_path = output_path

        case _:
            raise ValueError(f"Unsupported output mode: {mode}")