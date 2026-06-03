# utils/env_loader.py

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


DEFAULT_ENV_PATH = ".env"


def load_env_file(path: str | Path = DEFAULT_ENV_PATH) -> None:
    """
    Small dependency-free .env loader.

    Rules:
        KEY=value
        # comments supported
        existing environment variables are not overwritten
    """

    env_path = Path(path)

    if not env_path.exists():
        return

    with env_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            if "=" not in line:
                continue

            key, value = line.split("=", 1)

            key = key.strip()
            value = clean_env_value(value)

            if key and key not in os.environ:
                os.environ[key] = value


def clean_env_value(value: str) -> str:
    value = value.strip()

    if len(value) >= 2:
        if value[0] == value[-1] and value[0] in {"'", '"'}:
            return value[1:-1]

    return value


def env_str(
    name: str,
    default: str | None = None,
) -> str | None:
    value = os.getenv(name)

    if value is None:
        return default

    if value == "":
        return default

    return value


def env_int(
    name: str,
    default: int,
) -> int:
    value = env_str(name)

    if value is None:
        return default

    return int(value)


def env_bool(
    name: str,
    default: bool = False,
) -> bool:
    value = env_str(name)

    if value is None:
        return default

    return value.lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }


def env_choice(
    name: str,
    choices: list[str],
    default: str,
) -> str:
    value = env_str(name, default)

    if value not in choices:
        return default

    return value


def apply_env_to_namespace(args: Any) -> Any:
    """
    Optional helper if you want to apply env values after argparse.
    CLI args still win because argparse already resolved them.
    """

    return args