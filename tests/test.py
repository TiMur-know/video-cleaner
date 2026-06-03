# tests/test.py

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from tests.module_checker import run_module_checks
from tests.package_checker import run_package_checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="watermwark-tests",
        description="Run watermwark project checks.",
    )

    parser.add_argument(
        "--packages-only",
        action="store_true",
        help="Only check installed packages.",
    )

    parser.add_argument(
        "--modules-only",
        action="store_true",
        help="Only check project modules.",
    )

    parser.add_argument(
        "--heavy",
        action="store_true",
        help="Run optional heavier module smoke tests.",
    )

    parser.add_argument(
        "--no-optional-packages",
        action="store_true",
        help="Skip optional model package checks.",
    )

    parser.add_argument(
        "--require-rich",
        action="store_true",
        help="Treat rich as required.",
    )

    parser.add_argument(
        "--require-gradio",
        action="store_true",
        help="Treat gradio as required.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show verbose output.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    exit_code = 0

    if not args.modules_only:
        package_exit_code = run_package_checks(
            include_optional=not args.no_optional_packages,
            verbose=args.verbose,
            rich_required=args.require_rich,
            gradio_required=args.require_gradio,
        )

        exit_code = max(exit_code, package_exit_code)

    if not args.packages_only:
        module_exit_code = run_module_checks(
            include_heavy=args.heavy,
            verbose=args.verbose,
        )

        exit_code = max(exit_code, module_exit_code)

    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()