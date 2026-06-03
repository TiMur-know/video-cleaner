# controls/control_utils.py

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

from utils.env_loader import env_str


class BaseDetectorStage:
    """
    Base class for detector stage definitions.

    Used by:
        ProposalDetectorStage
        RefinerDetectorStage
    """

    stage_name: ClassVar[str] = "base"
    choices: ClassVar[tuple[str, ...]] = ()
    env_name: ClassVar[str] = ""

    @classmethod
    def normalize(
        cls,
        detectors: Sequence[str] | None,
    ) -> list[str]:
        if detectors is None:
            return []

        selected: list[str] = []

        for detector in detectors:
            detector = str(detector).strip()

            if not detector:
                continue

            if detector not in cls.choices:
                raise ValueError(
                    f"Unsupported {cls.stage_name} detector: {detector}"
                )

            if detector not in selected:
                selected.append(detector)

        return selected

    @classmethod
    def from_env(cls) -> list[str] | None:
        value = env_str(cls.env_name, None)

        if value is None:
            return None

        parts = parse_env_list(value)

        return cls.normalize(parts)

    @classmethod
    def contains(cls, detector: str) -> bool:
        return detector in cls.choices


class ProposalDetectorStage(BaseDetectorStage):
    stage_name: ClassVar[str] = "proposal"
    env_name: ClassVar[str] = "WATERMWARK_PROPOSAL_DETECTORS"

    choices: ClassVar[tuple[str, ...]] = (
        "opencv",
        "grounding_dino",
        "yolo",
        "fft",
        "anomaly",
        "paddle_ocr",
        "easy_ocr",
    )


class RefinerDetectorStage(BaseDetectorStage):
    stage_name: ClassVar[str] = "refiner"
    env_name: ClassVar[str] = "WATERMWARK_REFINER_DETECTORS"

    choices: ClassVar[tuple[str, ...]] = (
        "sam2",
        "mobile_sam_2",
    )


PROPOSAL_DETECTOR_CHOICES: tuple[str, ...] = ProposalDetectorStage.choices
REFINER_DETECTOR_CHOICES: tuple[str, ...] = RefinerDetectorStage.choices

DETECTOR_CHOICES: tuple[str, ...] = (
    *PROPOSAL_DETECTOR_CHOICES,
    *REFINER_DETECTOR_CHOICES,
)


def combine_detector_stages(
    proposal_detectors: list[str] | None,
    refiner_detectors: list[str] | None,
) -> list[str]:
    """
    Combine proposal and refiner detector lists while preserving order
    and removing duplicates.
    """
    return unique_preserve_order(
        [
            *(proposal_detectors or []),
            *(refiner_detectors or []),
        ]
    )


def split_detector_stages(
    detectors: list[str] | tuple[str, ...] | None,
) -> tuple[list[str], list[str]]:
    """
    Split old combined detector list into proposal/refiner stages.

    Example:
        ["opencv", "fft", "sam2"]

    Returns:
        (["opencv", "fft"], ["sam2"])
    """
    proposal_detectors: list[str] = []
    refiner_detectors: list[str] = []

    for detector in detectors or []:
        detector = str(detector).strip()

        if not detector:
            continue

        if detector in PROPOSAL_DETECTOR_CHOICES:
            if detector not in proposal_detectors:
                proposal_detectors.append(detector)

        elif detector in REFINER_DETECTOR_CHOICES:
            if detector not in refiner_detectors:
                refiner_detectors.append(detector)

        else:
            raise ValueError(f"Unsupported detector: {detector}")

    return proposal_detectors, refiner_detectors


def normalize_detectors(
    detectors: Sequence[str] | None,
) -> list[str]:
    """
    Normalize any detector list against all known detector choices.
    """
    if detectors is None:
        return []

    selected: list[str] = []

    for detector in detectors:
        detector = str(detector).strip()

        if not detector:
            continue

        if detector not in DETECTOR_CHOICES:
            raise ValueError(f"Unsupported detector: {detector}")

        if detector not in selected:
            selected.append(detector)

    return selected


def env_detectors() -> list[str] | None:
    """
    Backwards-compatible env detector list.

    Supports:
        WATERMWARK_DETECTORS=opencv,fft,anomaly,sam2
        WATERMWARK_DETECTOR=opencv

    Prefer:
        WATERMWARK_PROPOSAL_DETECTORS=opencv,fft,anomaly
        WATERMWARK_REFINER_DETECTORS=sam2
    """
    value = env_str("WATERMWARK_DETECTORS", None)

    if value is None:
        value = env_str("WATERMWARK_DETECTOR", None)

    if value is None:
        return None

    parts = parse_env_list(value)

    return normalize_detectors(parts)


def resolve_detector_mode_summary(
    proposal_detectors: list[str],
    refiner_detectors: list[str],
) -> str:
    proposal_count = len(proposal_detectors)
    refiner_count = len(refiner_detectors)

    if proposal_count == 0 and refiner_count == 0:
        return "disabled"

    if proposal_count > 0 and refiner_count > 0:
        return "proposal_refiner"

    if proposal_count > 0:
        return "proposal_only"

    return "refiner_only"


def parse_env_list(value: str | None) -> list[str]:
    """
    Parse comma/space-separated env values.

    Examples:
        "opencv,fft,sam2"
        "opencv fft sam2"
        ""

    Returns:
        list[str]
    """
    if value is None:
        return []

    value = value.strip()

    if value == "":
        return []

    return [
        item.strip()
        for item in value.replace(",", " ").split()
        if item.strip()
    ]


def env_list(
    name: str,
    default: list[str] | None = None,
) -> list[str] | None:
    """
    Read a comma/space-separated env var as a list.
    """
    value = env_str(name, None)

    if value is None:
        return default

    return parse_env_list(value)


def unique_preserve_order(
    values: Sequence[str],
) -> list[str]:
    """
    Remove duplicates while preserving order.
    """
    selected: list[str] = []

    for value in values:
        value = str(value).strip()

        if not value:
            continue

        if value not in selected:
            selected.append(value)

    return selected


def set_if_exists(
    obj: Any,
    name: str,
    value: Any,
) -> None:
    """
    Set an attribute only if it exists.

    This keeps tuning safe across detectors/config classes
    with different fields.
    """
    if hasattr(obj, name):
        setattr(obj, name, value)


def set_nested_attr(
    parent: Any,
    child_name: str,
    attr_name: str,
    value: Any,
) -> None:
    """
    Set parent.child.attr only if parent.child exists and attr exists.
    """
    child = getattr(parent, child_name, None)

    if child is not None:
        set_if_exists(child, attr_name, value)


def get_nested_attr(
    parent: Any,
    child_name: str,
    attr_name: str,
    default: Any = None,
) -> Any:
    """
    Read parent.child.attr safely.
    """
    child = getattr(parent, child_name, None)

    if child is None:
        return default

    return getattr(child, attr_name, default)


def clamp_float(
    value: int | float,
    minimum: float,
    maximum: float,
) -> float:
    return max(minimum, min(float(value), maximum))


def env_float(
    name: str,
    default: float,
) -> float:
    value = env_str(name, None)

    if value is None:
        return default

    try:
        return float(value)
    except ValueError:
        return default


def env_int(
    name: str,
    default: int,
) -> int:
    value = env_str(name, None)

    if value is None:
        return default

    try:
        return int(value)
    except ValueError:
        return default