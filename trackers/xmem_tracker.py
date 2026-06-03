# trackers/xmem_tracker.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import importlib
import numpy as np


_MODULE_CACHE: dict[str, Any] = {}


def lazy_import(module_name: str) -> Any:
    if module_name not in _MODULE_CACHE:
        try:
            _MODULE_CACHE[module_name] = importlib.import_module(module_name)
        except ImportError as exc:
            raise ImportError(
                f"Missing optional dependency '{module_name}'. "
                f"Install it before using XMemTracker."
            ) from exc

    return _MODULE_CACHE[module_name]


@dataclass(slots=True)
class XMemTrackerConfig:
    enabled: bool = True

    checkpoint_path: str | None = None
    device: str = "auto"

    # Output mask threshold if model returns probabilities.
    threshold: float = 0.5

    # XMem implementations differ between repos.
    # This lets you inject your own model builder later.
    model_factory: Callable[["XMemTrackerConfig"], Any] | None = None


class XMemTracker:
    """
    XMem video object segmentation tracker adapter.

    Purpose:
        Track watermark masks across video frames using an external XMem model.

    Important:
        This file is an adapter/wrapper.
        The actual XMem implementation depends on which repo/package you use.

    Expected model methods:
        Option 1:
            model.initialize(frame, mask)
            model.track(frame)

        Option 2:
            model.set_reference(frame, mask)
            model.step(frame)

        Option 3:
            model(frame)

    Output:
        binary mask as uint8, values 0 or 255
    """

    name = "xmem"

    def __init__(
        self,
        config: XMemTrackerConfig | None = None,
        model: Any | None = None,
    ) -> None:
        self.config = config or XMemTrackerConfig()
        self.model = model

        self.initialized = False
        self.previous_mask: np.ndarray | None = None

    def initialize(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
    ) -> None:
        if not self.config.enabled:
            self.previous_mask = self._to_mask_uint8(mask)
            self.initialized = True
            return

        self._validate_frame(frame)

        mask_uint8 = self._to_mask_uint8(mask)

        if self.model is None:
            self.model = self._build_model()

        if hasattr(self.model, "initialize"):
            self.model.initialize(frame, mask_uint8)

        elif hasattr(self.model, "set_reference"):
            self.model.set_reference(frame, mask_uint8)

        else:
            # Some implementations do not need explicit initialization.
            # We still store the first mask as reference.
            pass

        self.previous_mask = mask_uint8
        self.initialized = True

    def track(
        self,
        frame: np.ndarray,
    ) -> np.ndarray:
        if not self.initialized:
            raise RuntimeError(
                "XMemTracker must be initialized with first frame and mask "
                "before calling track()."
            )

        if not self.config.enabled:
            if self.previous_mask is None:
                raise RuntimeError("No previous mask available.")
            return self.previous_mask

        self._validate_frame(frame)

        if self.model is None:
            raise RuntimeError("XMem model is not loaded.")

        output = self._run_model(frame)
        mask = self._postprocess_output(output)

        self.previous_mask = mask

        return mask

    def track_sequence(
        self,
        frames: list[np.ndarray],
        initial_mask: np.ndarray,
    ) -> list[np.ndarray]:
        if not frames:
            return []

        self.initialize(frames[0], initial_mask)

        masks = [self._to_mask_uint8(initial_mask)]

        for frame in frames[1:]:
            masks.append(self.track(frame))

        return masks

    def _build_model(self) -> Any:
        if self.config.model_factory is not None:
            return self.config.model_factory(self.config)

        raise NotImplementedError(
            "No XMem model_factory was provided. "
            "Pass an initialized model to XMemTracker(model=...) or provide "
            "XMemTrackerConfig(model_factory=...)."
        )

    def _run_model(self, frame: np.ndarray) -> Any:
        if hasattr(self.model, "track"):
            return self.model.track(frame)

        if hasattr(self.model, "step"):
            return self.model.step(frame)

        if callable(self.model):
            return self.model(frame)

        raise TypeError(
            "Unsupported XMem model interface. Expected one of: "
            "track(frame), step(frame), or callable model(frame)."
        )

    def _postprocess_output(self, output: Any) -> np.ndarray:
        """
        Convert model output to binary uint8 mask.
        """
        torch = None

        try:
            torch = lazy_import("torch")
        except ImportError:
            pass

        if torch is not None and hasattr(torch, "Tensor") and isinstance(output, torch.Tensor):
            output = output.detach().cpu().numpy()

        if isinstance(output, dict):
            for key in ("mask", "masks", "prob", "probability", "logits"):
                if key in output:
                    output = output[key]
                    break

        output = np.asarray(output)

        while output.ndim > 2:
            output = output[0]

        if output.dtype == np.bool_:
            return output.astype(np.uint8) * 255

        if output.max() <= 1.0:
            mask = output >= self.config.threshold
            return mask.astype(np.uint8) * 255

        mask = output > 127
        return mask.astype(np.uint8) * 255

    @staticmethod
    def _to_mask_uint8(mask: np.ndarray) -> np.ndarray:
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask[..., 0]

        if mask.ndim != 2:
            raise ValueError(f"Mask must be HxW or HxWx1, got {mask.shape}")

        if mask.dtype == np.bool_:
            return mask.astype(np.uint8) * 255

        if mask.dtype == np.uint8:
            if mask.max() <= 1:
                return mask * 255
            return mask

        if mask.max() <= 1.0:
            return (mask > 0.5).astype(np.uint8) * 255

        return (mask > 127).astype(np.uint8) * 255

    @staticmethod
    def _validate_frame(frame: np.ndarray) -> None:
        if frame is None:
            raise ValueError("frame is None")

        if not isinstance(frame, np.ndarray):
            raise TypeError(f"frame must be np.ndarray, got {type(frame)}")

        if frame.ndim not in (2, 3):
            raise ValueError(f"Unsupported frame shape: {frame.shape}")