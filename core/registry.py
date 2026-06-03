# core/registry.py

from __future__ import annotations

from typing import Any, Callable


class Registry:
    """
    Small class registry.

    Used for:
        preprocessors
        detectors
        trackers
        inpainters
        postprocessors
        pipelines
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._items: dict[str, Any] = {}

    def register(
        self,
        key: str,
        item: Any | None = None,
    ) -> Callable[[Any], Any] | Any:
        """
        Usage 1:

            registry.register("yolo", YOLODetector)

        Usage 2:

            @registry.register("yolo")
            class YOLODetector:
                ...
        """

        if item is not None:
            self._items[key] = item
            return item

        def decorator(obj: Any) -> Any:
            self._items[key] = obj
            return obj

        return decorator

    def get(self, key: str) -> Any:
        if key not in self._items:
            available = ", ".join(sorted(self._items.keys())) or "none"
            raise KeyError(
                f"'{key}' is not registered in {self.name}. "
                f"Available: {available}"
            )

        return self._items[key]

    def build(self, key: str, *args: Any, **kwargs: Any) -> Any:
        item = self.get(key)
        return item(*args, **kwargs)

    def has(self, key: str) -> bool:
        return key in self._items

    def keys(self) -> list[str]:
        return sorted(self._items.keys())

    def items(self) -> dict[str, Any]:
        return dict(self._items)


PREPROCESSORS = Registry("preprocessors")
DETECTORS = Registry("detectors")
TRACKERS = Registry("trackers")
INPAINTERS = Registry("inpainters")
POSTPROCESSORS = Registry("postprocessors")
PIPELINES = Registry("pipelines")