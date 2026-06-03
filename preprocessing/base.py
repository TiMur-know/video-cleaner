# preprocessing/base.py

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BasePreprocessor(ABC):
    """
    Base class for all preprocessing modules.

    Examples:
    - CLAHE
    - gamma correction
    - resize
    - denoise
    - edge enhancement
    """

    name: str

    @abstractmethod
    def apply(self, image: np.ndarray, context: dict[str, Any] | None = None) -> np.ndarray:
        pass

    def __call__(
        self,
        image: np.ndarray,
        context: dict[str, Any] | None = None,
    ) -> np.ndarray:
        return self.apply(image, context=context)