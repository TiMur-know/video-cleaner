# trackers/base_tracker.py

from abc import ABC, abstractmethod
from typing import Dict, Any

import numpy as np


class BaseTracker(ABC):

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def initialize(
        self,
        frame: np.ndarray,
        detection: Dict[str, Any]
    ):
        """
        Initialize tracker using first detection.
        """
        pass

    @abstractmethod
    def track(
        self,
        frame: np.ndarray
    ) -> Dict[str, Any]:
        """
        Track watermark in next frame.

        Returns:
            {
                "mask": np.ndarray,
                "bbox": list,
                "confidence": float,
                "motion_vector": tuple
            }
        """
        pass