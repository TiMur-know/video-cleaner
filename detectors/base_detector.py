# detectors/base_detector.py

from abc import ABC, abstractmethod
from typing import Dict, Any

import numpy as np


class BaseDetector(ABC):

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def detect(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Detect watermark regions.

        Args:
            image:
                RGB image as numpy array

        Returns:
            {
                "mask": np.ndarray,
                "bboxes": list,
                "confidence": float,
                "metadata": dict
            }
        """
        pass