# inpainters/base_inpainter.py

from abc import ABC, abstractmethod

import numpy as np


class BaseInpainter(ABC):

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def inpaint(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """
        Remove watermark using mask.

        Args:
            image:
                RGB image

            mask:
                Binary mask

        Returns:
            Clean image
        """
        pass