# pipelines/base_pipeline.py

from abc import ABC, abstractmethod


class BasePipeline(ABC):

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def run(self, *args, **kwargs):
        """
        Execute pipeline.
        """
        pass