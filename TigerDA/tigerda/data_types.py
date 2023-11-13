"""Data Augmentation Engine."""

from abc import ABC, abstractmethod
from typing import Any


class BaseDataAugmentationEngine(ABC):
    """Base data augmentation engine."""

    @abstractmethod
    def augment(self) -> None:
        """Handles a range of tasks."""
