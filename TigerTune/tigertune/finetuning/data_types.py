"""Finetuning Engine."""

from abc import ABC, abstractmethod
from typing import Any


class BaseLLMFinetuneEngine(ABC):
    """Base LLM finetuning engine."""

    @abstractmethod
    def finetune(self) -> None:
        """Handles a range of tasks."""
