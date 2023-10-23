"""Test finetuning engine."""
import pkgutil
import pytest


def test_torch_imports() -> None:
    """Test that torch is an optional dependency."""
    # importing fine-tuning modules should be ok
    from tigertune.finetuning import TextGenerationTransformer  # noqa: F401
