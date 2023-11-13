"""Test Data Augmentation engine."""
import pkgutil
import pytest


def test_torch_imports() -> None:
    """Test that torch is an optional dependency."""
    # importing fine-tuning modules should be ok
    from tigerda.augmenters.text_generation_augmenter import TextGenerationDataAugmentationEngine  # noqa: F401
    from tigerda.augmenters.text_generation_augmenter import TextGenerationDataAugmentationEngine  # noqa: F401
