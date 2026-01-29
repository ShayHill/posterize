"""Test code in quantization module that isn't run in the safe path through posterize.

:author: Shay Hill
:created: 2025-06-16
"""

# pyright: reportPrivateUsage = false

from posterize.quantization import _min_max_normalize, TargetImage, new_target_image
import numpy as np
from pathlib import Path

TEST_RESOURCES = Path(__file__).parent / "resources"

FULL_COLOR = TEST_RESOURCES / "full_color.webp"


class TestQuantization:
    def test_min_max_all_equal(self):
        """Return an array of zeros when all values are equal."""
        array = np.array([5, 5, 5])
        result = _min_max_normalize(array)
        assert result.tolist() == [0.0, 0.0, 0.0]

    def test_reset_weights(self):
        """Reset target image weights to default."""
        target_image = new_target_image(FULL_COLOR)
        old_weights = target_image.weights
        new_weights = np.ones_like(old_weights)
        target_image.weights = new_weights
        assert np.array_equal(target_image.weights, new_weights)
        target_image.reset_weights()
        assert np.array_equal(target_image.weights, old_weights)
