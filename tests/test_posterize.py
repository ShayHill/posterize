"""Run a Golden Master test for posterize.

:author: Shay Hill
:created: 2025-06-16
"""

# pyright: reportPrivateUsage = false

from pathlib import Path

import numpy as np
import pytest
from posterize.paths import CACHE_DIR

from posterize.main import ImageApproximation, _set_max_val_to_one, posterize
from posterize.posterization import Posterization
from posterize.quantization import new_target_image, clear_all_quantized_image_caches, _CACHE_PREFIX

TEST_RESOURCES = Path(__file__).parent / "resources"

FULL_COLOR = TEST_RESOURCES / "full_color.webp"
MONOCHROME = TEST_RESOURCES / "monochrome.png"

FULL_COLOR_6_LAYERS = TEST_RESOURCES / "expect_full_color_6_layers.npy"
MONOCHROME_6_LAYERS = TEST_RESOURCES / "expect_monochrome_6_layers.npy"

# Set to True to ignore the cache and always recompute quantized images. Nothing else
# will be cached, so False is usually safe. The image-quantization code is stable.
IGNORE_QUANTIZED_IMAGE_CACHE = False


@pytest.fixture(scope="module")
def rgb_posterized() -> Posterization:
    """Fixture for the full-color image."""
    return posterize(
        FULL_COLOR, 6, ignore_quantized_image_cache=IGNORE_QUANTIZED_IMAGE_CACHE
    )


@pytest.fixture(scope="module")
def mono_posterized() -> Posterization:
    """Fixture for the monochrome image."""
    return posterize(
        MONOCHROME, 6, ignore_quantized_image_cache=IGNORE_QUANTIZED_IMAGE_CACHE
    )

def test_clear_all_quantized_image_caches():
    """The best assertion that this has run is that the tests take a long time."""
    cache = CACHE_DIR.glob(f"{_CACHE_PREFIX}*.npy")
    clear_all_quantized_image_caches()
    cache = CACHE_DIR.glob(f"{_CACHE_PREFIX}*.npy")
    assert not any(cache), "Cache should be empty after clearing."


class TestPosterize:
    """Test the posterize function."""

    def test_rgb_colors(self, rgb_posterized: Posterization):
        """Posterizing a full-color image results in 512 colors."""
        assert len(rgb_posterized.palette) == 512

    def test_rgb_layers(self, rgb_posterized: Posterization):
        """Golden Master test layers for full-color image."""
        expect = np.load(FULL_COLOR_6_LAYERS)
        actual = np.array(rgb_posterized.layers, dtype=np.intp)
        np.testing.assert_array_equal(actual, expect)

    def test_mono_colors(self, mono_posterized: Posterization):
        """Posterizing a monochrome image results in <= 256 colors."""
        assert len(mono_posterized.palette) <= 256

    def test_mono_layers(self, mono_posterized: Posterization):
        """Golden Master test layers for monochrome image."""
        expect = np.load(MONOCHROME_6_LAYERS)
        actual = np.array(mono_posterized.layers, dtype=np.intp)
        np.testing.assert_array_equal(actual, expect)


class TestSetMaxValToOne:
    """Test that the max value of the colors is set to 1."""

    def test_all_zeros(self):
        """Test that all zeros are returned as all zeros."""
        result = tuple(_set_max_val_to_one([0, 0, 0]))
        assert result == (0.0, 0.0, 0.0)


class TestExhaustColors:
    """Test that layer-gen methods do not raise an error when colors exhausted."""

    def test_fill_layers(self):
        """Asking for more than available colors will not raise an error."""
        target_image = new_target_image(FULL_COLOR)
        # init with only three colors
        image_approx = ImageApproximation(target_image, colors=(1, 3, 5))
        # asking for more than available colors
        image_approx.fill_layers(4)
        assert set(image_approx.layer_colors) == {1, 3, 5}

    def test_two_pass_fill_layers(self):
        """Asking for more than available colors will not raise an error."""
        target_image = new_target_image(FULL_COLOR)
        # init with only three colors
        image_approx = ImageApproximation(target_image, colors=(1, 3, 5))
        # asking for more than available colors
        image_approx.two_pass_fill_layers(4)
        assert set(image_approx.layer_colors) == {1, 3, 5}


class TestImageApproximation:
    """Test ImageApproximation code that isn't run by posterize."""

    def test_pass_layers(self, rgb_posterized: Posterization):
        """Pass previously-defined layers to ImageApproximation.init."""
        prev_run = posterize(
            FULL_COLOR, 6, ignore_quantized_image_cache=IGNORE_QUANTIZED_IMAGE_CACHE
        )
        target_image = new_target_image(FULL_COLOR)
        layers_array = np.array(prev_run.layers, dtype=np.intp)
        image_approx = ImageApproximation(target_image, layers=layers_array)
        expect = np.load(FULL_COLOR_6_LAYERS)
        np.testing.assert_array_equal(image_approx.layers, expect)

    def test_do_not_overfill_layers(
        self, rgb_posterized: Posterization
    ):
        """Do not overfill layers when passing layers to ImageApproximation."""
        target_image = new_target_image(FULL_COLOR)
        image_approx = ImageApproximation(target_image)
        image_approx.fill_layers(6)
        assert len(image_approx.layers) == 6
        image_approx.fill_layers(3)
        assert len(image_approx.layers) == 6
