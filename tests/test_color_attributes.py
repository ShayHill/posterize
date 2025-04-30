"""Test functions in the color_attributes module.

:author: Shay Hill
:created: 2025-04-30
"""

from posterize import color_attributes
import pytest
from numpy import typing as npt
import numpy as np
import random


@pytest.fixture
def random_chromatic_color() -> npt.NDArray[np.uint8]:
    """Return a random chromatic color."""
    channels = [0, 255, random.randint(0, 255)]
    random.shuffle(channels)
    return np.array(channels, dtype=np.uint8)


@pytest.fixture
def random_shaded_color() -> npt.NDArray[np.uint8]:
    """Return a random shaded color with no gray."""
    channels = [0, random.randint(0, 254), random.randint(0, 254)]
    random.shuffle(channels)
    return np.array(channels, dtype=np.uint8)


@pytest.fixture
def random_tinted_color() -> npt.NDArray[np.uint8]:
    """Return a random tinted color with no gray."""
    channels = [255, random.randint(1, 255), random.randint(1, 255)]
    random.shuffle(channels)
    return np.array(channels, dtype=np.uint8)


class TestGetChromacity:
    @pytest.mark.parametrize("_", range(100))
    def test_0_and_255(self, random_chromatic_color: npt.NDArray[np.uint8], _) -> None:
        """Return 1 for any color with at least one channel at 0 and one at 255."""
        expect = pytest.approx(1.0)  # pyright: ignore[reportUnknownMemberType]
        assert color_attributes.get_chromacity(random_chromatic_color) == expect

    def test_pure_gray(self) -> None:
        """Return 0 for a pure gray color."""
        gray = np.array([127, 127, 127], dtype=np.uint8)
        expect = pytest.approx(0.0)  # pyright: ignore[reportUnknownMemberType]
        assert color_attributes.get_chromacity(gray) == expect

    @pytest.mark.parametrize("_", range(100))
    def test_shaded(self, random_shaded_color: npt.NDArray[np.uint8], _) -> None:
        """Return some value less than 1 for a shaded color."""
        assert 0 <= color_attributes.get_chromacity(random_shaded_color) < 1

    @pytest.mark.parametrize("_", range(100))
    def test_tinted(self, random_tinted_color: npt.NDArray[np.uint8], _) -> None:
        """Return some value less than 1 for a tinted color."""
        assert 0 <= color_attributes.get_chromacity(random_tinted_color) < 1


class TestGetPurity:
    @pytest.mark.parametrize("_", range(100))
    def test_0_and_255(self, random_chromatic_color: npt.NDArray[np.uint8], _) -> None:
        """Return 1 for any color with at least one channel at 0 and one at 255."""
        expect = pytest.approx(1.0)  # pyright: ignore[reportUnknownMemberType]
        assert color_attributes.get_purity(random_chromatic_color) == expect

    def test_pure_gray(self) -> None:
        """Return 0 for a pure gray color."""
        gray = np.array([127, 127, 127], dtype=np.uint8)
        expect = pytest.approx(0.0)  # pyright: ignore[reportUnknownMemberType]
        assert color_attributes.get_purity(gray) == expect

    @pytest.mark.parametrize("_", range(100))
    def test_shaded(self, random_shaded_color: npt.NDArray[np.uint8], _) -> None:
        """Return 1 for a shaded color with no gray."""
        expect = pytest.approx(1.0)  # pyright: ignore[reportUnknownMemberType]
        assert color_attributes.get_purity(random_shaded_color) == expect

    @pytest.mark.parametrize("_", range(100))
    def test_tinted(self, random_tinted_color: npt.NDArray[np.uint8], _) -> None:
        """Return 1 for a tinted color with no gray."""
        expect = pytest.approx(1.0)  # pyright: ignore[reportUnknownMemberType]
        assert color_attributes.get_purity(random_tinted_color) == expect
