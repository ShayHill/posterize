"""Test TargetImage class.

:author: Shay Hill
:created: 2024-05-09
"""

# pyright: reportPrivateUsage = false

from posterize.paths import PROJECT
from lxml.etree import Element
from lxml.etree import _Element as EtreeElement  # type: ignore

from posterize.type_target_image import TargetImage
import pytest

TEST_IMAGE = PROJECT / "tests/resources/lion.jpg"


class TestTargetImage:
    """Test TargetImage class."""

    def test_init_path(self) -> None:
        """Test that the path is initialized correctly."""
        target = TargetImage(TEST_IMAGE)
        assert target.path == TEST_IMAGE

    def test_init_elements(self) -> None:
        """Test that the elements are initialized correctly."""
        target = TargetImage(TEST_IMAGE)
        assert target.elements == []

    def test_init_image_grid(self) -> None:
        """Test that the image grid is initialized correctly."""
        target = TargetImage(TEST_IMAGE)
        assert target._image_grid.shape == (1024, 1024, 3)

    def test_init_state_grid(self) -> None:
        """Test that the state grid is initialized correctly."""
        target = TargetImage(TEST_IMAGE)
        assert target._state_grid is None

    def test_init_error_grid(self) -> None:
        """Test that the error grid is initialized correctly."""
        target = TargetImage(TEST_IMAGE)
        assert target._error_grid is None

    def test_state_grid(self) -> None:
        """Test that the state grid is initialized correctly."""
        target = TargetImage(TEST_IMAGE)
        with pytest.raises(ValueError):
            _ = target.state_grid
        target.append(Element("g"))
        assert target.state_grid.shape == (1024, 1024, 3)

    def test_error_grid(self) -> None:
        """Test that the state grid is initialized correctly."""
        target = TargetImage(TEST_IMAGE)
        with pytest.raises(ValueError):
            _ = target.error_grid
        target.append(Element("g"))
        assert target.error_grid.shape == (1024, 1024)

    # def test_error_grid(self) -> None:
    #     """Test that the error grid is initialized correctly."""
    #     target = TargetImage(TEST_IMAGE)
    #     assert target.error_grid.shape == (1024, 1024)
