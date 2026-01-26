"""Test image generation functions.

:author: Shay Hill
:created: 2025-06-16
"""

# pyright: reportPrivateUsage = false

import tempfile
from pathlib import Path

from posterize.main import posterize

TEST_RESOURCES = Path(__file__).parent / "resources"

FULL_COLOR = TEST_RESOURCES / "full_color.webp"

class TestImageGeneration:
    def test_write_svg(self):
        """Test that the write_svg method works correctly."""
        posterized_image = posterize(FULL_COLOR, 6)
        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as f:
            result_path = Path(f.name)
        expect_path = TEST_RESOURCES / "expect_draw_approximation.svg"
        try:
            posterized_image.write_svg(result_path)
            with result_path.open("r") as result_file:
                result = result_file.read()
            with expect_path.open("r") as expect_file:
                expected = expect_file.read()
            assert result == expected
        finally:
            result_path.unlink(missing_ok=True)
