"""Try to break the posterize function.

:author: Shay Hill
:created: 2026-02-15
"""

import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from posterize.main import ImageApproximation, posterize, posterize_mono
from posterize.main import cache as posterize_cache
from posterize.quantization import TargetImage, new_target_image, quantize_image
from posterize.quantization import cache as quantization_cache

_TESTS_DIR = Path(__file__).resolve().parent
_GOLDEN_DIR = _TESTS_DIR / "golden"
_TMP_DIR = _TESTS_DIR / "_tmp"
_CHAUCER_PNG = _TESTS_DIR / "chaucer2.webp"
NUM_COLS = 9


def test_zero_cols() -> None:
    """posterize returns an empty posterization if cols is zero."""
    posterized = posterize(str(_CHAUCER_PNG), 0)
    assert len(posterized.palette) == 512
    assert len(posterized.layers) == 0


def test_empty_cols() -> None:
    """posterize returns an empty posterization if cols is empty."""
    posterized = posterize(str(_CHAUCER_PNG), [])
    assert len(posterized.palette) == 512
    assert len(posterized.layers) == 0


def test_add_one_color() -> None:
    """add_one_color adds one color to the posterization."""
    target = new_target_image(str(_CHAUCER_PNG))
    state = ImageApproximation(target)
    state.add_one_hex_color("#000000")
    assert len(state.layers) == 1
    assert np.all(state.layers[0] == 293)
    state.add_one_hex_color("#ffffff")
    assert len(state.layers) == 2
