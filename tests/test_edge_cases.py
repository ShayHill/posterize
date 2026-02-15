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

from posterize.main import cache as posterize_cache
from posterize.main import posterize, posterize_mono
from posterize.quantization import cache as quantization_cache
from posterize.quantization import quantize_image

_TESTS_DIR = Path(__file__).resolve().parent
_GOLDEN_DIR = _TESTS_DIR / "golden"
_TMP_DIR = _TESTS_DIR / "_tmp"
_CHAUCER_PNG = _TESTS_DIR / "chaucer2.webp"
NUM_COLS = 9


def test_zero_cols() -> None:
    """posterize returns an empty posterization if cols is zero."""
    posterized = posterize(str(_CHAUCER_PNG), 0)
    assert len(posterized.layers) == 0


def test_empty_cols() -> None:
    """posterize returns an empty posterization if cols is empty."""
    posterized = posterize(str(_CHAUCER_PNG), [])
    assert len(posterized.layers) == 0
