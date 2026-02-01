"""Golden-master tests for posterize and posterize_mono SVG output."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from posterize.main import posterize, posterize_mono

_TESTS_DIR = Path(__file__).resolve().parent
_GOLDEN_DIR = _TESTS_DIR / "golden"
_TMP_DIR = _TESTS_DIR / "_tmp"
_CHAUCER_PNG = _TESTS_DIR / "chaucer.png"
NUM_COLS = 9


@pytest.fixture(autouse=True)
def _clear_caches() -> None:  # pyright: ignore[reportUnusedFunction]
    """Clear posterize and quantization caches before each golden test."""
    # Path(".cache_posterize").unlink(missing_ok=True)
    # Path(".cache_quantize").unlink(missing_ok=True)


def test_posterize_golden() -> None:
    """Generated SVG matches golden chaucer_posterized.svg."""
    posterized = posterize(str(_CHAUCER_PNG), NUM_COLS)
    _TMP_DIR.mkdir(exist_ok=True)
    out = _TMP_DIR / "chaucer_posterized.svg"
    _ = posterized.write_svg(out)
    got = out.read_text()
    golden = (_GOLDEN_DIR / "chaucer_posterized.svg").read_text()
    assert got == golden


def test_posterize_mono_golden() -> None:
    """Generated SVG matches golden chaucer_posterized_mono.svg."""
    image = Image.open(_CHAUCER_PNG)
    mono = np.array(image)[:, :, 0]
    posterized = posterize_mono(mono, NUM_COLS)
    _TMP_DIR.mkdir(exist_ok=True)
    out = _TMP_DIR / "chaucer_posterized_mono.svg"
    _ = posterized.write_svg(out)
    got = out.read_text()
    golden = (_GOLDEN_DIR / "chaucer_posterized_mono.svg").read_text()
    assert got == golden
