"""Golden-master tests for posterize and posterize_mono SVG output."""

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


@pytest.fixture(autouse=True)
def _clear_caches() -> None:  # pyright: ignore[reportUnusedFunction]
    """Clear posterize and quantization caches before each golden test."""
    _ = posterize_cache.clear()
    _ = quantization_cache.clear()


def _start_got_and_golden(got_path: Path, golden_path: Path) -> None:
    """On failure, open both files with the default app (e.g. browser)."""
    _ = subprocess.run(["cmd", "/c", "start", "", str(got_path)], check=False)
    _ = subprocess.run(["cmd", "/c", "start", "", str(golden_path)], check=False)


def test_quantize_image() -> None:
    """Quantize image and write pixels to PNG."""
    quantized = quantize_image(str(_CHAUCER_PNG))
    pixels = quantized.palette[quantized.indices]
    out = "chaucer_quantized.png"
    Image.fromarray(pixels, mode="RGB").save(out)


def test_posterize_cols_list() -> None:
    """posterize accepts a list of palette indices for cols."""
    cols = list(range(NUM_COLS))
    posterized = posterize(str(_CHAUCER_PNG), cols)
    assert len(posterized.layers) <= NUM_COLS
    _TMP_DIR.mkdir(exist_ok=True)
    out = _TMP_DIR / "chaucer_posterized_cols_list.svg"
    _ = posterized.write_svg(out)
    got = out.read_text()
    golden_path = _GOLDEN_DIR / "chaucer_posterized_cols_list.svg"
    golden = golden_path.read_text()
    try:
        assert got == golden
    except AssertionError:
        _start_got_and_golden(out, golden_path)
        raise
    assert out.exists()
    assert out.read_text()


def test_posterize_golden() -> None:
    """Generated SVG matches golden chaucer_posterized.svg."""
    beg = time.time()
    posterized = posterize(str(_CHAUCER_PNG), NUM_COLS)
    end = time.time()
    print(f"Posterized color in {end - beg} seconds")  # noqa: T201
    _TMP_DIR.mkdir(exist_ok=True)
    out = _TMP_DIR / "chaucer_posterized.svg"
    _ = posterized.write_svg(out)
    got = out.read_text()
    golden_path = _GOLDEN_DIR / "chaucer_posterized.svg"
    golden = golden_path.read_text()
    try:
        assert got == golden
    except AssertionError:
        _start_got_and_golden(out, golden_path)
        raise


def test_posterize_mono_golden() -> None:
    """Generated SVG matches golden chaucer_posterized_mono.svg."""

    image = Image.open(_CHAUCER_PNG)
    mono = np.array(image)[:, :, 0]
    beg = time.time()
    posterized = posterize_mono(mono, NUM_COLS)
    end = time.time()
    print(f"Posterized mono in {end - beg} seconds")  # noqa: T201
    _TMP_DIR.mkdir(exist_ok=True)
    out = _TMP_DIR / "chaucer_posterized_mono.svg"
    _ = posterized.write_svg(out)
    got = out.read_text()
    golden_path = _GOLDEN_DIR / "chaucer_posterized_mono.svg"
    golden = golden_path.read_text()
    try:
        assert got == golden
    except AssertionError:
        _start_got_and_golden(out, golden_path)
        raise
