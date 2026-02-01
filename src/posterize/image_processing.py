"""Write images to disk, convert arrays to images, call potrace.

:author: Shay Hill
:created: 2024-10-13
"""

import importlib.resources
import os
import subprocess
from pathlib import Path
from typing import Annotated, Any, TypeAlias

import numpy as np
from lxml import etree
from numpy import typing as npt
from PIL import Image
from svg_path_data import format_svgd_shortest

from posterize.paths import CACHE_DIR

POTRACE_EXE = importlib.resources.files("posterize.bin") / "potrace.exe"


_IndexMatrix: TypeAlias = Annotated[npt.NDArray[np.integer[Any]], "(r,c)"]

_TMP_BMP = CACHE_DIR / "temp.bmp"


def _write_svg_from_mono_bmp(path_to_mono_bmp: str | os.PathLike[str]) -> Path:
    """Write an svg from a monochrome image.

    :param path_to_mono_bmp: path to the monochrome image
    :return: path to the output svg

    Images passed in this library will be not only grayscale, but monochrome. So, a
    blacklevel of 0.5 can be hardcoded.
    """
    svg_path = (CACHE_DIR / Path(path_to_mono_bmp).name).with_suffix(".svg")
    # fmt: off
    command = [
        str(POTRACE_EXE),
        str(path_to_mono_bmp),
        "-o", str(svg_path),  # output file
        "-k", str(0.5),  # black level
        "-u", "1",  # do not scale svg (points correspond to pixels array)
        "--flat",  # all paths combined in one element
        "-b", "svg",  # output format
        "--opttolerance", "2.8",  # higher values make paths smoother
    ]
    # fmt: on
    _ = subprocess.run(command, check=True)
    return svg_path


def _write_mono_bmp(layer: _IndexMatrix) -> Path:
    """Write a monochrome bitmap from an index matrix.

    :param layer: (r, c) array where -1 is transparent and opaque pixels are all
        filled with the same index colormap.
    :return: path to the monochrome bitmap

    This bmp will be the input argument for potrace. Black pixels will be inside the
    output path element.
    """
    mono_pixels = np.ones([*layer.shape, 3], dtype=np.uint8) * 255
    mono_pixels[np.where(layer != -1)] = (0, 0, 0)
    mono_bmp = Image.fromarray(mono_pixels)
    output_path = _TMP_BMP
    mono_bmp.save(output_path)
    return output_path


def layer_to_svgd(layer: _IndexMatrix) -> str:
    """Convert a layer to an SVG data string.

    :param layer: (r, c) array where -1 is transparent and opaque pixels are all
        filled with the same index colormap.
    :return: SVG data string

    If the layer has no -1 values (solid layer), returns a rectangle path covering
    the entire layer. Otherwise, writes a monochrome bitmap from the layer, converts
    it to SVG using potrace, and returns the SVG content as a string.
    """
    if np.all(layer != -1):
        height, width = layer.shape
        return format_svgd_shortest(f"M0 0 {width} 0 {width} {height} 0 {height}z")
    mono_bmp = _write_mono_bmp(layer)
    svg_path: Path | None = None
    try:
        svg_path = _write_svg_from_mono_bmp(mono_bmp)
        return format_svgd_shortest(
            etree.parse(str(svg_path)).getroot()[1][0].attrib["d"]
        )
    finally:
        mono_bmp.unlink()
        if svg_path is not None:
            svg_path.unlink()
