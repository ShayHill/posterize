"""irite images to disk, convert arrays to images, call potrace.

:author: Shay Hill
:created: 2024-10-13
"""

import subprocess
from pathlib import Path
from typing import Annotated, TypeAlias, Any
import os

from basic_colormath import float_tuple_to_8bit_int_tuple
import numpy as np
from lxml import etree
from lxml.etree import _Element as EtreeElement  # pyright: ignore[reportPrivateUsage]
from numpy import typing as npt
from PIL import Image
from svg_ultralight import new_element, new_svg_root, update_element, write_svg
from svg_ultralight.strings import svg_color_tuple
from posterize.paths import CACHE_DIR

from posterize import paths

_PixelVector: TypeAlias = Annotated[npt.NDArray[np.floating[Any]], "(r,3)"]
_IndexMatrix: TypeAlias = Annotated[npt.NDArray[np.integer[Any]], "(r,c)"]
_IndexMatrices: TypeAlias = Annotated[npt.NDArray[np.integer[Any]], "(n,r,c)"]

_TMP_BMP = CACHE_DIR / "temp.bmp"


def _write_svg_from_mono_bmp(path_to_mono_bmp: Path) -> Path:
    """Write an svg from a monochrome image.

    :param path_to_mono_bitmap: path to the monochrome image.
    :return: path to the output svg

    Images passed in this library will be not only grayscale, but monochrome. So, a
    blacklevel of 0.5 can be hardcoded.
    """
    svg_path = (paths.CACHE / path_to_mono_bmp.name).with_suffix(".svg")
    # fmt: off
    command = [
        str(paths.POTRACE),
        str(path_to_mono_bmp),
        "-o", str(svg_path),  # output file
        "-k", str(0.5),  # black level
        "-u", "1",  # do not scale svg (points correspond to pixels array)
        "--flat",  # all paths combined in one element
        # "-t", str(self._tsize),  # remove speckles
        "-b", "svg",  # output format
        "--opttolerance", "2.8",  # higher values make paths smoother
    ]
    # fmt: on
    _ = subprocess.run(command, check=True)
    return svg_path


def get_layer_color_index(layer: _IndexMatrix) -> int:
    """Get the color index of a layer.

    :param layer: (r, c) array where -1 is transparent and opaque pixels are all
        filled with the same index.
    :return: the color index

    Just a bunch of sanity checks.
    """
    layer_values = sorted(np.unique(layer))
    if len(layer_values) == 1:
        assert layer_values[0] != -1
        return layer_values[0]
    assert len(layer_values) == 2
    assert layer_values[0] == -1
    return layer_values[1]


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


def _get_svg_path_from_potrace_output(path_to_potrace_output: Path, fill_color: tuple[float, float, float]) -> EtreeElement:
    """Get an svg `g` element from the svg output of potrace.

    :param path_to_mono_bmp: path to the monochrome image.
    :return: a `g` element from the svg

    This has to write the image to disk then read it, then extract the `g` element.
    """
    fill_rgb = float_tuple_to_8bit_int_tuple(fill_color)
    elem = etree.parse(str(path_to_potrace_output)).getroot()[1]
    _ = update_element(elem, fill=svg_color_tuple(fill_rgb))
    return elem

def _new_background_elem(colormap: _PixelVector, layer: _IndexMatrix) -> EtreeElement:
    """Create a background element.

    :param colormap: (r, 3) array of colors
    :param layer: (r, c) array where -1 is transparent and opaque pixels are all
        filled with the same index colormap.
    :return: a `rect` element

    The background is the first layer, and it's just a rectangle filled with the color
    of the first layer.
    """
    bg_col = colormap[get_layer_color_index(layer)]
    height, width = layer.shape
    return new_element(
        "rect", x=0, y=0, width=width, height=height, fill=svg_color_tuple(bg_col)
    )

def _new_foreground_elem(colormap: _PixelVector, layer: _IndexMatrix) -> EtreeElement:
    """Create a foreground element.

    :param colormap: (r, 3) array of colors
    :param layer: (r, c) array where -1 is transparent and opaque pixels are all
        filled with the same index colormap.
    :return: a `g` element describing a path filled with the color indexed in layer

    The foreground is the last layer, and it's just a group of paths filled with the
    color of the last layer.
    """
    mono_bmp = _write_mono_bmp(layer)
    elem_col = get_layer_color_index(layer)
    mono_svg = _write_svg_from_mono_bmp(mono_bmp)
    elem = _get_svg_path_from_potrace_output(mono_svg, colormap[elem_col])
    os.unlink(mono_bmp)
    os.unlink(mono_svg)
    return elem


def draw_posterized_image(
    colormap: _PixelVector, layers: _IndexMatrices, filename_stem: str
):
    """Draw a posterized image.

    :param colormap: (r, 3) array of colors
    :param layers: list of (r, c) arrays where -1 is transparent and opaque pixels are
        all filled with the same index colormap.
    :param filename_stem: the filename stem for the output svg

    """
    root = new_svg_root(x_=0, y_=0, width_=layers.shape[2], height_=layers.shape[1])
    root.append(_new_background_elem(colormap, layers[0]))
    for layer in layers[1:]:
        root.append(_new_foreground_elem(colormap, layer))
    svg_path = (paths.WORKING / filename_stem).with_suffix(".svg")
    _ = write_svg(svg_path, root)


