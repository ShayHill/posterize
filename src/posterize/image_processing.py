"""Write images to disk, convert arrays to images, call potrace.

:author: Shay Hill
:created: 2024-10-13
"""

import importlib.resources
import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, TypeAlias, cast

import numpy as np
from basic_colormath import float_tuple_to_8bit_int_tuple
from lxml import etree
from lxml.etree import _Element as EtreeElement  # pyright: ignore[reportPrivateUsage]
from numpy import typing as npt
from PIL import Image
from svg_ultralight import new_element, new_svg_root, update_element, write_svg
from svg_ultralight.strings import svg_color_tuple

from posterize.main import ImageApproximation
from posterize.paths import CACHE_DIR

if TYPE_CHECKING:
    from collections.abc import Iterable


POTRACE_EXE = importlib.resources.files("posterize.bin") / "potrace.exe"


_PixelVector: TypeAlias = Annotated[npt.NDArray[np.uint8], "(r,3)"]
_IndexMatrix: TypeAlias = Annotated[npt.NDArray[np.integer[Any]], "(r,c)"]
_IndexMatrices: TypeAlias = Annotated[npt.NDArray[np.integer[Any]], "(n,r,c)"]

_TMP_BMP = CACHE_DIR / "temp.bmp"


def _write_svg_from_mono_bmp(path_to_mono_bmp: str | os.PathLike[str]) -> Path:
    """Write an svg from a monochrome image.

    :param path_to_mono_bitmap: path to the monochrome image.
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


def _get_layer_color_index(layer: _IndexMatrix) -> int:
    """Get the color index of a layer.

    :param layer: (r, c) array where -1 is transparent and opaque pixels are all
        filled with the same index.
    :return: the color index

    Just a bunch of sanity checks.
    """
    unique = cast("Iterable[int]", np.unique(layer))
    layer_values = sorted(unique)

    match layer_values:
        case [color_index] if color_index >= 0:
            return color_index
        case [-1, color_index]:
            return color_index
        case _:
            msg = "A layer needs one colors index and -1 for any transparent pixels."
            raise ValueError(msg)


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


def _get_svg_path_from_potrace_output(
    path_to_potrace_output: str | os.PathLike[str],
    fill_color: tuple[float, float, float],
) -> EtreeElement:
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
    bg_col = colormap[_get_layer_color_index(layer)]
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

    A foreground element is any element above the background. These are `<g>`
    elements copied directly from Potrace output then colored with a color from the
    colormap.
    """
    elem_col = _get_layer_color_index(layer)
    mono_bmp = _write_mono_bmp(layer)
    try:
        mono_svg = _write_svg_from_mono_bmp(mono_bmp)
    except Exception as e:
        mono_bmp.unlink()
        msg = "Potrace failed to convert the monochrome bitmap to svg."
        raise RuntimeError(msg) from e
    try:
        elem = _get_svg_path_from_potrace_output(mono_svg, colormap[elem_col])
    finally:
        mono_bmp.unlink()
        mono_svg.unlink()
    return elem


def _draw_posterized_image(
    filename: str | os.PathLike[str], colormap: _PixelVector, layers: _IndexMatrices
) -> Path:
    """Draw a posterized image.

    :param filename: the filename stem for the output svg
    :param colormap: (r, 3) array of colors
    :param layers: list of (r, c) arrays where -1 is transparent and opaque pixels are
        all filled with the same index colormap. These are not the same layers as
        ImageApproximation.layers. These are expanded to the size of the quantized
        image.

    You'll most likely want to call this function through `draw_approximation()`.
    """
    root = new_svg_root(x_=0, y_=0, width_=layers.shape[2], height_=layers.shape[1])
    root.append(_new_background_elem(colormap, layers[0]))
    for layer in layers[1:]:
        layer = cast("_IndexMatrix", layer)
        root.append(_new_foreground_elem(colormap, layer))
    return Path(write_svg(Path(filename), root))


def _expand_layers(
    quantized_image: Annotated[npt.NDArray[np.intp], "(r, c)"],
    d1_layers: Annotated[npt.NDArray[np.intp], "(n, 512)"],
) -> Annotated[npt.NDArray[np.intp], "(n, r, c)"]:
    """Expand layers to the size of the quantized image.

    :param quantized_image: (r, c) array with palette indices
    :param d1_layers: (n, 512) an array of layers. Layers may contain -1 or any
        palette index in [0, 511].
    :return: (n, r, c) array of layers, each layer with the same shape as the
        quantized image.

    Convert the (usually (512,)) layers of an ImageApproximation to the (n, r, c)
    layers required by draw_posterized_image.
    """
    d1_layers_ = cast("Iterable[npt.NDArray[np.intp]]", d1_layers)
    return np.array([x[quantized_image] for x in d1_layers_])


def draw_approximation(
    filename: str | os.PathLike[str],
    state: ImageApproximation,
    num_cols: int | None = None,
) -> Path:
    """Draw an image approximation to an SVG file.

    :param filename: path to the output SVG file
    :param state: an ImageApproximation object
    :param num_cols: optionally create the SVG with only the first `num_cols` colors.
    :return: path to the output SVG file
    """
    big_layers = _expand_layers(state.target.indices, state.layers)
    return _draw_posterized_image(filename, state.target.palette, big_layers[:num_cols])
