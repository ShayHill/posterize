"""Create one svg layer with potrace.

Potrace needs a bitmap file as input. Create this file in the project bitmaps folder,
then create all necessary svgs in the project svgs folder.

:author: Shay Hill
:created: 2023-07-06
"""

import subprocess
from pathlib import Path
from typing import Annotated, Union
from copy import deepcopy

import numpy as np
import numpy.typing as npt
from lxml import etree
from lxml.etree import _Element as EtreeElement  # type: ignore
from svg_ultralight import new_svg_root, write_svg
from PIL import Image

from posterize import image_arrays as ia
from posterize.constants import RGB_CONTRIB_TO_GRAY
from posterize.paths import POTRACE, clear_temp_files, get_bmp_path, get_svg_path
from basic_colormath import rgb_to_hex, hex_to_rgb

# a colored image with alpha
_Pixels = Annotated[npt.NDArray[np.uint8], (-1, -1, 4)]


# for float to int conversion
_BIG_INT = 2**24 - 1


def _float_array_to_uint8(array: npt.NDArray[np.float64]) -> npt.NDArray[np.uint8]:
    """Convert an array of floats [0.0, 255.0] to an array of uint8 [0, 255]."""
    as_uint32 = (array * _BIG_INT).astype(np.uint32)
    return np.right_shift(as_uint32, 24).astype(np.uint8)


def _write_alpha_bmp(path: Path, pixels: _Pixels) -> None:
    """Use the alpha channel of each pixel to create a black / white bmp file.

    :param path: path to output file
    :param pixels: array of pixels shape (m, n, 4) where the last dimension is [0, 255]

    All transparency will be white. All opaque pixels will be black.
    """
    alphas = (255 - pixels[:, :, 3]).astype(np.uint8)
    bmp_pixels = np.repeat(alphas[:, :, np.newaxis], 3, axis=2)
    ia.write_bitmap_from_array(bmp_pixels, path)


def _add_white_background(pixels: _Pixels) -> npt.NDArray[np.uint8]:
    image = Image.fromarray(pixels, mode="RGBA")  # type: ignore
    new_image = Image.new("RGBA", image.size, "WHITE")
    new_image.paste(image, mask=image)
    return np.array(new_image.convert("RGB"))


def _write_monochrome_bmp(path: Path, pixels: _Pixels) -> None:
    """Use the color channels of each pixel to create a grayscale bmp file.

    :param path: path to output file
    :param pixels: array of pixels shape (m, n, 4) [0, 255]
    """
    pixels_ = _add_white_background(pixels)
    grays = _float_array_to_uint8(np.dot(pixels_, RGB_CONTRIB_TO_GRAY))
    bmp_pixels = np.repeat(grays[:, :, np.newaxis], 3, axis=2)
    ia.write_bitmap_from_array(bmp_pixels, path)


class SVGLayerGetter:
    """Create necessary temp files then generate SVGs at different illumination
    levels.
    """

    def __init__(self, pixels: _Pixels, filename_stem: Union[str, Path]) -> None:
        """Cache rgb pixels and determine filenames for output images

        :param pixels: pixel array with four channels, rgba (-1, -1, 4)
        :param filename_stem: name for output files. The extension will be added
        """
        # filenames for output images
        self._filename_stem = filename_stem
        self._bmp = get_bmp_path(filename_stem)
        self._alpha_bmp = get_bmp_path(filename_stem, "_alpha")

        # write required bmp files
        _write_monochrome_bmp(self._bmp, pixels)
        _write_alpha_bmp(self._alpha_bmp, pixels)

    @classmethod
    def from_image(cls, filename: Union[Path, str]) -> "SVGLayerGetter":
        """Create a source bitmap from an image."""
        return cls(ia.get_image_colors(filename), filename)

    def _write_svg(self, blacklevel: float):
        """Create an svg for a given blacklevel."""
        svg_path = get_svg_path(self._filename_stem, blacklevel)
        if blacklevel == 0:
            bitmap = self._alpha_bmp
        else:
            bitmap = self._bmp
            blacklevel = 1 - blacklevel
        # fmt: off
        command = [
            str(POTRACE),
            str(bitmap),
            "-o", str(svg_path),
            "-k", str(blacklevel),
            "-u", "1",  # do not scale svg (points correspond to pixels array)
            "--flat",  # all paths combined in one element
            "-t", "15",  # remove speckles
            "-b", "svg",  # output format
        ]
        # fmt: on
        _ = subprocess.run(command)
        return svg_path

    def _get_svg(self, blacklevel: float) -> Path:
        """Create an svg file for a given blacklevel if one does not exist."""
        svg_path = get_svg_path(self._filename_stem, blacklevel)
        if not svg_path.exists():
            _ = self._write_svg(blacklevel)
        return svg_path

    def __call__(self, blacklevel: float) -> EtreeElement:
        """Get the svg group for a given blacklevel.

        :param blacklevel: The blacklevel to use.
        :return: An svg group element. The group element contains path elements. Each
            path element surrounds a black area in the image.

        Output svgs from potrace look like this:

        ```xml
        <element tree>
            <root>
                <metadata></metadata>
                <g></g>
            </root>
        </element tree>
        ```
        """
        filename = self._get_svg(blacklevel)
        root = etree.parse(str(filename)).getroot()
        return root[1]


def hex_interp(hex_a: str, hex_b: str, time: float) -> str:
    """Interpolate between two hex colors.

    :param hex_a: The first hex color.
    :param hex_b: The second hex color.
    :param time: The time to interpolate at. 0.0 returns hex_a, 1.0 returns hex_b.
    :return: The interpolated hex color.
    """
    rgb_a = np.array(hex_to_rgb(hex_a))
    rgb_b = np.array(hex_to_rgb(hex_b))
    r, g, b = np.round(rgb_a + time * (rgb_b - rgb_a))
    return rgb_to_hex((r, g, b))


clear_temp_files()

count_levels = 10

levels = np.linspace(0, 1, count_levels+1)[:-1]
colors = [hex_interp("#ffffff", "#000000", level) for level in levels]

power = 1.8
levels = np.linspace(0, .8**power, count_levels)
levels = [pow(x, 1/power) for x in levels]

def create_svg(image: Union[Path, str]) -> EtreeElement:
    pixels = ia.get_image_colors(image)
    get_layer = SVGLayerGetter.from_image(image)
    root = new_svg_root(x_ = 0, y=0, width=pixels.shape[1], height=pixels.shape[0])
    level = levels[0]
    color = "#ff66aa"
    layer = get_layer(level)
    layer.attrib['stroke'] = color
    layer.attrib['stroke-width'] = "10"
    root.append(layer)

    for level, color in zip(levels, colors):
        layer = get_layer(level)
        layer.attrib['fill'] = color
        root.append(layer)
    write_svg("binaries/charles.svg", root)

create_svg("binaries/charles.png")

# get_layer(0.5)
