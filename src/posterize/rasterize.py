"""Rasterize and convert svg lxml.etree._Element instances.

This rasterization brings in two dependencies, cairosvg and pillow, and it does not
support as many features as Inkscape png conversion, but it is a lot faster and
*does* support the paths, transparency, and rects used in this project.

:author: Shay Hill
:created: 2024-08-19
"""

import io
from typing import cast

import numpy as np
from cairosvg import svg2png  # type: ignore
from lxml import etree
from lxml.etree import _Element as EtreeElement  # type: ignore
from numpy import typing as npt
from PIL import Image
from PIL.Image import Image as ImageType


def elem_to_png_bytes(elem: EtreeElement) -> bytes:
    """Get an svg EtreeElement as PNG bytes.

    :param elem: element to convert
    :return: png as a byte string
    """
    return cast(bytes, svg2png(etree.tostring(elem)))


def elem_to_png_image(elem: EtreeElement) -> ImageType:
    """Get as svg EtreeElement as a PNG image.

    :param elem: element to convert
    :return: png as a PIL.Image.Image instance
    """
    return Image.open(io.BytesIO(elem_to_png_bytes(elem)))


def elem_to_png_array(elem: EtreeElement) -> npt.NDArray[np.uint8]:
    """Get an svg EtreeElement state as an array of pixel colors.

    :param elem: element to convert
    :return: array of pixel colors (h, w, 3) where h is height and w is width
    """
    return np.array(elem_to_png_image(elem)).astype(np.uint8)[:, :, :3]
