"""Convert images to numpy arrays. Manipulate and save those arrays.

Potrace needs two bitmap images to create vectors for a layered svg:

1. a monochrome image
2. a silhouette image where transparent pixels are white and opaque pixels are black.
    This is used in the special case lux == 0 to create a vector for the outline around
    the entire image.

:author: Shay Hill
:created: 2023-07-06
"""

from pathlib import Path
from typing import Annotated, Union

import numpy as np
import numpy.typing as npt
from PIL import Image

from posterize.constants import RGB_CONTRIB_TO_GRAY

# a colored image with alpha
_RgbaPixels = Annotated[npt.NDArray[np.uint8], (-1, -1, 4)]

# a colored image with no alpha
_RgbPixels = Annotated[npt.NDArray[np.uint8], (-1, -1, 3)]

# cache this number for float to int conversion
_BIG_INT = 2**24 - 1


def _float_array_to_uint8(array: npt.NDArray[np.float64]) -> npt.NDArray[np.uint8]:
    """Convert an array of floats [0.0, 255.0] to an array of uint8 [0, 255]."""
    as_uint32 = (array * _BIG_INT).astype(np.uint32)
    return np.right_shift(as_uint32, 24).astype(np.uint8)


def get_image_pixels(filename: Union[Path, str]) -> _RgbaPixels:
    """Get colors from a quantized image.

    :param filename: path to an image, presumable a png with transparency
    :return: array with shape (-1, -1, 4) of uint8 values
    """
    image = Image.open(filename)
    image = image.convert("RGBA")
    return np.array(image)


def _write_bitmap_from_array(
    pixels: _RgbPixels,
    filename: Union[Path, str],
) -> None:
    """Create a bitmap file from an nxn array of pixel colors.

    :param pixels: (-1, -1, 3) array of rgb unit8 values
    :param filename: path to output bitmap (will end up with extension .bmp)
    :effects: writes a bitmap to the filesystem
    """
    image = Image.fromarray(pixels)  # type: ignore
    image.save(Path(filename).with_suffix(".bmp"))


def write_silhouette_bmp(path: str, pixels: _RgbaPixels) -> None:
    """Use the alpha channel of each pixel to create a black / white bmp file.

    :param path: path to output file
    :param pixels: array of pixels shape (m, n, 4) where the last dimension is [0, 255]

    All transparency will be white. All opaque pixels will be black.
    """
    alphas = (255 - pixels[:, :, 3]).astype(np.uint8)
    bmp_pixels = np.repeat(alphas[:, :, np.newaxis], 3, axis=2)
    _write_bitmap_from_array(bmp_pixels, path)


def _add_white_background(pixels: _RgbaPixels) -> _RgbPixels:
    """Replace transparency with white.

    :param pixels: array of pixels shape (m, n, 4) [0, 255]
    :return: array of pixels shape (m, n, 3) [0, 255]
    """
    image = Image.fromarray(pixels, mode="RGBA")  # type: ignore
    new_image = Image.new("RGBA", image.size, "WHITE")
    new_image.paste(image, mask=image)
    return np.array(new_image.convert("RGB"))


def write_monochrome_bmp(path: str, pixels: _RgbaPixels) -> None:
    """Use the color channels of each pixel to create a grayscale bmp file.

    :param path: path to output file
    :param pixels: array of pixels shape (m, n, 4) [0, 255]

    Replace transparent background with white.
    """
    pixels_ = _add_white_background(pixels)
    grays = _float_array_to_uint8(np.dot(pixels_, RGB_CONTRIB_TO_GRAY))
    bmp_pixels = np.repeat(grays[:, :, np.newaxis], 3, axis=2)
    _write_bitmap_from_array(bmp_pixels, path)
