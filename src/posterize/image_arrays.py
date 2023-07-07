"""Convert images to numpy arrays and manipulate those arrays.

:author: Shay Hill
:created: 2023-07-06
"""

import numpy as np
import numpy.typing as npt
from typing import Iterable, Annotated
from posterize.constants import RGB_CONTRIB_TO_GRAY, WORKING_SIZE
from pathlib import Path
from PIL import Image


_MonoPixels = Annotated[npt.NDArray[np.uint8], (-1, -1)]
_Pixels = Annotated[npt.NDArray[np.uint8], (-1, -1, 3)]
_Vec2 = Annotated[npt.NDArray[np.int32], 2]

FPArray = npt.NDArray[np.float_]


def robust_interp(values: Iterable[float]) -> Annotated[FPArray, -1]:
    """Interp values to 0 to 1, bypassing outliers.

    :param values: values to interp
    :return: values interp to 0 to 1

    This is a documented, bog-standard algorithm. Plenty of examples online.
    """
    values = np.array([x for x in values])
    median = np.median(values)
    twostd = np.std(values) * 2

    min_ = float(max(np.min(values), median - twostd))
    max_ = float(max(np.min(values), median + twostd))
    return np.interp(values, np.array([min_, max_]), np.array([0, 1]))


def get_image_colors(filename: str | Path) -> _Pixels:
    """Get colors from a quantized image.

    :param filename: path to an image
    :return: three-element array for each pixel
    """
    image = Image.open(filename)
    image = image.convert("RGBA")
    # image.thumbnail((WORKING_SIZE, WORKING_SIZE))
    return np.array(image)


def convert_mono_pixels_to_rgb(mono_pixels: _MonoPixels) -> _Pixels:
    """Transform integer pixel values to [int, int, int]

    Given an array of image pixels (r, c) defined by one uint8 value per pixel,
    create an array (r, c, 3) defining each pixel as three uint8 values.

        >>> mono = np.array([[5, 6], [7, 8]], dtype=np.uint8)
        >>> _convert_mono_pixels_to_rgb(mono)
        array([[[5, 5, 5],
                [6, 6, 6]],
        <BLANKLINE>
               [[7, 7, 7],
                [8, 8, 8]]], dtype=uint8)
    """
    out_shape = mono_pixels.shape + (3,)
    return (np.ones(out_shape) * mono_pixels[..., np.newaxis]).astype(np.uint8)


def convert_rgb_pixels_to_mono(rgb_pixels: _Pixels) -> _MonoPixels:
    """Transform rgb pixel values [int, int, int] to int

    Given an array of image pixels (r, c, 3) defined by one three uint8 values per
    pixel, create an array (r, c) defining each pixel as one uint8 value. Uses a
    standard rgb conversion formula.

        >>> rgb = np.array([[[5, 5, 5], [6, 6, 6]],
        ...                 [[7, 7, 7], [8, 8, 8]]], dtype=np.uint8)
        >>> _convert_rgb_pixels_to_mono(rgb)
        array([[5, 6],
               [7, 8]], dtype=uint8)
    """
    # TODO: see if I can use better rounding for this
    gray = np.round(np.dot(rgb_pixels[..., :3], RGB_CONTRIB_TO_GRAY))
    breakpoint()
    return np.array(gray.reshape(rgb_pixels.shape[:2]), dtype=np.uint8)


def write_bitmap_from_array(
    pixels: _Pixels,
    filename: str | Path,
) -> None:
    """Create a bitmap file from an nxn array of pixel colors.

    :param pixels: (-1, -1, 3) array of rgb unit8 values
    :param filename: path to output bitmap (will end up with extension .bmp)
    :effects: writes a bitmap to the filesystem
    """
    image = Image.fromarray(pixels)
    image.save(Path(filename).with_suffix(".bmp"))


# TODO: find the purpose of _get_median_gray
def _get_median_gray(pixels: _Pixels | _MonoPixels) -> float:
    """Median gray level of a pixel array

    :param pixels: mono pixels (-1, -1) or rgb pixels (-1, -1, 3) as uint8
    :return: float value between 0 and 1

    Tries to return the median value. If the median is 0 or 1, returns 0.5.
    """
    if len(pixels.shape) == 2:  # mono
        mono = pixels
    else:  # rgb
        mono = convert_rgb_pixels_to_mono(pixels)
    median = np.median(mono)
    if median in (0, 255):
        return 0.5
    return float(median) / 255
