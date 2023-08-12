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
from typing import Annotated, Iterable

import numpy as np
import numpy.typing as npt
from PIL import Image
from stacked_quantile import get_stacked_quantile

# a monochrome image with alpha
_LaPixels = Annotated[npt.NDArray[np.uint8], (-1, -1, 2)]

# a monochrome image with no alpha channel
_LPixels = Annotated[npt.NDArray[np.uint8], (-1, -1)]


def quantize_pixels(
    pixels: _LaPixels,
) -> tuple[
    Annotated[npt.NDArray[np.float_], (-1,)], Annotated[npt.NDArray[np.float_], (-1,)]
]:
    """Quantize pixels to uint8 values and weights per value.

    :param pixels: array of pixels shape (m, n, 2) [0, 255]
    :return: tuple of (gray values [0, 255], float weight for each gray value)

    Borrowing this use of quantize from PIL, but PIL will not quantize a
    grayscale image.
    """
    # value/weight pairs
    vws = np.reshape(pixels, (-1, 2))
    vws = vws[vws[:, 1] > 0]
    accumulated_weights = [0.0] * 256
    for value, weight in vws:
        accumulated_weights[int(value)] += weight
    # fmt: off
    return (
        np.array(tuple(range(256)), dtype=np.float_),
        np.array(accumulated_weights, dtype=np.float_)
    )
    # fmt: on


def get_image_pixels(filename: Path | str) -> _LaPixels:
    """Get gray and alpha levels from a (presumably rgba) image.

    :param filename: path to an image, presumable a png with transparency
    :return: array with shape (-1, -1, 2) of uint8 values
    """
    image = Image.open(filename)
    image = image.convert("LA")
    return np.array(image)


def get_quantiles(pixels: _LaPixels, times: Iterable[float]) -> list[float]:
    """Adjust times to approximate percentages of image pixels.

    :param pixels: array of pixels shape (m, n, 2) [0, 255]
    :param times: percentage values in [0, 1]
    :return: list of adjusted times [0, 1]

    Given an array of grayscale values and a sequence of time values, for each
    time value, return a float that is higher than approximately time percent
    of pixels.

    E.g., if times = [0, 0.5, 1], then the return value would be 0 for 0 (higher
    than 0% of gray values), 1 for 1 (higher or equal to than 100% of gray
    values), and some value for 0.5 than would be higher than or equal to 50% of
    gray values.
    """
    gray_values, weights = quantize_pixels(pixels)
    q_times: list[float] = [0]
    for time in times:
        if time == 0:
            # return 0 instead of lowest grayscale value to preserve special
            # silhouette case of lux == 0
            q_times.append(0)
            continue
        q_times.append(get_stacked_quantile(gray_values, weights, time) / 255)
    return q_times


def _write_bitmap_from_array(pixels: _LPixels, filename: Path | str) -> None:
    """Create a bitmap file from an nxn array of pixel colors.

    :param pixels: (-1, -1, 3) array of rgb unit8 values
    :param filename: path to output bitmap (will end up with extension .bmp)
    :effects: writes a bitmap to the filesystem
    """
    image = Image.fromarray(pixels)  # type: ignore
    image.save(Path(filename).with_suffix(".bmp"))


def write_silhouette_bmp(path: str | Path, pixels: _LaPixels) -> None:
    """Use the alpha channel of each pixel to create a black / white bmp file.

    :param path: path to output file
    :param pixels: array of pixels shape (m, n, 4) where the last dimension is [0, 255]

    All transparency will be white. All opaque pixels will be black.
    """
    alphas = (255 - pixels[:, :, 1]).astype(np.uint8)
    _write_bitmap_from_array(alphas, path)


def _add_white_background(pixels: _LaPixels) -> _LaPixels:
    """Replace transparency with white.

    :param pixels: array of pixels shape (m, n, 4) [0, 255]
    :return: array of pixels shape (m, n, 3) [0, 255]
    """
    image = Image.fromarray(pixels, mode="LA")  # type: ignore
    new_image = Image.new("LA", image.size, "WHITE")
    new_image.paste(image, mask=image)
    return np.array(new_image)


def write_monochrome_bmp(path: str | Path, pixels: _LaPixels) -> None:
    """Use the color channels of each pixel to create a grayscale bmp file.

    :param path: path to output file
    :param pixels: array of pixels shape (m, n, 4) [0, 255]

    Replace transparent background with white.
    """
    pixels_ = _add_white_background(pixels)
    _write_bitmap_from_array(pixels_[:, :, 0], path)
