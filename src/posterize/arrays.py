"""Manipulate nxm numpy arrays.

:author: Shay Hill
:created: 2024-08-15
"""

import numpy as np
from numpy import typing as npt

_MAX_8BIT = 255
_BIG_INT = 2**32 - 1
_BIG_SCALE = _BIG_INT / _MAX_8BIT


def _get_iqr_bounds(floats: npt.NDArray[np.float64]) -> tuple[np.float64, np.float64]:
    """Get the IQR bounds of a set of floats.

    :param floats: floats to get the IQR bounds of
    :return: the IQR bounds of the floats (lower, upper)

    Use the interquartile range to determine the bounds of a set of floats (ignoring
    outliers). If an image has a lot of background (> 50% pixels are the same color),
    it will be necessary to stretch the interquartile range to have a range at all.
    """
    bot_percent = 25
    top_percent = 75
    q25 = q75 = 0
    while bot_percent >= 0:
        q25 = np.quantile(floats, bot_percent / 100)
        q75 = np.quantile(floats, top_percent / 100)
        if q25 < q75:
            break
        bot_percent -= 5
        top_percent += 5
    else:
        # image is solid color
        return np.float64(q25 - 1), np.float64(q25)

    iqr = q75 - q25
    lower: np.float64 = max(q25 - 1.5 * iqr, np.min(floats))
    upper: np.float64 = min(q75 + 1.5 * iqr, np.max(floats))
    return lower, upper


def normalize_errors_to_8bit(grid: npt.NDArray[np.float64]) -> npt.NDArray[np.uint8]:
    """Normalize a grid of floats (error delta per pixel) to 8-bit integers.

    :param grid: grid to normalize - array[float64] shape=(h, w)
    :return: normalized grid - array[uint8] shape=(h, w)

    Here, this takes a grid of floats representing the difference in error-per-pixel
    between two images. Where

    * errors_a = the error-per-pixel between the target image and the current state
    * errors_b = the error-per-pixel between the target image some potential state

    The `grid` arg here is errors_b - errors_a, so the lowest (presumably negative)
    values are where the potential state is better than the current state, and the
    highest values (presumably postive) are where the potential state is worse.

    The return value is a grid of the same shape with the values normalized to 8-bit
    integers, clipping outliers. This output grid will be used to create a monochrome
    bitmap where the color of each pixel represents the improvement we would see if
    using the potential state instead of the current state on that pixel.

    * 0: potential state is better
    * 255: potential state is worse
    """
    lower, upper = _get_iqr_bounds(grid.flatten())
    shift = 0 - lower
    scale = _MAX_8BIT / (upper - lower)
    grid = np.clip((grid + shift) * scale, 0, _MAX_8BIT)
    return ((grid * _BIG_SCALE).astype(np.uint32) >> 24).astype(np.uint8)
