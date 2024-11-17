"""Apply matrix convolution to an layer mask.

:author: Shay Hill
:created: 2024-10-14
"""

from typing import cast
import numpy as np
from scipy.signal import convolve2d  # type: ignore
from typing import Annotated, TypeAlias, cast


import numpy as np
from lxml.etree import _Element as EtreeElement  # type: ignore
from numpy import typing as npt


_IndexMatrix: TypeAlias = Annotated[npt.NDArray[np.int64], "(r,c)"]
_FloatMatrix: TypeAlias = Annotated[npt.NDArray[np.float64], "(r,c)"]

_UNSCALED_GAUSSIAN_BLUR = np.array(
    [
        [1, 4, 7, 4, 1],
        [4, 16, 26, 16, 4],
        [7, 26, 41, 26, 7],
        [4, 16, 26, 16, 4],
        [1, 4, 7, 4, 1],
    ]
)


def _blur2d(mask: _IndexMatrix) -> _FloatMatrix:
    """Apply a 5x5 Gaussian kernel of integers to a 2D mask of integers.

    :param mask: 2D array of integers. The intended use will be for layer masks
        containing only ones and zeros, but this is not enforced.
    :return: 2D array of integers. The result of the convolution. This will be
        unscaled, so a 5x5 section of 1s will end up with 273 in the center. It could
        be scaled to a normal Gaussian blur by dividing the result by 273.
    """
    return cast(
        _FloatMatrix,
        convolve2d(mask, _UNSCALED_GAUSSIAN_BLUR, mode="same", boundary="symm"),
    )


def quantify_mask_continuity(mask: _IndexMatrix) -> float:
    """Quantify how much 1s in a mask are surrounded by other 1s.

    :param mask: 2D array of integers. The intended use will be for layer masks
        containing only ones and zeros, but this is not enforced.
    :return: from 41 to 273, the average value of a blurred mask where the mask is 1.
    """
    blurred = _blur2d(mask)
    return float(np.sum(blurred[mask == 1]))


def quantify_layer_continuity(layer: _IndexMatrix) -> float:
    """Quantify how much opaque pixels are surrounded by other opaque pixels.

    :param layer: a (r, c) array of integers where -1 is transparent and one other
        integer is a colormap index on all opaque pixels.
    :return: from 41 to 273, the average value of a blurred mask where the mask is 1.
    """
    mask = np.where(layer == -1, 0, 1)
    return quantify_mask_continuity(mask)


def shrink_mask(mask: _IndexMatrix, max_blur_steps: int = 3) -> _IndexMatrix:
    """Blur a mask to feather its edges. Then set all values below 1 to 0.

    :param mask: 2D array of 1s and 0s.
    :param max_blur_steps: the number of times to apply 5x5 convolution matrix. Each
        step will shrink the mask.
    :return: 2D array of of 1s and 0s. The result should have fewer 1s than the
        input, with 1s on boundaries replaced with 0s.
    """
    gaussian_matrix_sum = np.sum(_UNSCALED_GAUSSIAN_BLUR)
    for _ in range(max_blur_steps):
        blurred = np.where(_blur2d(mask) == gaussian_matrix_sum, 1, 0)
        if not np.any(blurred):
            break
        mask = blurred
    return mask.copy()
