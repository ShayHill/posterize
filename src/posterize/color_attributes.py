"""Return quantified color attributes between 0 and 1 for a given color.

For the purposes of this module, gray is (127, 127, 127). Values of (128, 128, 128)
will be treated as *almost* gray.

:author: Shay Hill
:created: 2025-04-29
"""

from typing import Annotated, TypeAlias

import numpy as np
from numpy import typing as npt

_RGB: TypeAlias = Annotated[npt.NDArray[np.uint8], (3,)]
_RGBWCMYK: TypeAlias = Annotated[npt.NDArray[np.uint8], (8,)]


def get_chromacity(color: _RGB) -> float:
    """Return the vibrance of the color.

    What is the inverse, relative distance to the color circle?

    A color with a vibrace of 0 is a shade of gray, while a color with a vibrance of
    1 is a pure color with maximum saturation and lightness.
    """
    return (max(color) - min(color)) / 255


def _get_rgb_dist(rgb: _RGB) -> _RGBWCMYK:
    """Convert red, green, blue, transparent to an 8-channel color.

    :param rgba: 4-channel rgba color
    :return: 8-channel color distribution
    """
    r, g, b = rgb
    w: float = min(r, g, b)
    k: float = 255 - max(r, g, b)
    y: float = min(r, g) - w
    c: float = min(g, b) - w
    m: float = min(b, r) - w
    r -= max(m, y) + w
    g -= max(y, c) + w
    b -= max(c, m) + w
    return np.array([r, g, b, w, c, m, y, k])


def get_purity(color: _RGB) -> float:
    """Return the purity of the color.

    What is the inverse, relative distance to the surface of the color cube? A color
    with a purity of 0 is pure gray (127, 127, 127). A color with a purity of one is
    a pure color, a shaded (mixed with black) or a tinted (mixed with white) color.
    """
    _, _, _, white, _, _, _, black = _get_rgb_dist(color)
    return 1 - (min(white, black) / 127)
