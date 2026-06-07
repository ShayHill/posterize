"""Functions to deal with image-approximation layers.

Each layer is 512 values, each either a single color index or -1.

The original colormap is 512 unique values. For each layer, 1 color index is selected
and each color in the original map is compared to it. Where the selected color index
would improve the approximation (in comparison to lower layers), the value in that
colormap position [0..511] is set to the selected color index. Where the selected
color index would *not* improve the approximation, the value is set to -1.

:author: Shay Hill
:created: 2025-04-25
"""

from typing import TypeAlias

import numpy as np
from numpy import typing as npt

_IntA: TypeAlias = npt.NDArray[np.intp]


def merge_layers(*layers: _IntA, size: int | None = None) -> _IntA:
    """Merge layers into a single layer.

    :param layers: n shape (palette_size,) layer arrays, each containing at most two
        values:
        * a color index that will replace one or more indices in the quantized image
        * -1 for transparent. The first layer will be a solid color and contain no -1
    :param size: palette size; required only when no layers are provided, used to
        shape the all-transparent result. For a color image, this should be 512. For
        a mono image, this should be 256.
    :return: one (palette_size,) array with the last non-transparent color in each
        position
    :raises ValueError: if no layers are provided and size is None

    Where an image is a (rows, cols) array of indices---each layer of an
    approximation will color some of those indices with one palette index per layer,
    and others with -1 for transparency.
    """
    if not layers:
        if size is None:
            msg = "size is required when merging zero layers"
            raise ValueError(msg)
        return np.full(size, -1, dtype=np.intp)
    merged: _IntA = layers[0] * 1
    for layer in layers[1:]:
        non_transparent = layer != -1
        merged[non_transparent] = layer[non_transparent]
    return merged


def apply_mask(layer: _IntA, mask: _IntA | None) -> _IntA:
    """Apply a mask to a layer if the mask is not None.

    :param layer: the layer to apply the mask to (shape (512,) consisting of one
        palette index and -1 where transparent)
    :param mask: the mask to apply to the layer (shape (512,)) consisting of 1s and 0s
    :return: the layer with the mask applied (shape (512,)) with, most likely,
        additional transparent (-1) values
    """
    if mask is None:
        return layer
    return np.where(mask == 1, layer, -1)
