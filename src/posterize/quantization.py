"""Reduce an image to 512 indexed colors.

This is much slower than the quantize methods in PIL, but it is much less inclined to
eat bright colors. It might take a few minutes for just one image, but the results
are cached and you end up with not only a quantized image, but a proximity matrix
between the colors in the palette.

There is a limit to how big an array you can create, so large images are resized to
_MAX_DIM if they are larger than _MAX_DIM in either dimension.

:author: Shay Hill
:created: 2025-04-24
"""

import dataclasses
import os
from typing import Annotated, Any, TypeAlias, TypeVar, cast

import diskcache
import numpy as np
from basic_colormath import floats_to_uint8, get_delta_e_matrix
from cluster_colors import stack_pool_cut_colors
from numpy import typing as npt
from PIL import Image

from posterize.layers import merge_layers

cache = diskcache.Cache(".cache_quantize")

# Resize images larger than this to this maximum dimension. This value is necessary,
# because you can only create arrays of a certain size. A smaller value might speed
# up testing, but the quantization cache will need to be cleared if this value
# changes.
_MAX_DIM = 500

_Colors: TypeAlias = Annotated[npt.NDArray[np.uint8], "(m,3)"]
_Indices: TypeAlias = Annotated[npt.NDArray[np.intp], "(m,)"]
_Layers: TypeAlias = Annotated[npt.NDArray[np.intp], "(n, 512)"]
_Layer: TypeAlias = Annotated[npt.NDArray[np.intp], "(512,) in [0, 512)"]
_Mask: TypeAlias = Annotated[npt.NDArray[np.intp], "(n, 512) in [0, 1]"]


_AnyArray = TypeVar("_AnyArray", bound=npt.NDArray[Any])


def _np_reshape(array: _AnyArray, shape: tuple[int, ...]) -> _AnyArray:
    """Reshape an array to a given shape, preserving the dtype.

    Wrap np.reshape to eliminate the partially unknown pyright type error.
    """
    if array.shape == shape:
        return array
    return cast("_AnyArray", np.reshape(array, shape))


def _min_max_normalize(
    array: npt.NDArray[np.intp | np.floating[Any]],
) -> npt.NDArray[np.float64]:
    """Normalize an array to the range [0, 1].

    :param array: array to normalize
    :return: normalized array or array of ones if all values are the same
    """
    min_val = np.min(array)
    max_val = np.max(array)
    if min_val == max_val:
        return np.ones_like(array, dtype=np.float64)
    return cast("npt.NDArray[np.float64]", (array - min_val) / (max_val - min_val))


class TargetImage:
    """Cached array values for a quantized image and methods to calculate costs.

    These are intended to be created deterministically from an image where the weight
    of each color in the palette will correspond to the number of times that color is
    used in the image, but I have created a `weights` attribute setter for
    experimentation with *weighting the weights* with, for instance, color vibrance
    values.
    """

    def __init__(
        self,
        palette: Annotated[npt.NDArray[np.uint8], "(512,3)"],
        indices: Annotated[npt.NDArray[np.intp], "(r,c)"],
        pmatrix: Annotated[npt.NDArray[np.float64], "(512,512)"],
        weights: None | Annotated[npt.NDArray[np.float64], "(512,)"] = None,
    ) -> None:
        """Initialize a TargetImage object.

        :param palette: (512, 3) array of color vectors
        :param indices: (r, c) array of indices to the palette colors. This defines
            the source image in 512 colors.
        :param pmatrix: (512, 512) array of delta E values between the palette
            colors.
        :param weights: (512,) optionally initialize with something other than the
            default weights (pixel count per color).
        """
        self.palette = palette
        self.indices = indices
        self.pmatrix = pmatrix
        self._weights = weights

        # cache and expose the default weights for weight transformation functions.
        self.default_weights = np.bincount(
            self.indices.flatten(), minlength=self.palette.shape[0]
        ).astype(np.float64)

    @property
    def weights(self) -> npt.NDArray[np.float64]:
        """Get a weight for each color in the palette.

        By default, this will be a min-maxed normalization of the pixel count for
        each palette color. Min-max normalization isn't required for palette
        generation or transformation by multiplicative combination, but it simplifies
        transforming these values into a weighted average.
        """
        if self._weights is not None:
            return self._weights
        self._weights = _min_max_normalize(self.default_weights)
        return self._weights

    @weights.setter
    def weights(self, value: npt.NDArray[np.float64]) -> None:
        """Set the weights for each color in the palette.

        :param value: (512,) array of weights for each color in the palette.

        This is expected to be a transformation of the default weights.
        """
        self._weights = value

    def reset_weights(self) -> None:
        """Reset weights to the number of times each color is used in the image."""
        self._weights = None

    def get_cost_matrix(self, *layers: _Layers) -> npt.NDArray[np.floating[Any]]:
        """Get the cost-per-pixel between self.image and the merged layers.

        :param layers: zero or more layers; merged in order to form the current state
        :return: cost-per-pixel between image and merged state

        This should decrease after every layer appended.
        """
        state = merge_layers(*layers)
        filled = np.where(state != -1)
        image = np.array(range(self.pmatrix.shape[0]), dtype=int)
        cost_matrix = np.full_like(state, np.inf, dtype=float)
        cost_matrix[filled] = (
            self.pmatrix[image[filled], state[filled]] * self.weights[filled]
        )
        return cost_matrix

    def get_cost(self, *layers: _Layer, mask: _Mask | None = None) -> float:
        """Get the cost between self.image and the merged layers.

        :param layers: one or more layers (e.g. current state, or state plus a
            candidate layer) merged in order
        :param mask: optional mask; where 0, cost is excluded from the sum
        :return: sum of the cost between image and merged state
        """
        cost_matrix = self.get_cost_matrix(*layers)
        if mask is not None:
            cost_matrix[np.where(mask == 0)] = 0
        return float(np.sum(cost_matrix))


def _index_to_nearest_color(colormap: _Colors, colors: _Colors) -> _Indices:
    """Map a full set of image colors to a colormap.

    :param colormap: colormap to map to (m, 3)
    :param colors: colors (p, 3)
    :return: (p,) array of indices into colormap

    For each row in colors, find the index of the closest row in colormap.
    """
    unique_colors, reverse_index = np.unique(colors, axis=0, return_inverse=True)
    pmatrix = get_delta_e_matrix(unique_colors, colormap)
    return np.argmin(pmatrix[reverse_index], axis=1)


@dataclasses.dataclass
class Quantized:
    """A quantized image.

    :param palette: (n, 3) array of color vectors
    :param indices: (r, c) array of indices to the palette colors. This defines
        the source image in n colors.
    :param pmatrix: (n, n) array of delta E values between the palette colors.
    :param alphas: (n,) array of alpha values for each color.
    """

    palette: Annotated[npt.NDArray[np.uint8], "(n,3)"]
    indices: Annotated[npt.NDArray[np.intp], "(r,c)"]
    pmatrix: Annotated[npt.NDArray[np.float64], "(n,n)"]
    alphas: Annotated[npt.NDArray[np.uint8], "(n,)"]


def quantize_image(
    image_path: str | os.PathLike[str], max_dim: int | None = None
) -> Quantized:
    """Reduce an image file to at most 512 indexed colors.

    :param image_path: path to the image file
    :param max_dim: maximum width or height; image is thumbnailed if larger
    :return: Quantized with palette, indices, pmatrix, and alphas
    """
    max_dim = max_dim or _MAX_DIM
    image = Image.open(image_path)
    if max(image.size) > max_dim:
        image.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    image = image.convert("RGBA")
    return quantize_rgba(np.array(image))


@cache.memoize()
def quantize_rgba(
    rgba_pixels: Annotated[npt.NDArray[np.uint8], "(r, c, 4)"],
) -> Quantized:
    """Reduce an (r, c, 4) RGBA array to at most 512 indexed colors.

    :param rgba_pixels: (r, c, 4) uint8 array of RGBA pixels
    :return: Quantized with palette, indices, pmatrix, and alphas
    """
    rgbs = rgba_pixels[..., :3]
    alphas = rgba_pixels[..., 3]
    rgb_colors = _np_reshape(rgbs, (-1, 3)).astype(float)
    palette = np.array(floats_to_uint8(stack_pool_cut_colors(rgb_colors)))
    indices = _index_to_nearest_color(palette, rgb_colors)
    indices = _np_reshape(indices, rgba_pixels.shape[:2])
    pmatrix = get_delta_e_matrix(palette)
    return Quantized(palette, indices, pmatrix, alphas)


@cache.memoize()
def quantize_mono(mono_pixels: Annotated[npt.NDArray[np.uint8], "(r, c)"]) -> Quantized:
    """Reduce a (r, c) uint8 array to at most 256 indexed colors.

    :param mono_pixels: (r, c) uint8 array of grayscale pixels
    :return: Quantized with palette, indices, pmatrix, and alphas
    """
    unique_vals = np.unique(mono_pixels)
    flat = mono_pixels.ravel()
    indices = _np_reshape(np.searchsorted(unique_vals, flat), mono_pixels.shape)
    palette = np.repeat(unique_vals[:, np.newaxis], 3, axis=1).astype(np.uint8)
    pmatrix = get_delta_e_matrix(palette)
    alphas = np.ones_like(unique_vals, dtype=np.uint8)
    return Quantized(palette, indices, pmatrix, alphas)


def new_target_image(
    path: str | os.PathLike[str], max_dim: int | None = None
) -> TargetImage:
    """Reduce an image to 512 indexed colors.

    :param path: path to an image file
    :param max_dim: maximum width or height; image is thumbnailed if larger
    :return: a TargetImage object (palette, indices, pmatrix, weights)
    """
    quantized = quantize_image(path, max_dim)
    return TargetImage(quantized.palette, quantized.indices, quantized.pmatrix)


def new_target_image_mono(
    pixels: Annotated[npt.NDArray[np.uint8], "(r, c)"],
) -> TargetImage:
    """Reduce a grayscale image to 256 indexed colors.

    :param pixels: (r, c) array of uint8 values.
    :return: a TargetImage object (palette, indices, pmatrix, weights)
    """
    quantized = quantize_mono(pixels)
    return TargetImage(quantized.palette, quantized.indices, quantized.pmatrix)
