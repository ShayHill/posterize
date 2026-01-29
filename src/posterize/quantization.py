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

from contextlib import suppress
from pathlib import Path
from typing import Annotated, Any, TypeAlias, TypeVar, Union, cast

import numpy as np
from basic_colormath import floats_to_uint8, get_delta_e_matrix
from cluster_colors import stack_pool_cut_colors
from numpy import typing as npt
from PIL import Image

from posterize.layers import merge_layers
from posterize.paths import CACHE_DIR

_CACHE_PREFIX = "quantized_"

# Resize images larger than this to this maximum dimension. This value is necessary,
# because you can only create arrays of a certain size. A smaller value might speed
# up testing, but the quantization cache will need to be cleared if this value
# changes.
_MAX_DIM = 500

_Colors: TypeAlias = Annotated[npt.NDArray[np.uint8], "(m,3)"]
_Indices: TypeAlias = Annotated[npt.NDArray[np.intp], "(m,)"]
_Layers: TypeAlias = Annotated[npt.NDArray[np.intp], "(n, 512e"]
_Layer: TypeAlias = Annotated[npt.NDArray[np.intp], "(512,) in [0, 512)"]
_Mask: TypeAlias = Annotated[npt.NDArray[np.intp], "(n, 512) in [0, 1]"]


_AnyArray = TypeVar("_AnyArray", bound=npt.NDArray[Any])


def _np_reshape(array: _AnyArray, shape: tuple[int, ...]) -> _AnyArray:
    """Reshape an array to a given shape, preserving the dtype.

    Wrap np.reshape to eliminate the partially unknown pyright type error.
    """
    if array.shape == shape:
        return array
    # TODO: try dtype=array.dtype instead of cast
    return cast("_AnyArray", np.reshape(array, shape))


def _min_max_normalize(
    array: npt.NDArray[np.intp | np.floating[Any]],
) -> npt.NDArray[np.float64]:
    """Normalize an array to the range [0, 1].

    :param array: array to normalize
    :return: normalized array or array of zeros if all values are the same
    """
    min_val = np.min(array)
    max_val = np.max(array)
    if min_val == max_val:
        return np.zeros_like(array, dtype=np.float64)
    return (array - min_val) / (max_val - min_val)


class TargetImage:
    """Cached array values for a quantized image and methods to calculate costs.

    These are intended to be created deterministacally from an image where the weight
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
        """Get the cost-per-pixel between self.image and (state + layers).

        :param layers: layers to apply to the current state. There will only ever be
            0 or 1 layers. If 0, the cost matrix of the current state will be
            returned.  If 1, the cost matrix with a layer applied over it.
        :return: cost-per-pixel between image and (state + layer)

        This should decrease after every append.
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
        """Get the cost between self.image and state with layers applied.

        :param layers: layers to apply to the current state. There will only ever be
            one layer.
        :return: sum of the cost between image and (state + layer)
        """
        cost_matrix = self.get_cost_matrix(*layers)
        if mask is not None:
            cost_matrix[np.where(mask == 0)] = 0
        return float(np.sum(cost_matrix))


def _index_to_nearest_color(colormap: _Colors, colors: _Colors) -> _Indices:
    """Map a full set of image colors to a colormap.

    :param colormap: colormap to map to (m, n)
    :param colors: colors (p, n)
    :return: a (p, 1) array of indices to the colormap

    For each pixel in an image array (n, 3), find the index of the closest vector
    in `colormap`.
    """
    unique_colors, reverse_index = np.unique(colors, axis=0, return_inverse=True)
    pmatrix = get_delta_e_matrix(unique_colors, colormap)
    return np.argmin(pmatrix[reverse_index], axis=1)


def _get_cache_paths(source: Path) -> dict[str, Path]:
    """Get the cache paths for the quantized image.

    :param source: path to an image
    :return: a dictionary with the cache paths for the quantized image
        * The keys are "palette", "indices", and "pmatrix"
        * The values are the path + stems to the cache files
    """
    prefix = f"{_CACHE_PREFIX}_{source.stem}"
    attribs = ("palette", "indices", "pmatrix")
    return {a: CACHE_DIR / f"{prefix}_{a}.npy" for a in attribs}


def clear_quantized_image_cache(source: Path) -> None:
    """Clear the cache for one quantized image.

    :param source: path to the source image previously quantizes
    """
    cache_paths = _get_cache_paths(source)
    for path in cache_paths.values():
        if path.exists():
            path.unlink()


def clear_all_quantized_image_caches() -> None:
    """Clear all caches for quantized images."""
    for path in CACHE_DIR.glob(f"{_CACHE_PREFIX}*.npy"):
        path.unlink()


def _load_quantized_image(source: Path) -> TargetImage:
    """Load a quantized image from the cache.

    :param source: path to the source image previously quantizes
    :return: a TargetImage object
    """
    cache_paths = _get_cache_paths(source)
    if not all(path.exists() for path in cache_paths.values()):
        msg = f"Cache files not found for {source}. "
        raise FileNotFoundError(msg)
    palette = np.load(cache_paths["palette"])
    indices = np.load(cache_paths["indices"])
    pmatrix = np.load(cache_paths["pmatrix"])
    return TargetImage(palette, indices, pmatrix)


def _dump_quantized_image(quantized_image: TargetImage, source: Path) -> None:
    """Dump a quantized image to the cache.

    :param quantized_image: a TargetImage object
    :param source: path to the source image previously quantizes
    """
    cache_paths = _get_cache_paths(source)
    for name, path in cache_paths.items():
        with path.open("wb") as f:
            np.save(f, getattr(quantized_image, name))


def _quantize_image(image: Image.Image) -> TargetImage:
    """Reduce an RGBA image to 512 indexed colors.

    :param image: PIL Image in RGBA mode
    :return: TargetImage (palette, indices, pmatrix)
    """
    rgba_colors = _np_reshape(np.array(image), (-1, 4))
    rgb_colors = rgba_colors[:, :3]
    palette = np.array(floats_to_uint8(stack_pool_cut_colors(rgba_colors)[:, :3]))
    indices = _index_to_nearest_color(palette, rgb_colors)
    indices = _np_reshape(indices, (image.height, image.width))
    pmatrix = get_delta_e_matrix(palette)
    return TargetImage(palette, indices, pmatrix)


def new_target_image(
    source: Path | Image.Image, *, ignore_cache: bool = False
) -> TargetImage:
    """Reduce an image to 512 indexed colors.

    :param source: path to an image or a PIL Image object
    :param ignore_cache: if True, ignore any cached results
        (only used when source is a Path)
    :return: a TargetImage object (palette, indices, pmatrix, weights)
    """
    if isinstance(source, Path):
        if ignore_cache:
            clear_quantized_image_cache(source)
        with suppress(FileNotFoundError):
            return _load_quantized_image(source)
        image = Image.open(source)
        if max(image.size) > _MAX_DIM:
            image.thumbnail((_MAX_DIM, _MAX_DIM), Image.Resampling.LANCZOS)
        image = image.convert("RGBA")
        quantized_image = _quantize_image(image)
        _dump_quantized_image(quantized_image, source)
        return quantized_image
    image = source.convert("RGBA")
    return _quantize_image(image)


def new_target_image_mono(
    pixels: Annotated[npt.NDArray[np.uint8], "(r, c)"],
) -> TargetImage:
    """Build a TargetImage from an (r, c) uint8 array (e.g. grayscale).

    :param pixels: (r, c) array of uint8 values.
    :return: TargetImage with palette (n_unique, 3), indices (r, c), pmatrix

    This is a faster way for creating a TargetImage from a grayscale image or an array
    of values created by the bezograph algorithm.
    """
    unique_vals = np.unique(pixels)
    flat = pixels.ravel()
    indices = _np_reshape(np.searchsorted(unique_vals, flat), pixels.shape)
    palette = np.repeat(unique_vals[:, np.newaxis], 3, axis=1).astype(np.uint8)
    pmatrix = get_delta_e_matrix(palette)
    return TargetImage(palette, indices, pmatrix)
