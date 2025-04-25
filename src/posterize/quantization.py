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
import functools as ft
from contextlib import suppress
from pathlib import Path
from typing import Annotated, TypeAlias, Any

import numpy as np
from basic_colormath import floats_to_uint8, get_delta_e_matrix
from cluster_colors import stack_pool_cut_colors
from numpy import typing as npt
from PIL import Image

from posterize.paths import CACHE_DIR
from posterize.layers import merge_layers

_CACHE_PREFIX = "quantized_"

# Resize images larger than this to this maximum dimension. This value is necessary,
# because you can only create arrays of a certain size. A smaller value might speed
# up testing, but the quantization cache will need to be cleared if this value
# changes.
_MAX_DIM = 1000

_Colors: TypeAlias = Annotated[npt.NDArray[np.uint8], "(m,3)"]
_Indices: TypeAlias = Annotated[npt.NDArray[np.intp], "(m,)"]
_Layers: TypeAlias = Annotated[npt.NDArray[np.intp], "(n, 256)"]
_IntA: TypeAlias = npt.NDArray[np.intp]

class TargetImage:
    """A type to store input images and evaluate approximation costs."""

    def __init__(self, path: Path) -> None:
        """Initialize a TargetImage.

        :param path: path to the image
        """
        self.path = path

        quantized_image = quantize_image(path)
        self.vectors = quantized_image.palette
        self.image = quantized_image.indices
        self.pmatrix = quantized_image.pmatrix
        self.weights = quantized_image.weights

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

    def get_cost(self, *layers: _IntA, mask: _IntA | None = None) -> float:
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


@dataclasses.dataclass(frozen=True)
class QuantizedImage:
    """Cached array values for a quantized image.

    palette: (512, 3) array of color vectors
    indices: (r, c) array of indices to the palette colors. This defines the source
        image in 512 colors.
    pmatrix: (512, 512) array of delta E values between the palette colors.
    weights: (512,) array of the number of times each palette color is used in the
        quantized image.
    """

    palette: Annotated[npt.NDArray[np.uint8], "(512,3)"]
    indices: Annotated[npt.NDArray[np.intp], "(r,c)"]
    pmatrix: Annotated[npt.NDArray[np.float64], "(512,512)"]

    @ft.cached_property
    def weights(self) -> Annotated[npt.NDArray[np.intp], "(512,)"]:
        """Count the number of times each color is used in the quantized image."""
        return np.bincount(self.indices.flatten(), minlength=len(self.palette))


def _get_cache_paths(source: Path) -> dict[str, Path]:
    """Get the cache paths for the quantized image.

    :param source: path to an image
    :return: a dictionary with the cache paths for the quantized image
        * The keys are "palette", "indices", and "pmatrix"
        * The values are the path + stems to the cache files
    """
    prefix = f"{_CACHE_PREFIX}_{source.stem}"
    attribs = ("palette", "indices", "pmatrix")
    return {a: CACHE_DIR.with_name(f"{prefix}_{a}.npy") for a in attribs}


@staticmethod
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


def _load_quantized_image(source: Path) -> QuantizedImage:
    """Load a quantized image from the cache.

    :param source: path to the source image previously quantizes
    :return: a QuantizedImage object
    """
    cache_paths = _get_cache_paths(source)
    if not all(path.exists() for path in cache_paths.values()):
        msg = f"Cache files not found for {source}. "
        raise FileNotFoundError(msg)
    palette = np.load(cache_paths["palette"])
    indices = np.load(cache_paths["indices"])
    pmatrix = np.load(cache_paths["pmatrix"])
    return QuantizedImage(palette, indices, pmatrix)


def _dump_quantized_image(quantized_image: QuantizedImage, source: Path) -> None:
    """Dump a quantized image to the cache.

    :param quantized_image: a QuantizedImage object
    :param source: path to the source image previously quantizes
    """
    cache_paths = _get_cache_paths(source)
    for name, path in cache_paths.items():
        with path.open("wb") as f:
            np.save(f, getattr(quantized_image, name))


def quantize_image(source: Path, *, ignore_cache: bool = False) -> QuantizedImage:
    """Reduce an image to 512 indexed colors.

    :param source: path to an image
    :param ignore_cache: if True, ignore any cached results
    :return: a QuantizedImage object (palette, indices, pmatrix, weights)
    """
    if ignore_cache:
        clear_quantized_image_cache(source)
    with suppress(FileNotFoundError):
        return _load_quantized_image(source)

    image = Image.open(source)
    if max(image.size) > _MAX_DIM:
        image.thumbnail((_MAX_DIM, _MAX_DIM), Image.LANCZOS)
    image = image.convert("RGBA")

    rgba_colors = np.array(image).reshape(-1, 4)
    rgb_colors = rgba_colors[:, :3]
    palette = np.array(floats_to_uint8(stack_pool_cut_colors(rgba_colors)[:, :3]))
    indices = _index_to_nearest_color(palette, rgb_colors).reshape(
        image.height, image.width
    )
    pmatrix = get_delta_e_matrix(palette)

    quantized_image = QuantizedImage(palette, indices, pmatrix)
    _dump_quantized_image(quantized_image, source)
    return quantized_image
