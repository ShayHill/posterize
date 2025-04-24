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

from pathlib import Path
from typing import Annotated, TypeAlias, TypeVar
import functools as ft
import dataclasses
import numpy as np
from basic_colormath import (
    get_sqeuclidean,
    get_sqeuclidean_matrix,
    get_delta_e_matrix,
    floats_to_uint8,
)
from contextlib import suppress
from cluster_colors import (
    Members,
    SuperclusterBase,
    get_image_supercluster,
    stack_pool_cut_colors,
)
from cluster_colors.cluster_supercluster import SuperclusterBase, Members
from numpy import typing as npt
from PIL import Image
from typing import Any
import time

from posterize.paths import CACHE_DIR

_CACHE_PREFIX = "quantized_"

_MAX_DIM = 1000

_IndexMatrix: TypeAlias = Annotated[npt.NDArray[np.int64], "(r,c)"]
_RgbMatrix: TypeAlias = Annotated[npt.NDArray[np.uint8], "(r,c,3)"]
_Colors: TypeAlias = Annotated[npt.NDArray[np.uint8], "(m,3)"]
_Indices: TypeAlias = Annotated[npt.NDArray[np.integer[Any]], "(m,)"]
_TSuperclusterBase = TypeVar("_TSuperclusterBase", bound=SuperclusterBase)


def resample_image(
    supercluster: _TSuperclusterBase, pixel_indices: _IndexMatrix
) -> _TSuperclusterBase:
    """Resample an image to the supercluster's palette.

    :param supercluster: supercluster containing members containing the palette
    :param pixels: (r, c) array with a colormap array index for each pixel
    :return: a new supercluster with a new Members instance
        * The new Members instance will contain the vectors and pmatrix of the input
        supercluster.members with an updated weight array.
        * The new SuperclusterBase instance will contain the new Members instance and
        whatever indices are used in the input pixels.

    Examine a grid of indices to `supercluster.members.vectors`:

    * replace each weight with the number of times the index appears in
      `pixel_indices`
    * replace supercluster.ixs with the indices used in `pixel_indices`
    """
    members = supercluster.members
    weights = np.zeros(len(supercluster.members.vectors))
    ixs, weights_subset = np.unique(pixel_indices, return_counts=True)
    weights[ixs] = weights_subset
    return type(supercluster)(
        Members(members.vectors, weights=weights, pmatrix=members.pmatrix), ixs
    )


def _map_pixels_to_members_vectors(
    supercluster: SuperclusterBase, path: Path, *, ignore_cache: bool = False
) -> _IndexMatrix:
    """Map an image to a colormap.

    :param supercluster: supercluster containing members with a colormap
    :param path: path to an image
    :param ignore_cache: if True, ignore any cached results
    :return: an (r, c) array with a colormap array index for each pixel

    For each pixel in an image array (r, c, 3), find the index of the closest vector
    in `supercluster.members.vectors`.
    """
    cache_path = CACHE_DIR / f"{path.stem}_colormapped_{len(supercluster.ixs):03}.npy"
    if not ignore_cache and cache_path.exists():
        return np.load(cache_path)

    colormap = supercluster.members.vectors
    image = Image.open(path)
    image = image.convert("RGB")
    image_rgb_vector = np.array(image).astype(float).reshape(-1, 3)
    image_idx_vector = np.argmin(
        get_sqeuclidean_matrix(image_rgb_vector, colormap), axis=1
    )
    image_idx_matrix = image_idx_vector.reshape(image.size[1], image.size[0])

    np.save(cache_path, image_idx_matrix)
    return image_idx_matrix


def new_supercluster_with_quantized_image(
    supercluster_type: type[_TSuperclusterBase],
    path: Path,
    *,
    ignore_cache: bool = False,
) -> tuple[_TSuperclusterBase, _IndexMatrix]:
    """Quantize an image to a colormap.

    :param supercluster: supercluster containing members with a colormap
    :param path: path to an image
    :param ignore_cache: if True, ignore any cached results
    :return: a new type(supercluster) instance with a new Members instance
        * The new Members instance will contain the vectors and pmatrix of the input
        supercluster.members with an updated weight array.
        * The new SuperclusterBase instance will contain the new Members instance and
        whatever indices are used in the input pixels.

    * Approximate and image by mapping the image pixel colors to
      supercluster.members.vectors.
    * Update supercluster.ixs to contain indices to only the vectors used in the
      image approximation.
    * Update supercluster.members.weights to contain the number of times each vector
      is used in the image approximation.
    """
    supercluster = get_image_supercluster(
        supercluster_type, path, ignore_cache=ignore_cache
    )
    pixel_indices = _map_pixels_to_members_vectors(
        supercluster, path, ignore_cache=ignore_cache
    )
    return resample_image(supercluster, pixel_indices), pixel_indices


def _index_to_nearest_color(colormap: _Colors, colors: _Colors) -> _IndexMatrix:
    """Map an image to a colormap.

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
        return np.bincount(self.indices.flatten(), minlength=len(self.palette))


def _get_cache_paths(source: Path) -> dict[str, Path]:
    """Get the cache paths for the quantized image.

    :param source: path to an image
    :return: a dictionary with the cache paths for the quantized image
        * The keys are "palette", "indices", and "pmatrix"
        * The values are the path + stems to the cache files
    """
    attribs = ("palette", "indices", "pmatrix")
    return {a: CACHE_DIR.with_name(f"{_CACHE_PREFIX}_{a}.npy") for a in attribs}


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
        raise FileNotFoundError("Cache files not found.")
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


def quantize_image(source: Path, ignore_cache: bool = False) -> QuantizedImage:
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
    indices = _index_to_nearest_color(palette, rgb_colors).reshape(*image.size)
    pmatrix = get_delta_e_matrix(palette)

    quantized_image = QuantizedImage(palette, indices, pmatrix)
    _dump_quantized_image(quantized_image, source)
    return quantized_image


test_image = Path(__file__).parents[2] / "tests" / "resources" / "songs2.jpg"


if __name__ == "__main__":
    # Example usage
    quantize_image(test_image)
