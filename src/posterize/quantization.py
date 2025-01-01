"""Replace colors with colormap indices.

A colormap has four associated properties:

* `supercluster.members.vectors` is a (n, 3) array of RGB vectors.
* `supercluster.members.weights` is a (n,) array of weights.
* `supercluster.members.pmatrix` is a (n, n) array of pairwise distances.
* `supercluster.ixs` is a (m,) array of indices to `.vectors`. After the operations
  here, this array may only contain indices to a subset of the original `.vectors`.

So, to capture all required information related to a colormap, these functions return
a `SuperclusterBase` instance with an `.ixs` attribute containing the subset of
indices used and `.members` attribute containing a `Members` instance with vectors,
weights, and pmatrix.

To complete the entire process:

* Create an initial SuperclusterBase instance from an input image.
* Map the image pixel colors to `supercluster.members.vectors`.
* Update `supercluster.ixs` to contain indices to only the vectors used in the image
  approximation.
* Update `supercluster.members.weights` to contain the number of times each vector is
  used in the image approximation.

The "updates" are done by creating new SuperclusterBase and Members instances.

:author: Shay Hill
:created: 2024-10-13
"""

from pathlib import Path
from typing import Annotated, TypeAlias, TypeVar

import numpy as np
from basic_colormath import get_sqeuclidean_matrix
from cluster_colors import Members, SuperclusterBase, get_image_supercluster
from cluster_colors.cluster_supercluster import SuperclusterBase
from numpy import typing as npt
from PIL import Image

from posterize.paths import CACHE_DIR

_IndexMatrix: TypeAlias = Annotated[npt.NDArray[np.int64], "(r,c)"]
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
    print(f"Quantizing {path.name} to {supercluster_type.__name__}")
    supercluster = get_image_supercluster(
        supercluster_type, path, ignore_cache=ignore_cache
    )
    print(f"Quantizing complete: {path.name} to {supercluster_type.__name__}")
    pixel_indices = _map_pixels_to_members_vectors(
        supercluster, path, ignore_cache=ignore_cache
    )
    return resample_image(supercluster, pixel_indices), pixel_indices
