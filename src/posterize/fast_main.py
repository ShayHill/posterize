import numpy as np
from pathlib import Path
from typing import TypeAlias, Annotated, cast
from numpy import typing as npt
from posterize import paths
from tempfile import TemporaryFile
from svg_ultralight import new_element, update_element
from svg_ultralight.strings import svg_color_tuple
from basic_colormath import floats_to_uint8
import subprocess

import logging
from operator import itemgetter

from cluster_colors import EmptySuperclusterError
from cluster_colors import get_image_supercluster, stack_pool_cut_image_colors
from cluster_colors.cluster_supercluster import SuperclusterBase
from cluster_colors.cluster_cluster import Cluster
from cluster_colors.cut_colors import cut_colors
from cluster_colors.pool_colors import pool_colors
import basic_colormath as cm
from lxml import etree
from lxml.etree import _Element as EtreeElement  # type: ignore
from PIL import Image
from PIL.Image import Image as ImageType
from svg_ultralight import new_element, update_element, new_svg_root, write_svg
from svg_ultralight.strings import svg_color_tuple
from cluster_colors import get_image_supercluster, stack_pool_cut_image_colors
from basic_colormath import (
    get_delta_e,
    get_delta_e_matrix,
    floats_to_uint8,
    get_sqeuclidean,
    get_sqeuclidean_matrix,
)
import functools

import time

logging.basicConfig(level=logging.INFO)

_PixelMatrix: TypeAlias = Annotated[npt.NDArray[np.uint8], "(r,c,3)"]
_PixelVector: TypeAlias = Annotated[npt.NDArray[np.uint8], "(r,3)"]
_IndexMatrix: TypeAlias = Annotated[npt.NDArray[np.int64], "(r,c)"]
_LabArray: TypeAlias = Annotated[npt.NDArray[np.float64], "(n,m,3)"]
_MonoPixelArray: TypeAlias = Annotated[npt.NDArray[np.uint8], "(n,m,1)"]
_ErrorArray: TypeAlias = Annotated[npt.NDArray[np.float64], "(n,m,1)"]
_RGBTuple = tuple[int, int, int]
_FPArray = npt.NDArray[np.float64]

with TemporaryFile() as f:
    CACHE_DIR = Path(f.name).parent / "cluster_colors_cache"


class Supercluster(SuperclusterBase):
    """A SuperclusterBase that uses divisive clustering."""

    quality_metric = "avg_error"
    quality_centroid = "weighted_medoid"
    assignment_centroid = "weighted_medoid"
    clustering_method = "divisive"


def apply_colormap(
    path: Path, colormap: _FPArray, *, ignore_cache: bool = False
) -> _IndexMatrix:
    """Map an image to a colormap.

    :param path: path to the image
    :param colormap: colormap to map to
    :return: index matrix of the image mapped to the colormap
    """
    cache_path = CACHE_DIR / f"{path.stem}_colormapped.npy"
    if not ignore_cache and cache_path.exists():
        return np.load(cache_path)

    image = Image.open(path)
    image = image.convert("RGB")
    image_rgb_vector = np.array(image).astype(float).reshape(-1, 3)
    image_idx_vector = np.argmin(
        get_sqeuclidean_matrix(image_rgb_vector, colormap), axis=1
    )
    image_idx_matrix = image_idx_vector.reshape(image.size[1], image.size[0])

    np.save(cache_path, image_idx_matrix)
    return image_idx_matrix


def _merge_layers(*layers: _IndexMatrix) -> _IndexMatrix:
    """Merge layers into a single layer.

    :param layers: (m, n) arrays with an integer in opaque pixels and -1 in transparent
    :return: one (m, n) array with the last non-transparent pixel in each position
    """
    merged = layers[0].copy()
    for layer in layers[1:]:
        merged[np.where(layer != -1)] = layer[np.where(layer != -1)]
    return merged


class TargetImage:
    """A type to store input images, current state, and current cost."""

    def __init__(
        self,
        path: Path,
        bite_size: float | None = None,
        *,
        ignore_cache: bool = False,
    ) -> None:
        """Initialize a TargetImage.

        :param path: path to the image
        :param path_to_cache: path to the cache file or None. If given, will cache
            the root element only. Will refresh the cache with each append.
        """
        self.clusters = get_image_supercluster(Supercluster, path)
        self._bite_size = 9 if bite_size is None else bite_size
        colormap = self.clusters.members.vectors

        self.image = apply_colormap(path, colormap, ignore_cache=ignore_cache)
        self._state: _IndexMatrix = np.zeros(self.image.shape[:2], dtype=int)
        self.layers = np.empty((0,) + self.image.shape[:2], dtype=int)

        # cache for each state
        self.__state_cost_matrix: _FPArray | None = None

    @property
    def state(self) -> _IndexMatrix:
        """Read cached state."""
        return self._state

    @state.setter
    def state(self, state: _IndexMatrix):
        """Set a new state and reset all cached properties."""
        self._state = state
        self.__state_cost_matrix = None

    def get_cost_matrix_with_layer(self, layer: _IndexMatrix) -> _ErrorArray:
        """Get the cost-per-pixel between self.image and (state + layer).

        :param layer: layer to apply to the current state
        :return: cost-per-pixel between image and (state + layer)

        This should decrease after every append.
        """
        with_layer_applied = self.get_state_with_layer_applied(layer)
        pmatrix = self.clusters.members.pmatrix
        return pmatrix[self.image, with_layer_applied]

    def get_cost_with_layer(self, layer: _IndexMatrix) -> float:
        """Get the cost between self.image and state with layer applied.

        :param layer: layer to apply to the current state
        :return: sum of the cost between image and (state + layer)
        """
        return float(np.sum(self.get_cost_matrix_with_layer(layer)))

    @property
    def state_cost_matrix(self) -> _ErrorArray:
        """Get the current cost-per-pixel between state and input.

        This should decrease after every append.
        """
        if self.__state_cost_matrix is None:
            pmatrix = self.clusters.members.pmatrix
            self.__state_cost_matrix = pmatrix[self.image, self.state]
        return self.__state_cost_matrix

    @property
    def state_cost(self) -> float:
        """Get the sum of the cost at the current state.

        Use this to guarantee any new element will improve the approximation.
        """
        return float(np.sum(self.state_cost_matrix))

    def new_candidate_layer(self, palette_index: int) -> _IndexMatrix:
        """Create a new candidate state.

        A candidate is the current state with state indices replaced with
        palette_index where palette_index has a lower cost that the index in state at
        that position.
        """
        solid = np.full(self.image.shape, palette_index)
        solid_cost_matrix = self.get_cost_matrix_with_layer(solid)
        layer = np.full(self.image.shape, -1)
        layer[np.where(self.state_cost_matrix > solid_cost_matrix)] = palette_index
        return layer

    def new_background_candidate_layer(self, palette_index: int) -> _IndexMatrix:
        """Create a new candidate for a background color.

        This is only a method to simplify accessing self.state.shape. The candidate
        will be a solid color despite what costs this may or may not improve.
        """
        return np.full(self.image.shape, palette_index)

    def get_state_with_layer_applied(self, layer: _IndexMatrix) -> _IndexMatrix:
        """Apply a layer to the current state."""
        return _merge_layers(self.state, layer)

    def _split_to_exclude_vibrant(self, palette_index: int):
        """Exclude vibrant colors from a cluster.

        Split clusters until the cluster containing palette_index does not contain
        any vibrant outliers. This protects vibrant colors from disappearing into
        less-vibrant clusters.
        """
        if palette_index in self.vibrant_colors:
            return

        pmatrix = self.clusters.members.pmatrix
        while True:
            cluster_ix = self.clusters.find_member(palette_index)
            cluster = self.clusters.clusters[cluster_ix]
            vibrant_in_cluster = list(self.vibrant_colors & set(cluster.ixs))
            if not vibrant_in_cluster:
                break
            vibrant_in_cluster_proximities = pmatrix[palette_index, vibrant_in_cluster]
            if max(vibrant_in_cluster_proximities) < self._bite_size:
                break
            self.clusters.split()

    def append_layer(self, layer: _IndexMatrix) -> None:
        """Append a layer to the current state.

        :param layer: (m, n) array with a palette index in opaque pixels and -1 in
            transparent
        """
        self.layers = np.append(self.layers, [layer], axis=0)

        layer_color = max(np.unique(layer))
        assert layer_color != -1

        self.state = self.get_state_with_layer_applied(layer)

        self.clusters.set_max_avg_error(self._bite_size)
        self._split_to_exclude_vibrant(layer_color)
        self.clusters = self.clusters.copy(exc_member_clusters=[layer_color])
        print(f"####### {len(self.clusters.ixs)} members remain")

    @functools.cached_property
    def vibrant_colors(self) -> set[int]:
        vectors = self.clusters.members.vectors
        v_max = cast(npt.NDArray[np.float64], np.max(vectors, axis=1))
        v_min = cast(npt.NDArray[np.float64], np.min(vectors, axis=1))
        v_vib = v_max - v_min
        threshold = max(float(np.percentile(v_vib, 90)), 64)
        vibrant = np.where(v_vib >= threshold)[0]
        print(f"vibrant colors: {len(vibrant)}")
        return set(map(int, vibrant))

    def get_colors(self) -> list[int]:
        """Get the most common colors in the image.

        :param num: number of colors to get
        :return: the num+1 most common colors in the image

        Return one color that represents the entire self.custers plus the exemplar of
        cluster after splitting to at most num clusters.
        """
        self.clusters.set_max_avg_error(16)
        # max_w = max(self._clusters.clusters, key=lambda x: x.weight)
        print(f"generated {len(self.clusters.clusters)} colors")
        # return [max_w.centroid]

        return [x.centroid for x in self.clusters.clusters]


def _write_svg(mono: Path) -> Path:
    """Create an svg for a given illumination.

    :param lux: illumination level
    :return: path to the output svg
    """
    # svg_path = self._file_paths.get_tmp_svg(lux)
    svg_path = (paths.CACHE / mono.name).with_suffix(".svg")
    command = [
        str(paths.POTRACE),
        str(mono),
        "-o",
        str(svg_path),
        "-k",
        str(0.5),
        "-u",
        "1",  # do not scale svg (points correspond to pixels array)
        "--flat",  # all paths combined in one element
        # "-t", str(self._tsize),  # remove speckles
        "-b",
        "svg",  # output format
        "--opttolerance",
        "2.8",  # higher values make paths smoother
    ]
    # fmt: on
    _ = subprocess.run(command, check=True)
    return svg_path


def draw_posterized_image(
    filename: str, layers: _IndexMatrix, colormap: _PixelVector
) -> ImageType:
    """Draw a posterized image.

    :param filename: path to the image
    :param layers: layers to draw
    :return: posterized image
    """

    bg_col = colormap[layers[0][0][0]]
    height, width = layers[0].shape
    bg_geo = new_element(
        "rect", x=0, y=0, width=width, height=height, fill=svg_color_tuple(bg_col)
    )
    root = new_svg_root(x_=0, y_=0, width_=width, height_=height)
    root.append(bg_geo)
    for i, layer in enumerate(layers[1:], 1):
        bmp_path = paths.CACHE / f"{filename}_{i}.bmp"
        cols = np.unique(layer)
        assert len(cols) == 2
        elem_color = max(cols)
        mono_pixels = np.ones([*layer.shape, 3], dtype=np.uint8) * 255
        mono_pixels[np.where(layer != -1)] = (0, 0, 0)
        mono_bmp = Image.fromarray(mono_pixels)
        mono_bmp.save(bmp_path)
        mono_svg = _write_svg(bmp_path)
        elem = etree.parse(str(mono_svg)).getroot()[1]
        _ = update_element(elem, fill=svg_color_tuple(colormap[elem_color]))
        root.append(elem)
    svg_path = paths.WORKING / f"{filename}.svg"
    _ = write_svg(svg_path, root)


def posterize(image_path: Path, bite_size: float | None = None) -> _IndexMatrix:
    """Posterize an image.

    :param image_path: path to the image
    :return: posterized image
    """
    target = TargetImage(image_path, bite_size)

    # set a background color
    colors = target.get_colors()
    candidates = [target.new_background_candidate_layer(x) for x in colors]
    scored = [(target.get_cost_with_layer(x), x) for x in candidates]
    best = min(scored, key=itemgetter(0))[1]
    target.append_layer(best)

    # add layers
    count = 1
    while len(target.clusters.ixs) > 0:
        colors = target.get_colors()
        candidates = [target.new_candidate_layer(x) for x in colors]
        scored = [(target.get_cost_with_layer(x), x) for x in candidates]
        best = min(scored, key=itemgetter(0))[1]
        target.append_layer(best)
        count += 1
    print(f"appended {count} layers")

    image_path = Path("temp.svg")
    clusters = target.clusters
    aaa = clusters.members.vectors

    _ = draw_posterized_image(
        "temp_test", target.layers, floats_to_uint8(clusters.members.vectors)
    )

    return target.state


if __name__ == "__main__":
    posterize(paths.PROJECT / "tests/resources/adidas.jpg", bite_size=25.2)
