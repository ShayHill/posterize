import functools
import logging
import subprocess
from operator import itemgetter
from pathlib import Path
from tempfile import TemporaryFile
from typing import Annotated, TypeAlias, TypeVar, cast
from posterize.image_processing import get_layer_color_index
import itertools as it
from operator import attrgetter


import numpy as np
from basic_colormath import floats_to_uint8
from cluster_colors.cluster_supercluster import SuperclusterBase, AgglomerativeSupercluster
from lxml import etree
from lxml.etree import _Element as EtreeElement  # type: ignore
from numpy import typing as npt
from PIL import Image
from PIL.Image import Image as ImageType
from svg_ultralight import new_element, new_svg_root, update_element, write_svg
from svg_ultralight.strings import svg_color_tuple
from posterize.image_processing import draw_posterized_image

from posterize import paths
from posterize.quantization import new_supercluster_with_quantized_image
from typing import Sequence, TypeVar, Union

logging.basicConfig(level=logging.INFO)

_IndexMatrix: TypeAlias = Annotated[npt.NDArray[np.int64], "(r,c)"]
_IndexMatrices: TypeAlias = Annotated[npt.NDArray[np.int64], "(n,r,c)"]
_IndexVector: TypeAlias = Annotated[npt.NDArray[np.int64], "(n,)"]
_IndexVectorLike: TypeAlias = _IndexVector | Sequence[int]
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


_TSuperclusterBase = TypeVar("_TSuperclusterBase", bound=SuperclusterBase)


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
        self.clusters, self.image = new_supercluster_with_quantized_image(
            Supercluster, path
        )
        self._bite_size = 9 if bite_size is None else bite_size

        self._state: _IndexMatrix | None = None
        self.layers = np.empty((0,) + self.image.shape[:2], dtype=int)

        # cache for each state
        self.__state_cost_matrix: _FPArray | None = None

    @property
    def state(self) -> _IndexMatrix:
        """Read cached state."""
        if self._state is None:
            msg = "State is undefined. No layers have been appended."
            raise ValueError(msg)
        return self._state

    @state.setter
    def state(self, state: _IndexMatrix):
        """Set a new state and reset all cached properties."""
        self._state = state
        self.__state_cost_matrix = None

    def get_cost_matrix(self, *layers: _IndexMatrix) -> _ErrorArray:
        """Get the cost-per-pixel between self.image and (state + layer).

        :param layers: layers to apply to the current state. There will only ever be
            0 or 1 layers. If 0, the cost matrix of the current state will be
            returned.  If 1, the cost matrix with a layer applied over it.
        :return: cost-per-pixel between image and (state + layer)

        This should decrease after every append.
        """
        if self._state is not None:
            layers = (self._state,) + layers
        if not layers:
            msg = "No state and no layers to apply."
            raise ValueError(msg)
        state_with_layers_applied = _merge_layers(*layers)
        if -1 in state_with_layers_applied:
            msg = "There are still transparent pixels in the state."
            raise ValueError(msg)
        return self.clusters.members.pmatrix[self.image, state_with_layers_applied]

    def get_cost(self, *layers: _IndexMatrix) -> float:
        """Get the cost between self.image and state with layer applied.

        :param layer: layers to apply to the current state. There will only ever be 1
            layer.
        :return: sum of the cost between image and (state + layer)
        """
        return float(np.sum(self.get_cost_matrix(*layers)))

    @property
    def state_cost_matrix(self) -> _ErrorArray:
        """Get the current cost-per-pixel between state and input.

        This should decrease after every append.
        """
        if self.__state_cost_matrix is None:
            self.__state_cost_matrix = self.get_cost_matrix()
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

        If there are no layers, the candidate will be a solid color.
        """
        solid = np.full(self.image.shape, palette_index)
        if self._state is None:
            return solid
        solid_cost_matrix = self.get_cost_matrix(solid)
        layer = np.full(self.image.shape, -1)
        layer[np.where(self.state_cost_matrix > solid_cost_matrix)] = palette_index
        return layer

    def get_state_with_layer_applied(self, layer: _IndexMatrix) -> _IndexMatrix:
        """Apply a layer to the current state."""
        if self._state is None:
            return layer
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
        """Get the vibrant colors in the image.

        :return: a set of palette indices that are considered vibrant

        These colors cannot be removed from `self._clusters.ixs` unless they are
        within `bite_size` of the cluster centroid. This prevents vibrant outliers
        from disappearing into less-vibrant clusters.

        Will take the 10% most vibrant colors in the image, but will exclude colors
        with vibrance < 64. This is to prevent muddy colors from being considered
        vibrant just because an image is mostly grayscale.
        """
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


def posterize(image_path: Path, bite_size: float | None = None, ixs: _IndexVectorLike | None = None) -> TargetImage:
    """Posterize an image.

    :param image_path: path to the image
    :param bite_size: the max average error of the cluster removed for each color
    :param ixs: optionally pass a subset of the palette indices to use
    :return: posterized image
    """
    target = TargetImage(image_path, bite_size)
    if ixs is not None:
        target.clusters = target.clusters.copy(inc_members=ixs)

    # set a background color
    colors = target.get_colors()
    candidates = [target.new_candidate_layer(x) for x in colors]
    scored = [(target.get_cost(x), x) for x in candidates]
    best = min(scored, key=itemgetter(0))[1]
    target.append_layer(best)

    # add layers
    count = 1
    while len(target.clusters.ixs) > 0:
        colors = target.get_colors()
        candidates = [target.new_candidate_layer(x) for x in colors]
        scored = [(target.get_cost(x), x) for x in candidates]
        best = min(scored, key=itemgetter(0))[1]
        target.append_layer(best)
        count += 1
    print(f"appended {count} layers")

    image_path = Path("temp.svg")
    clusters = target.clusters

    draw_posterized_image(
         clusters.members.vectors, target.layers, "temp_test"
    )
    return target

class SuperclusterMax(SuperclusterBase):
    """A SuperclusterBase that uses divisive clustering."""

    quality_metric = "max_error"
    quality_centroid = "weighted_medoid"
    assignment_centroid = "weighted_medoid"
    clustering_method = "divisive"

def contrast(colormap: _FPArray, ixs: _IndexVector) -> float:
    """Get the contrast between the colors in colormap at indices ixs.

    :param colormap: (r, 3) array of colors
    :param ixs: indices of colors in colormap
    :return: contrast between colors at ixs

    The contrast is the sum of the euclidean distances between all pairs of colors.
    """
    return float(np.sum(colormap[np.ix_(ixs, ixs)]))

def count_continuity(array):
    m, n = array.shape
    continuity_count = 0
    
    # Check horizontal continuity
    for i in range(m):
        continuity_count += count_sequences(array[i, :])

    # Check vertical continuity
    for j in range(n):
        continuity_count += count_sequences(array[:, j])

    # Check diagonal continuity (top-left to bottom-right)
    for k in range(-m + 1, n):
        diagonal = np.diagonal(array, offset=k)
        continuity_count += count_sequences(diagonal)

    # Check anti-diagonal continuity (top-right to bottom-left)
    for k in range(-m + 1, n):
        diagonal = np.diagonal(np.fliplr(array), offset=k)
        continuity_count += count_sequences(diagonal)
    
    return continuity_count

def count_sequences(arr):
    """Count the total length of consecutive 1s in a 1D array."""
    count = 0
    length = 0
    for value in arr:
        if value == 1:
            length += 1
        else:
            if length > 1:  # Only interested in sequences longer than 1
                count += length
            length = 0
    # If array ends with a sequence of 1s
    if length > 1:
        count += length
    return count


def posterize_to_n_colors(image_path: Path, num_cols: int, bite_size: float | None = None, ixs: _IndexVectorLike | None = None) -> _IndexMatrices:
    target = posterize(image_path, bite_size, ixs)
    # breakpoint()
    colors = [get_layer_color_index(x) for x in target.layers]
    continuity = [count_continuity(x) for x in target.layers]
    breakpoint()
    clusters, _ = new_supercluster_with_quantized_image(
        AgglomerativeSupercluster, image_path
    )

    clusters.ixs = np.array(colors)
    clusters.set_n(num_cols)
    pals = it.product(*map(attrgetter("ixs"), clusters.clusters))
    breakpoint()
    pick = max(pals, key=lambda x: contrast(clusters.members.pmatrix, x))

    # breakpoint()

    # vibrant = set(target.vibrant_colors)
    # pick: set[int] = set()
    # while len(pick) < num_cols:
    #     clu = clusters.clusters
    #     new_picks = {x.centroid for x in clu if x.centroid in vibrant}
    #     if new_picks:
    #         clusters = clusters.copy(exc_member_clusters=new_picks)
    #         pick.update(new_picks)
    #         continue
    #     new_picks = {x.centroid for x in clu if not vibrant & set(x.ixs)}
    #     if new_picks:
    #         clusters = clusters.copy(exc_member_clusters=new_picks)
    #         pick.update(new_picks)
    #         continue
    #     clusters.split()
    layers =  posterize(image_path, 0, list(pick))




    

if __name__ == "__main__":
    image_path = paths.PROJECT / "tests/resources/cafe_at_arles.jpg"
    # _ = posterize(image_path, bite_size=9)
    _ = posterize_to_n_colors(image_path, bite_size=9, num_cols=6)
    print("done")
