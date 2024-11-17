import functools
import logging
from operator import itemgetter
from pathlib import Path
from typing import Annotated, Iterable, Sequence, TypeAlias, TypeVar, cast, Iterator
import pickle
import numpy as np
from basic_colormath import get_delta_e
from cluster_colors import SuperclusterBase
from lxml.etree import _Element as EtreeElement  # type: ignore
from numpy import typing as npt

from posterize import paths
from posterize.convolution import shrink_mask
from posterize.image_processing import draw_posterized_image
from posterize.quantization import new_supercluster_with_quantized_image

logging.basicConfig(level=logging.INFO)

# an image-sized array of -1 where transparent and palette indices where opaque
_IndexMatrix: TypeAlias = Annotated[npt.NDArray[np.int64], "(r,c)"]
_IndexMatrices: TypeAlias = Annotated[npt.NDArray[np.int64], "(n,r,c)"]

_IndexVector: TypeAlias = Annotated[npt.NDArray[np.int64], "(n,)"]
_IndexVectorLike: TypeAlias = _IndexVector | Sequence[int]
_LabArray: TypeAlias = Annotated[npt.NDArray[np.float64], "(n,m,3)"]
_MonoPixelArray: TypeAlias = Annotated[npt.NDArray[np.uint8], "(n,m,1)"]
_ErrorArray: TypeAlias = Annotated[npt.NDArray[np.float64], "(n,m,1)"]
_RGBTuple = tuple[int, int, int]
_FPArray = npt.NDArray[np.float64]


class Supercluster(SuperclusterBase):
    """A SuperclusterBase that uses divisive clustering."""

    quality_metric = "avg_error"
    quality_centroid = "weighted_medoid"
    assignment_centroid = "weighted_medoid"
    clustering_method = "divisive"


# _TSuperclusterBase = TypeVar("_TSuperclusterBase", bound=SuperclusterBase)


def _merge_layers(*layers: _IndexMatrix) -> _IndexMatrix:
    """Merge layers into a single layer.

    :param layers: (r, c) arrays with non-negative integers (palette indices) in
        opaque pixels and -1 in transparent
    :return: one (r, c) array with the last non-transparent pixel in each position
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
        :param bite_size: the max error of the cluster removed for each color. If the
            cluster is vibrant, max error will be limited to bite_size. If the
            cluster is not vibrant, average error will be limited to bite_size.
        """
        self._path = path
        self._bite_size = 9 if bite_size is None else bite_size


        self.clusters, self.image = new_supercluster_with_quantized_image(
            Supercluster, path
        )

        # initialize cached properties
        self._layers = np.empty((0,) + self.image.shape[:2], dtype=int)
        self.state = np.ones_like(self.image) * -1
        self.state_cost_matrix = np.ones_like(self.image) * np.inf

    @property
    def layers(self) -> _IndexMatrices:
        return self._layers

    @layers.setter
    def layers(self, value: _IndexMatrices) -> None:
        self._layers = value
        self.state = _merge_layers(*value)
        self.state_cost_matrix = self._get_cost_matrix(self.state)

    @functools.cached_property
    def cache_stem(self) -> str:
        cache_bite_size = f"{self._bite_size:05.2f}".replace(".", "_")
        return f"{self._path.stem}-{cache_bite_size}"

    def _get_cost_matrix(self, *layers: _IndexMatrix) -> _ErrorArray:
        """Get the cost-per-pixel between self.image and (state + layers).

        :param layers: layers to apply to the current state. There will only ever be
            0 or 1 layers. If 0, the cost matrix of the current state will be
            returned.  If 1, the cost matrix with a layer applied over it.
        :return: cost-per-pixel between image and (state + layer)

        This should decrease after every append.
        """
        state_with_layers_applied = _merge_layers(self.state, *layers)
        if -1 in state_with_layers_applied:
            msg = "There are still transparent pixels in the state."
            raise ValueError(msg)
        return self.clusters.members.pmatrix[self.image, state_with_layers_applied]

    def get_cost(self, *layers: _IndexMatrix) -> float:
        """Get the cost between self.image and state with layers applied.

        :param layers: layers to apply to the current state. There will only ever be
            one layer.
        :return: sum of the cost between image and (state + layer)
        """
        if layers:
            return float(np.sum(self._get_cost_matrix(*layers)))
        return self.state_cost

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
        if self.layers.shape[0] == 0:
            return solid
        solid_cost_matrix = self._get_cost_matrix(solid)
        layer = np.full(self.image.shape, -1)
        layer[np.where(self.state_cost_matrix > solid_cost_matrix)] = palette_index
        return layer

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

        :return: the rough color clusters in the image
        """
        self.clusters.set_max_avg_error(self._bite_size)
        print(f"generated {len(self.clusters.clusters)} colors")
        return [x.centroid for x in self.clusters.clusters]

    def get_best_candidate(self) -> _IndexMatrix:
        """Get the best candidate layer.

        :return: the candidate layer with the lowest cost
        """
        candidates = map(self.new_candidate_layer, self.get_colors())
        scored = ((self.get_cost(x), x) for x in candidates)
        return min(scored, key=itemgetter(0))[1]


def pick_nearest_color(
    rgb: _RGBTuple, colormap: _FPArray, colors: Iterable[int]
) -> int:
    """Pick the nearest color in colormap to rgb.

    :param rgb: (r, g, b) tuple
    :param colormap: (r, 3) array of colors
    :param colors: indices of colors in colormap to consider
    :return: index of the nearest color in colormap to rgb
    """
    return min(colors, key=lambda x: get_delta_e(rgb, colormap[x]))


def _draw_target(target: TargetImage, num_cols: int | None = None):
    """Infer a name from TargetImage args and write image to file.

    This is for debugging how well image is visually represented and what colors
    might be "eating" others in the image.
    """
    # vectors = target.clusters.members.vectors
    # stem_parts = (target.cache_stem, len(target.clusters.ixs), num_cols)
    # output_stem = "-".join(_stemize(*stem_parts))
    # draw_posterized_image(vectors, target.layers[:num_cols], output_stem)

def _stemize(*args: Path | float | int | str | None) -> Iterator[str]:
    """Convert args to strings and filter out empty strings."""
    if not args:
        return
    if args[0] is None:
        yield from _stemize(*args[1:])
    elif isinstance(args[0], str):
        yield args[0]
        yield from _stemize(*args[1:])
    elif isinstance(args[0], Path):
        yield args[0].stem
        yield from _stemize(*args[1:])
    elif isinstance(args[0], float):
        yield f"{args[0]:05.2f}".replace(".", "_")
        yield from _stemize(*args[1:])
    else:
        assert isinstance(args[0], int)
        yield f"{args[0]:03d}"
        yield from _stemize(*args[1:])

def _new_cache_path(*args: Path | float | int | str | None, suffix: str) -> Path:
    stem = '-'.join(_stemize(*args))
    return (paths.CACHE_DIR / stem).with_suffix(suffix)

def posterize(
    image_path: Path,
    bite_size: float | None = None,
    ixs: _IndexVectorLike | None = None,
    num_cols: int | None = None,
    *,
    ignore_cache: bool = True,
) -> TargetImage:
    """Posterize an image.

    :param image_path: path to the image
    :param bite_size: the max average error of the cluster removed for each color
    :param ixs: optionally pass a subset of the palette indices to use
    :return: posterized image
    """
    target = TargetImage(image_path, bite_size, ignore_cache=ignore_cache)

    cache_path = _new_cache_path(image_path, bite_size, num_cols, suffix=".npy")
    if cache_path.exists() and not ignore_cache:
        target.layers = np.load(cache_path)
        return target

    if ixs:
        target.clusters = target.clusters.copy(inc_members=ixs)
    while len(target.clusters.ixs) > 0:
        target.append_layer(target.get_best_candidate())
    print(f"appended {len(target.layers)} layers")

    _draw_target(target, num_cols)

    np.save(cache_path, target.layers)

    return target




def _compress_to_n_colors(
    target: TargetImage, net_cols: int, gross_cols: int | None = None
):
    state_at_n = _merge_layers(*target.layers[:gross_cols])


def posterize_to_n_colors(
    image_path: Path,
    ixs: _IndexVectorLike | None = None,
    bite_size: float | None = None,
    num_cols: int,
    skip_cols: list[tuple[float, float, float]] = [],
    skip_fields: list[tuple[float, float, float]] = [],
) -> _IndexMatrices:

    ixs_ = () if ixs is None else tuple(ixs)

    # draw the image with all colors as a visual aid
    target = posterize(image_path, bite_size, (), num_cols, ignore_cache=False)
    state_copy = target.state.copy()

    # dump skipped fields
    f_skip = {
        pick_nearest_color(x, target.clusters.members.vectors, np.unique(target.state))
        for x in skip_fields
    }

    net_colors: list[int] = []
    state_at_n = _merge_layers(*target.layers[:num_cols])
    for gross_cols in range(num_cols + 1, 512):
        net_colors = [x for x in np.unique(state_at_n) if x not in f_skip]
        if len(net_colors) == num_cols:
            break
        state_at_n = _merge_layers(*target.layers[:gross_cols])

    masks = [np.where(state_at_n == x, 1, 0) for x in net_colors]

    pick: list[int] = []

    c_skip = {
        pick_nearest_color(x, target.clusters.members.vectors, np.unique(target.state))
        for x in skip_cols
    }

    for mask in (shrink_mask(m, 0) for m in masks):
        masked = state_copy[np.where(mask == 1)]
        pmatrix = target.clusters.members.pmatrix
        image = target.image

        def get_cost(ix):
            r, g, b = target.clusters.members.vectors[ix]
            min_c = min(r, g, b)
            max_c = max(r, g, b)
            return np.max(pmatrix[ix, image[np.where(mask)]])
            return np.sum(pmatrix[ix, image[np.where(mask)]])

        # if c_skip & set(np.unique(masked)):
        #     breakpoint()

        cols = set(np.unique(masked)) - c_skip
        cols = cols or (np.unique(masked))

        pick.append(min(cols, key=get_cost))
        # pick.append(supercluster.clusters[0].centroid)

    # pick = [max(x.ixs, key=lambda x: supercluster.members.weights[x]) for x in supercluster.clusters]
    layers = posterize(image_path, 0, tuple(pick))


if __name__ == "__main__":
    image_path = paths.PROJECT / "tests/resources/manet.jpg"
    # _ = posterize(image_path, bite_size=9)
    _ = posterize_to_n_colors(
        image_path,
        bite_size=9,
        ixs = (),
        num_cols=6,
        # skip_cols=[(123, 116, 106), (126, 103, 101)],
    )
    print("done")
