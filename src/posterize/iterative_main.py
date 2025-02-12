"""Create a palette with backtracking after each color choicee

ehis is an edit of fast_main, but it's too much of a departure to create without
tearing fast_main apart.

:author: Shay Hill
:created: 2025-02-06
"""

import functools
import logging
from operator import itemgetter
import numpy as np
from pathlib import Path
from typing import Annotated, Iterator

from palette_image.svg_display import write_palette
from palette_image.color_block_ops import sliver_color_blocks

import numpy as np
from basic_colormath import (
    rgb_to_hsv,
    get_delta_e_lab,
    get_deltas_e,
    rgbs_to_hsv,
)
from cluster_colors import SuperclusterBase, Members
from lxml.etree import _Element as EtreeElement  # type: ignore
from numpy import typing as npt

from posterize import paths
from posterize.image_processing import draw_posterized_image
from posterize.quantization import new_supercluster_with_quantized_image

from typing import Any, Callable

logging.basicConfig(level=logging.INFO)


WHITE = (255, 255, 255)


class Supercluster(SuperclusterBase):
    """A SuperclusterBase that uses divisive clustering."""

    quality_metric = "avg_error"
    quality_centroid = "weighted_medoid"
    assignment_centroid = "weighted_medoid"
    clustering_method = "divisive"


# _TSuperclusterBase = TypeVar("_TSuperclusterBase", bound=SuperclusterBase)


def _merge_layers(*layers: npt.NDArray[np.integer[Any]]) -> npt.NDArray[np.integer[Any]]:
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

        self.ws = np.bincount(self.image.flatten(), minlength=self.pmatrix.shape[0])

        # initialize cached propertiej
        self._layers = np.empty((0, self.pmatrix.shape[0]), dtype=int)
        self.state = np.ones_like(self.image) * -1
        self.state_cost_matrix = np.ones_like(self.image) * np.inf

    @property
    def pmatrix(self) -> npt.NDArray[np.floating[Any]]:
        """Shorthand for self.clusters.members.pmatrix."""
        return self.clusters.members.pmatrix

    def get_state_weight(self, idx: int) -> float:
        """Get the weight of a color index in the current state."""
        return float(np.sum(self.ws[self.state == idx]))

    @property
    def layers(self) -> npt.NDArray[np.integer[Any]]:
        return self._layers

    @layers.setter
    def layers(self, value: npt.NDArray[np.integer[Any]]) -> None:
        self._layers = value
        self.state = _merge_layers(*value)
        self.state_cost_matrix = self._get_cost_matrix(self.state)

    @functools.cached_property
    def cache_stem(self) -> str:
        cache_bite_size = f"{self._bite_size:05.2f}".replace(".", "_")
        return f"{self._path.stem}-{cache_bite_size}"


    def _get_cost_matrix(self, *layers: npt.NDArray[np.integer[Any]]) -> npt.NDArray[np.floating[Any]]:
        """Get the cost-per-pixel between self.image and (state + layers).

        :param layers: layers to apply to the current state. There will only ever be
            0 or 1 layers. If 0, the cost matrix of the current state will be
            returned.  If 1, the cost matrix with a layer applied over it.
        :return: cost-per-pixel between image and (state + layer)

        This should decrease after every append.
        """
        state = _merge_layers(*layers)
        if -1 in state:
            msg = "There are still transparent pixels in the state."
            raise ValueError(msg)
        image = np.array(range(self.pmatrix.shape[0]), dtype=int)
        return self.pmatrix[image, state] * self.ws

    def get_cost(self, *layers: npt.NDArray[np.integer[Any]]) -> tuple[float, float]:
        """Get the cost between self.image and state with layers applied.

        :param layers: layers to apply to the current state. There will only ever be
            one layer.
        :return: sum of the cost between image and (state + layer)
        # TODO: stop returnin fallback
        """
        if not layers:
            raise ValueError("At least one layer is required.")
        cost_matrix = self._get_cost_matrix(*layers)
        primary = float(np.sum(cost_matrix))
        return primary, primary

    @property
    def state_cost(self) -> float:
        """Get the sum of the cost at the current state.

        Use this to guarantee any new element will improve the approximation.
        """
        return float(np.sum(self.state_cost_matrix))

    def new_candidate_layer(
        self, palette_index: int, state_layers: npt.NDArray[np.integer[Any]]
    ) -> npt.NDArray[np.integer[Any]]:
        """Create a new candidate state.

        :param palette_index: the index of the color to use in the new layer
        :param state_layers: the current state or a presumed state

        A candidate is the current state with state indices replaced with
        palette_index where palette_index has a lower cost that the index in state at
        that position.

        If there are no layers, the candidate will be a solid color.
        """
        solid = np.full(self.pmatrix.shape[0], palette_index)
        if len(state_layers) == 0:
            return solid

        solid_cost_matrix = self._get_cost_matrix(solid)
        state_cost_matrix = self._get_cost_matrix(*state_layers)

        layer = np.full(self.pmatrix.shape[0], -1)
        layer[np.where(state_cost_matrix > solid_cost_matrix)] = palette_index
        return layer

    def append_color(
        self, layers: npt.NDArray[np.integer[Any]], *palette_indices: int
    ) -> npt.NDArray[np.integer[Any]]:
        """Append a color to the current state.

        :param layers: the current state or a presumed state
        :param palette_indices: the index of the color to use in the new layer.
            Multiple args allowed.
        """
        if not palette_indices:
            return layers
        new_layer = self.new_candidate_layer(palette_indices[0], layers)
        layers = np.append(layers, [new_layer], axis=0)
        return self.append_color(layers, *palette_indices[1:])

    def _match_layer_color(
        self, layer_a: npt.NDArray[np.integer[Any]], layer_b: npt.NDArray[np.integer[Any]]
    ) -> npt.NDArray[np.integer[Any]]:
        """Match the color of layer_a to layer_b.

        :param layer_a: (r, c) array with a palette index in opaque pixels and -1 in
            transparent
        :param layer_b: (r, c) array with a palette index in opaque pixels and -1 in
            transparent
        :return: (r, c) array with the same shape as layer_a where the color of each
            pixel is matched to the color of the corresponding pixel in layer_b.
        """
        color = np.max(layer_b)
        return np.where(layer_a == -1, -1, color)

    def find_layer_substitute(
        self, layers: npt.NDArray[np.integer[Any]], index: int
    ) -> tuple[int, float]:
        """Find a substitute color for a layer.

        :param layers: (n, r, c) array with palette indices in opaque pixels and -1 in
            transparent
        :param index: the index of the layer to find a substitute for. This can only
            work if the index is less than len(layers) - 1.
        :return: the index of the substitute color and the delta E between the
            substitute and the original color
        """
        this_color = max(layers[index])
        with_layer_hidden = layers.copy()
        with_layer_hidden[index] = self._match_layer_color(
            layers[index], layers[index + 1]
        )
        candidate = self.get_best_candidate(with_layer_hidden)
        candidate_color = max(candidate)
        if this_color == candidate_color:
            return candidate_color, 0
        return candidate_color, float(self.pmatrix[this_color, candidate_color])

    def check_layers(
        self,
        layers: npt.NDArray[np.integer[Any]],
        num_layers: int | None = None,
        seen: dict[tuple[int, ...], float] | None = None,
    ) -> npt.NDArray[np.integer[Any]]:
        """Check that each layer is the same it would be if it were a candidate."""
        if num_layers is None:
            num_layers = len(layers)

        if len(layers) == num_layers:
            if seen is None:
                seen = {}
            print(f"checking {num_layers=} {len(seen)=}")
            for k in sorted(seen.keys()):
                print(f"        {k} {seen[k]}")
            key = tuple(int(np.max(x)) for x in layers)
            if key in seen:
                best_key = min(seen.items(), key=itemgetter(1))[0]
                if key == best_key:
                    return layers
                layers = layers[:0]
                layers = self.append_color(layers, *key)
                return layers

            seen[key] = self.get_cost(*layers)[0]
            print(f"     ++ {key} {seen[key]}")

        for i, _ in enumerate(layers[:-1]):
            new_color, delta_e = self.find_layer_substitute(layers, i)
            if delta_e > 0:
                print(f"color mismatch {i=} {len(layers)=}")
                layers = self.append_color(layers[:i], new_color)
                return self.check_layers(layers, num_layers, seen=seen)

        # this will only be true if no substitutes were found
        if num_layers == len(layers):
            return layers

        while len(layers) < num_layers:
            new_layer = self.get_best_candidate(layers)
            layers = np.append(layers, [new_layer], axis=0)

        return self.check_layers(layers, seen=seen)

    def append_layer_to_state(self, layer: npt.NDArray[np.integer[Any]]) -> None:
        """Append a layer to the current state.

        param layer: (m, n) array with a palette index in opaque pixels and -1 in
            transparent
        """
        self.layers = np.append(self.layers, [layer], axis=0)
        if len(self.layers) > 2:
            self.layers = self.check_layers(self.layers)

    def get_colors(self) -> list[int]:
        """Get the most common colors in the image.

        :return: the rough color clusters in the image
        """
        # self.clusters.set_max_avg_error(self._bite_size)
        return list(range(self.pmatrix.shape[0]))
        return [x.centroid for x in self.clusters.clusters]

    def get_best_candidate(
        self, state_layers: npt.NDArray[np.integer[Any]] | None = None
    ) -> npt.NDArray[np.integer[Any]]:
        """Get the best candidate layer.

        :return: the candidate layer with the lowest cost
        """
        if state_layers is None:
            state_layers = self.layers
        available_colors = self.get_colors()
        used = tuple(np.unique(state_layers))
        if used:
            available_colors = [
                x
                for x in available_colors
                if np.min(self.pmatrix[x, used]) > self._bite_size
            ]
        get_cand = functools.partial(
            self.new_candidate_layer, state_layers=state_layers
        )
        candidates = tuple(map(get_cand, available_colors))
        scored = tuple((self.get_cost(*state_layers, x), x) for x in candidates)
        winner = min(scored, key=itemgetter(0))[1]
        return winner


def _expand_layers(
    quantized_image: Annotated[npt.NDArray[np.integer[Any]], "(r, c)"],
    d1_layers: Annotated[npt.NDArray[np.integer[Any]], "(n, 512)"],
) -> Annotated[npt.NDArray[np.integer[Any]], "(n, r, c)"]:
    """Expand layers to the size of the quantized image.

    :param quantized_image: (r, c) array with palette indices
    :param d1_layers: (n, 512) an array of layers. Layers may contain -1 or any
        palette index in [0, 511].
    :return: (n, r, c) array of layers, each layer with the same shape as the
        quantized image.
    """
    return np.array([x[quantized_image] for x in d1_layers])


def draw_target(
    target: TargetImage, num_cols: int | None = None, stem: str = ""
) -> None:
    """Infer a name from TargetImage args and write image to file.

    This is for debugging how well image is visually represented and what colors
    might be "eating" others in the image.
    """
    vectors = target.clusters.members.vectors
    stem_parts = (target.cache_stem, len(target.layers), num_cols, stem)
    output_stem = "-".join(_stemize(*stem_parts))

    big_layers = _expand_layers(target.image, target.layers)
    draw_posterized_image(vectors, big_layers[:num_cols], output_stem)


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
    stem = "-".join(_stemize(*args))
    return (paths.CACHE_DIR / stem).with_suffix(suffix)


def posterize(
    image_path: Path,
    bite_size: float,
    ixs: npt.ArrayLike | None = None,
    num_cols: int | None = None,
    *,
    ignore_cache: bool = True,
) -> TargetImage:
    """Posterize an image.

    :param image_path: path to the image
    :param bite_size: the max average error of the cluster removed for each color
    :param ixs: optionally pass a subset of the palette indices to use. This is used
        with bite_size == 0 to generate an image with exactly the colors in ixs.
    :return: posterized image
    """
    ignore_cache = True
    print(f"{bite_size=}")

    target = TargetImage(image_path, bite_size)

    cache_path = _new_cache_path(image_path, bite_size, num_cols, suffix=".npy")
    if cache_path.exists() and not ignore_cache:
        target.layers = np.load(cache_path)
    else:
        # if ixs:
        #     target.clusters = target.clusters.copy(inc_members=ixs)

        while len(target.layers) < (num_cols or 1) and len(target.layers) < len(
            target.get_colors()
        ):
            target.append_layer_to_state(target.get_best_candidate())

    np.save(cache_path, target.layers)

    if len(target.layers) < (num_cols or 1):
        return posterize(
            image_path,
            max(bite_size - 1, 0),
            ixs,
            num_cols,
            ignore_cache=ignore_cache,
        )

    return target


