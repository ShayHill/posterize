"""Create a palette with backtracking after each color choice.

Starting from a quantized image, select colors, one by one, that best approximate
that image. The first color will be a solid layer. Additional layers will have a
single color index where that color would improve the image, and -1 where it would
not.

After layers are added, check the original layers and update their colors if it
improves the approximation. To understand how this would improve an approximation,
imagine a Japanese flag. The algorithm would begin by selecting a color that
minimizes the error across the entire image. This would be some shade of pink. The
second layer would only cover part of the image and would be white or red. Let's say
it's red. Once red is added, the underlying pink layer is no longer responsible for
approximating the red circle in the middle of the flag. So the pink layer can be
updated to white. This saves a third white layer which, combined with the red layer,
would completely cover the pink layer anyway.

:author: Shay Hill
:created: 2025-02-06
"""

from __future__ import annotations

import functools

import logging
from operator import itemgetter
import numpy as np
from pathlib import Path
from typing import Annotated, Iterator, TypeAlias, Iterable, Container


import numpy as np
from cluster_colors import SuperclusterBase
from lxml.etree import _Element as EtreeElement  # type: ignore
from numpy import typing as npt
import dataclasses
from posterize import paths
from posterize.image_processing import draw_posterized_image
from posterize.quantization import new_supercluster_with_quantized_image

from typing import Any

logging.basicConfig(level=logging.INFO)


IntA: TypeAlias = npt.NDArray[np.integer[Any]]
Ints: TypeAlias = npt.ArrayLike

FltA: TypeAlias = npt.NDArray[np.floating[Any]]
Flts: TypeAlias = npt.ArrayLike


class ColorsExhaustedError(Exception):
    """Exception raised when a new layer is requested, but no colors are available."""

    def __init__(self, message: str = "No available colors.") -> None:
        self.message = message
        super().__init__(self.message)


class Supercluster(SuperclusterBase):
    """A SuperclusterBase that uses divisive clustering."""

    quality_metric = "avg_error"
    quality_centroid = "weighted_medoid"
    assignment_centroid = "weighted_medoid"
    clustering_method = "divisive"


@dataclasses.dataclass
class Layers:
    """State for an image approximation.

    :param colors: the subset of color indices (TargetImage.clusters.ixs) available
        for use in layers.
    :param layers: (n, c) array of n layers, each containing a value (color index) in
        colors and -1 for transparent. The first layer will be a solid color and
        contain no -1 values.
    """

    colors: set[int]
    min_delta: float

    def __init__(
        self,
        colors: Iterable[int],
        min_delta: float,
        layers: IntA | None = None,
    ) -> None:
        self.colors = set(colors)
        self.min_delta = min_delta
        if layers is None:
            self.layers = np.empty((0, 512), dtype=int)
        else:
            self.layers = layers
        self.cached_states: dict[tuple[int, ...], float] = {}

    @property
    def layer_colors(self) -> list[int]:
        """Get the non-transparent color in each layer."""
        return [np.max(x) for x in self.layers]

    def get_layer_color(self, index: int) -> int:
        """Get the color of a layer."""
        return np.max(self.layers[index])

    def with_layer_hidden(self, index: int) -> Layers:
        """Hide a layer by matching its color to the next layer."""
        if len(self.layers) == 1:
            msg = "Cannot hide a single layer."
            raise ValueError(msg)
        if index >= len(self.layers) - 1:
            msg = "Cannot hide the last layer."
            raise ValueError(msg)
        if index < 0:
            msg = "Cannot use negative index to hide a layer."
            raise ValueError(msg)
        with_hidden = self.layers.copy()
        mask_color = self.get_layer_color(index + 1)
        with_hidden[index] = np.where(self.layers[index] == -1, -1, mask_color)
        return Layers(self.colors, self.min_delta, with_hidden)


def _merge_layers(
    *layers: IntA,
) -> IntA:
    """Merge layers into a single layer.

    :param layers: (n, c) array of n layers, each containing a value (color index) for each color
        index. These will all be the same color or -1 for colors that are transparent
        in each layer.
    :return: one (c,) array with the last non-transparent color in each position

    Where an image is a (rows, cols) array of indices--usually in (0, 511)--each
    layer of an approximation will color some of those indices with one palette index
    per layer, and others with -1 for transparency.
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
    ) -> None:
        """Initialize a TargetImage.

        :param path: path to the image
        """
        self._path = path
        self.clusters, self.image = new_supercluster_with_quantized_image(
            Supercluster, path
        )

    @property
    def vectors(self) -> npt.NDArray[np.floating[Any]]:
        """Shorthand for self.clusters.members.vectors."""
        return self.clusters.members.vectors

    @property
    def pmatrix(self) -> npt.NDArray[np.floating[Any]]:
        """Shorthand for self.clusters.members.pmatrix."""
        return self.clusters.members.pmatrix

    @property
    def weights(self) -> npt.NDArray[np.floating[Any]]:
        """Shorthand for self.clusters.members.weights."""
        return self.clusters.members.weights

    def get_state_weight(self, state: Layers, idx: int) -> float:
        """Get the weight of a color index in the current state."""
        state_image = _merge_layers(*state.layers)
        return float(np.sum(self.weights[state_image == idx]))

    # def get_cache_stem(self, state: Layers) -> str:
    #     min_delta = f"{state.min_delta:05.2f}".replace(".", "_")
    #     return f"{self._path.stem}-{min_delta}"

    def _get_cost_matrix(self, *layers: IntA) -> npt.NDArray[np.floating[Any]]:
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
        return self.pmatrix[image, state] * self.weights

    def get_cost(self, *layers: IntA) -> float:
        """Get the cost between self.image and state with layers applied.

        :param layers: layers to apply to the current state. There will only ever be
            one layer.
        :return: sum of the cost between image and (state + layer)
        """
        if not layers:
            raise ValueError("At least one layer is required.")
        cost_matrix = self._get_cost_matrix(*layers)
        return float(np.sum(cost_matrix))

    def new_candidate_layer(self, state: Layers, palette_index: int) -> IntA:
        """Create a new candidate state.

        :param palette_index: the index of the color to use in the new layer
        :param state_layers: the current state or a presumed state

        A candidate is the current state with state indices replaced with
        palette_index where palette_index has a lower cost that the index in state at
        that position.

        If there are no layers, the candidate will be a solid color.
        """
        solid = np.full(self.pmatrix.shape[0], palette_index)
        if len(state.layers) == 0:
            return solid

        solid_cost_matrix = self._get_cost_matrix(solid)
        state_cost_matrix = self._get_cost_matrix(*state.layers)

        layer = np.full(self.pmatrix.shape[0], -1)
        layer[np.where(state_cost_matrix > solid_cost_matrix)] = palette_index
        return layer

    def append_color(self, state: Layers, *palette_indices: int):
        """Append a color to the current state.

        :param layers: the current state or a presumed state
        :param palette_indices: the index of the color to use in the new layer.
            Multiple args allowed.
        """
        if not palette_indices:
            return
        new_layer = self.new_candidate_layer(state, palette_indices[0])
        state.layers = np.append(state.layers, [new_layer], axis=0)
        self.append_color(state, *palette_indices[1:])

    def find_layer_substitute(self, state: Layers, index: int) -> tuple[int, float]:
        """Find a substitute color for a layer.

        :param state: Layers instance
        :param index: the index of the layer to find a substitute for. This can only
            work if the index is less than len(layers) - 1.
        :return: the index of the substitute color and the delta between the
            substitute and the original color
        """
        this_color = max(state.layers[index])
        candidate = self.get_best_candidate(state.with_layer_hidden(index))
        candidate_color = max(candidate)
        if this_color == candidate_color:
            return candidate_color, 0
        return candidate_color, float(self.pmatrix[this_color, candidate_color])

    def _fill_layers(self, state: Layers, num_layers: int):
        """Add layers (without check_layers) until there are num_layers.

        :param num_layers: the number of layers to add
        :param layers: the current state or a presumed state
        :return: layers with num_layers. This does not alter the state.
        """
        while len(state.layers) < num_layers:
            new_layer = self.get_best_candidate(state)
            state.layers = np.append(state.layers, [new_layer], axis=0)

    def check_layers(self, state: Layers):
        key = tuple(map(int, state.layer_colors))
        if key in state.cached_states:
            candidates = (
                x for x in state.cached_states.items() if len(x[0]) == len(key)
            )
            best = min(candidates, key=itemgetter(1))[0]
            state.layers = state.layers[:0]
            self.append_color(state, *best)
            return

        state.cached_states[key] = self.get_cost(*state.layers)
        at: int = 0
        while at < len(state.layers) - 1:
            new_color, delta_e = self.find_layer_substitute(state, at)
            if delta_e == 0:  # no benefit to changing color
                at += 1
                continue

            print(f"color mismatch {at=} {len(state.layers)=}")
            state.layers = state.layers[:at]
            self.append_color(state, new_color)
            self._fill_layers(state, len(key))
            self.check_layers(state)
            return
            at = 0

    def fill_layers(self, state: Layers, num_layers: int):
        """Add layers until there are num_layers. Then check lower layers.

        :param state: the Layers instance to be updated.
        :param num_layers: the number of layers to end up with. Will silently return
            fewer layers if all colors are exhausted.
        :effect: update state.layers
        """
        if len(state.layers) == num_layers:
            self.check_layers(state)
            return
        try:
            self._fill_layers(state, len(state.layers) + 1)
            self.check_layers(state)
            self.fill_layers(state, num_layers)
        except ColorsExhaustedError:
            self.fill_layers(state, num_layers - 1)

    # def append_layer(self, state: Layers, layer: IntA) -> None:
    #     """Append a layer to the current state.

    #     param layer: (m, n) array with a palette index in opaque pixels and -1 in
    #         transparent
    #     """
    #     state.layers = np.append(state.layers, [layer], axis=0)
    #     # if len(state.layers) > 2:
    #     # self.layers = self.check_layers(state.layers)

    def get_colors(self, state: Layers) -> set[int]:
        """Get available colors in the image."""
        if len(state.layers) == 0:
            return state.colors
        layer_colors = state.layer_colors
        assert -1 not in state.layer_colors
        return {
            int(x)
            for x in state.colors
            if min(self.pmatrix[x, layer_colors]) > state.min_delta
        }

    def get_best_candidate(self, state: Layers) -> IntA:
        """Get the best candidate layer to add to layers.

        :param state_layers: the current state or a presumed state
        :return: the candidate layer with the lowest cost
        """
        available_colors = self.get_colors(state)
        if not available_colors:
            raise ColorsExhaustedError
        candidates = (self.new_candidate_layer(state, x) for x in available_colors)
        scored = ((self.get_cost(*state.layers, x), x) for x in candidates)
        winner = min(scored, key=itemgetter(0))[1]
        return winner


def _expand_layers(
    quantized_image: Annotated[IntA, "(r, c)"],
    d1_layers: Annotated[IntA, "(n, 512)"],
) -> Annotated[IntA, "(n, r, c)"]:
    """Expand layers to the size of the quantized image.

    :param quantized_image: (r, c) array with palette indices
    :param d1_layers: (n, 512) an array of layers. Layers may contain -1 or any
        palette index in [0, 511].
    :return: (n, r, c) array of layers, each layer with the same shape as the
        quantized image.
    """
    return np.array([x[quantized_image] for x in d1_layers])


def draw_target(
    target: TargetImage, state: Layers, num_cols: int | None = None, stem: str = ""
) -> None:
    """Infer a name from TargetImage args and write image to file.

    This is for debugging how well image is visually represented and what colors
    might be "eating" others in the image.
    """
    vectors = target.vectors
    stem_parts = (target.get_cache_stem(state), len(state.layers), num_cols, stem)
    output_stem = "-".join(_stemize(*stem_parts))

    big_layers = _expand_layers(target.image, state.layers)
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
) -> tuple[TargetImage, Layers]:
    """Posterize an image.

    :param image_path: path to the image
    :param bite_size: the max average error of the cluster removed for each color
    :param ixs: optionally pass a subset of the palette indices to use. This is used
        with bite_size == 0 to generate an image with exactly the colors in ixs.
    :return: posterized image
    """
    ignore_cache = True

    target = TargetImage(image_path)

    state = Layers(set(target.clusters.ixs), bite_size)

    cache_path = _new_cache_path(image_path, bite_size, num_cols, suffix=".npy")
    if cache_path.exists() and not ignore_cache:
        target.layers = np.load(cache_path)
    else:
        target.fill_layers(state, num_cols or 1)

    np.save(cache_path, state.layers)

    if len(state.layers) < (num_cols or 1):
        return posterize(
            image_path,
            max(bite_size - 1, 0),
            ixs,
            num_cols,
            ignore_cache=ignore_cache,
        )

    return target, state
