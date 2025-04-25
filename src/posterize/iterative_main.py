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

import dataclasses
from pathlib import Path
from typing import Annotated, Any, Iterable, Iterator, TypeAlias

import numpy as np
from cluster_colors import SuperclusterBase
from numpy import typing as npt

from posterize.image_processing import draw_posterized_image
from posterize.quantization import quantize_image

_IntA: TypeAlias = npt.NDArray[np.intp]


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


def _new_empty_layers() -> _IntA:
    """Create an empty layers array.

    :return: (0, 512) array of layers. Each layer is a palette index or -1 for
        transparent.
    """
    return np.empty((0, 512), dtype=int)


def _merge_layers(*layers: _IntA) -> _IntA:
    """Merge layers into a single layer.

    :param layers: n shape (512,) layer arrays, each containing at most two values:
        * a color index that will replace one or more indices in the quantized image
        * -1 for transparent. The first layer will be a solid color and contain no -1
    :return: one (512,) array with the last non-transparent color in each position

    Where an image is a (rows, cols) array of indices---each layer of an
    approximation will color some of those indices with one palette index per layer,
    and others with -1 for transparency.
    """
    if len(layers) == 0:
        return np.full((512,), -1, dtype=int)
    merged = layers[0].copy()
    for layer in layers[1:]:
        merged[np.where(layer != -1)] = layer[np.where(layer != -1)]
    return merged


def _apply_mask(layer: _IntA, map: _IntA | None) -> _IntA:
    """Apply a map to a layer if the map is not None.

    :param layer: the layer to apply the map to (shape (512,) consisting of one
        palette index and -1 where transparent)
    :param map: the map to apply to the layer (shape (512,)) consisting of 1s and 0s
    :return: the layer with the map applied (shape (512,)) with, most likely,
        additional transparent (-1) values
    """
    if map is None:
        return layer
    return np.where(map == 1, layer, -1)


@dataclasses.dataclass
class ImageApproximation:
    """State for an image approximation.

    :param colors: the subset of color indices (TargetImage.clusters.ixs) available
        for use in layers.
    :param layers: (n, c) array of n layers, each containing a value (color index) in
        colors and -1 for transparent. The first layer will be a solid color and
        contain no -1 values.
    """

    target_image: TargetImage
    colors: tuple[int, ...]
    min_delta: float

    def __init__(
        self,
        target_image: TargetImage,
        min_delta: float,
        colors: Iterable[int] | None = None,
        layers: _IntA | None = None,
    ) -> None:
        self.target = target_image
        self.min_delta = min_delta
        if colors is None:
            self.colors = tuple(range(512))
        else:
            self.colors = tuple(colors)
        if layers is None:
            self.layers = _new_empty_layers()
        else:
            self.layers = layers
        self.cached_states: dict[tuple[int, ...], float] = {}

    @property
    def layer_colors(self) -> list[int]:
        """Get the non-transparent color in each layer."""
        return [int(np.max(x)) for x in self.layers]

    def get_layer_color(self, index: int) -> int:
        """Get the color of a layer."""
        return int(np.max(self.layers[index]))

    def get_available_colors(self) -> list[int]:
        """Get available colors in the i.wmage.

        The available colors will depend on three things:
        1. The 512 colors in the quantized image
        2. The colors used in previous layers
        3. The min_delta defined for the current state
        """
        if len(self.layers) == 0:
            return list(self.colors)
        layer_colors = self.layer_colors
        assert -1 not in layer_colors
        layer_prox = self.target.pmatrix[:, layer_colors]
        return [x for x in self.colors if min(layer_prox[x]) > self.min_delta]

    def _add_one_layer(self, mask: _IntA | None = None) -> None:
        """Add one layer to the state."""
        new_layer = self.get_best_candidate_layer(mask=mask)
        self.layers = np.append(self.layers, [new_layer], axis=0)

    def fill_layers(self, num_layers: int) -> None:
        """Add layers until there are num_layers. Then check lower layers.

        :param state: the ImageApproximation instance to be updated.
        :param num_layers: the number of layers to end up with. Will silently return
            fewer layers if all colors are exhausted.
        :effect: update state.layers
        """
        if len(self.layers) >= num_layers:
            return
        for _ in range(num_layers - len(self.layers)):
            self._add_one_layer()

    def two_pass_fill_layers(self, num_layers: int) -> None:
        for i in range(2, num_layers + 1):
            self.fill_layers(i)
            image = _merge_layers(*self.layers)
            image_masks = np.array(
                [np.where(image == x, 1, 0) for x in self.layer_colors]
            )

            self.layers.resize((0, 512), refcheck=False)
            for mask in image_masks:
                self._add_one_layer(mask=mask)

    # ===============================================================================
    #   Define and select new candidate layers
    # ===============================================================================

    def _new_candidate_layer(self, palette_index: int) -> _IntA:
        """Create a new candidate state.

        :param palette_index: the index of the color to use in the new layer
        :param state: a ImageApproximation instance

        A candidate is the current state with state indices replaced with
        palette_index where palette_index has a lower cost that the index in state at
        that position.

        If there are no layers, the candidate will be a solid color.
        """
        solid = np.full(self.target.pmatrix.shape[0], palette_index)
        if len(self.layers) == 0:
            return solid

        solid_cost_matrix = self.target.get_cost_matrix(solid)
        state_cost_matrix = self.target.get_cost_matrix(*self.layers)
        return np.where(state_cost_matrix > solid_cost_matrix, palette_index, -1)

    def get_best_candidate_layer(self, mask: _IntA | None = None) -> _IntA:
        """Get the best candidate layer to add to layers.

        :param state_layers: the current state or a presumed state
        :return: the candidate layer with the lowest cost
        """
        state = _merge_layers(*self.layers)
        state_cost = self.target.get_cost(state, mask=mask)
        available_colors = self.get_available_colors()

        if not available_colors:
            raise ColorsExhaustedError

        candidates = [self._new_candidate_layer(x) for x in available_colors]
        scores = [self.target.get_cost(state, x, mask=mask) for x in candidates]

        if state_cost == np.inf:
            state_cost = max(scores)

        savings = [state_cost - x for x in scores]

        pixels: list[np.intp] = []
        for candidate in candidates:
            masked = _apply_mask(candidate, mask)
            weights = self.target.weights[np.where(masked != -1)]
            pixels.append(np.sum(weights))

        avgs = [
            float(s / p) if p > 0 else 0.0 for s, p in zip(savings, pixels, strict=True)
        ]

        savings = [x / max(savings) for x in savings]
        avgs = [x / max(avgs) for x in avgs]

        savings_weight = 0.25
        more_is_better = [
            s * savings_weight + p * (1 - savings_weight) for s, p in zip(savings, avgs)
        ]

        best_idx = np.argmax(np.array(more_is_better))
        return candidates[best_idx]

    def get_cache_stem(self) -> str:
        min_delta = f"{self.min_delta:05.2f}".replace(".", "_")
        return f"{self.target.path.stem}-{min_delta}"


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

    def get_cost_matrix(self, *layers: _IntA) -> npt.NDArray[np.floating[Any]]:
        """Get the cost-per-pixel between self.image and (state + layers).

        :param layers: layers to apply to the current state. There will only ever be
            0 or 1 layers. If 0, the cost matrix of the current state will be
            returned.  If 1, the cost matrix with a layer applied over it.
        :return: cost-per-pixel between image and (state + layer)

        This should decrease after every append.
        """
        state = _merge_layers(*layers)
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


def _expand_layers(
    quantized_image: Annotated[_IntA, "(r, c)"],
    d1_layers: Annotated[_IntA, "(n, 512)"],
) -> Annotated[_IntA, "(n, r, c)"]:
    """Expand layers to the size of the quantized image.

    :param quantized_image: (r, c) array with palette indices
    :param d1_layers: (n, 512) an array of layers. Layers may contain -1 or any
        palette index in [0, 511].
    :return: (n, r, c) array of layers, each layer with the same shape as the
        quantized image.
    """
    return np.array([x[quantized_image] for x in d1_layers])


def draw_approximation(
    state: ImageApproximation,
    num_cols: int | None = None,
    stem: str = "",
) -> None:
    """Infer a name from the state and draw the approximation.

    This is for debugging how well image is visually represented and what colors
    might be "eating" others in the image.
    """
    stem_parts = (state.get_cache_stem(), len(state.layers), num_cols, stem)
    output_stem = "-".join(_stemize(*stem_parts))

    big_layers = _expand_layers(state.target.image, state.layers)
    draw_posterized_image(state.target.vectors, big_layers[:num_cols], output_stem)


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


def posterize(
    image_path: Path,
    min_delta: float,
    num_cols: int,
    *,
    ignore_cache: bool = True,
) -> ImageApproximation:
    """Posterize an image.

    :param image_path: path to the image
    :param min_delta: the minimum delta_e between colors in the final image. This
        will be lowered if necessary to achieve the desired number of colors.
    :param num_cols: the number of colors in the posterization image
    :return: posterized image
    """
    target = TargetImage(image_path)
    state = ImageApproximation(target, min_delta)
    state.two_pass_fill_layers(num_cols)
    return state
