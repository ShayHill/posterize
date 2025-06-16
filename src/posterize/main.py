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

from collections.abc import Iterable
import dataclasses
from pathlib import Path
from typing import Annotated, Iterable, Iterator, TypeAlias

import numpy as np
from numpy import typing as npt
from posterize.color_attributes import get_vibrance
from posterize.image_processing import draw_posterized_image
from posterize.quantization import new_target_image, TargetImage

from posterize.layers import merge_layers, apply_mask

_IntA: TypeAlias = npt.NDArray[np.intp]
_FltA: TypeAlias = npt.NDArray[np.float64]
_RGB: TypeAlias = Annotated[npt.NDArray[np.uint8], (3,)]

# Default weight for sum savings vs. average savings. Average savings is, by default,
# weighted highly. These values are used when selecting the best candidate for the
# next layer color. A higher average savings weight means colors that improve the
# approximation a lot in a small area are chosen over colors that improve the
# approximation a tiny amount over a large area. _DEFAULT_SAVINGS_WEIGHT is the
# weight given to sum savings. (1 - _DEFAULT_SAVINGS_WEIGHT) is the weight given to
# average savings.
_DEFAULT_SAVINGS_WEIGHT = 0.25


# A higher number (1.0 is maximum) means colors that are more vibrant are more likely
# to be selected as layer colors. The default is 0.0, which will be good for most
# images, but the parameter is available if you have an overall drab image with a few
# bright highlights and want to pay less attention to the background.
_DEFAULT_VIBRANT_WEIGHT = 0.0


class ColorsExhaustedError(Exception):
    """Exception raised when a new layer is requested, but no colors are available."""

    def __init__(self, message: str = "No available colors.") -> None:
        self.message = message
        super().__init__(self.message)


def _set_max_val_to_one(floats: Iterable[float]) -> _FltA:
    """Scale an array so the maximum value is 1.

    :param array: the array to scale
    :return: the scaled array

    This is used to scale layer savings and layer weights to a similar range before
    taking a weighted average. I did not use min-max scaling because I would prefer
    layers with similar savings or weights to be treated as roughly equivalent. I
    don't want to treat one as 1 (full weight) and the other as 0 (no weight) when
    they are very nearly equal.

    The scaling here allows for values to cluster around 1.0 if the input values are
    all high.
    """
    array = np.array(list(floats))
    max_val = np.max(array)
    if max_val == 0:
        return array
    return array / max_val


@dataclasses.dataclass
class ImageApproximation:
    """State for an image approximation.

    :param target_image: the quantized image to approximate
    :param colors: the subset of color indices (TargetImage.clusters.ixs) available
        for use in layers.
    :param layers: (n, c) array of n layers, each containing a value (color index) in
        colors and -1 for transparent. The first layer will be a solid color and
        contain no -1 values. This is a parameter for algorithms that might want to
        restart an ImageApproximation from a previous state, but you can for the most
        part ignore it.
    :param savings_weight: optional kwarg only param - weight for sum savings vs
        average savings when selecting a layer color.
    :param vibrant_weight: optional kwarg only param - [0.0, 1.0] increase tendency
        to select vibrant colors.
    """

    target_image: TargetImage
    colors: tuple[int, ...]

    def __init__(
        self,
        target_image: TargetImage,
        colors: Iterable[int] | None = None,
        layers: _IntA | None = None,
        *,
        savings_weight: float | None = None,
        vibrant_weight: float | None = None,
    ) -> None:
        self.target = target_image
        if colors is None:
            self.colors = tuple(range(len(target_image.palette)))
        else:
            self.colors = tuple(colors)
        if layers is None:
            self.layers = np.empty((0, len(self.colors)), dtype=int)
        else:
            self.layers = layers
        self.cached_states: dict[tuple[int, ...], float] = {}
        self.savings_weight = savings_weight or _DEFAULT_SAVINGS_WEIGHT
        self.vibrant_weight = vibrant_weight or _DEFAULT_VIBRANT_WEIGHT

        vibrancies = np.array(list(map(get_vibrance, self.target.palette)))
        self.target.weights *= 1 - self.vibrant_weight
        vibrancies *= self.vibrant_weight
        self.target.weights += vibrancies

    @property
    def layer_colors(self) -> list[int]:
        """Get the non-transparent color in each layer."""
        return [int(np.max(x)) for x in self.layers]

    def get_layer_masks(self) -> _IntA:
        """Get the masks for each layer.

        :return: (n, r, c) array of masks for each layer
        """
        image = merge_layers(*self.layers)
        return np.array([np.where(image == x, 1, 0) for x in self.layer_colors])

    def get_available_colors(self) -> list[int]:
        """Get available colors in the image."""
        if len(self.layers) == 0:
            return list(self.colors)
        layer_colors = self.layer_colors
        assert -1 not in layer_colors
        return [x for x in self.colors if x not in layer_colors]

    def _add_one_layer(self, mask: _IntA | None = None) -> None:
        """Add one layer to the state."""
        new_layer = self.get_best_candidate_layer(mask=mask)
        self.layers = np.append(self.layers, [new_layer], axis=0)

    def fill_layers(self, num_layers: int) -> None:
        """Add layers until there are num_layers.

        :param num_layers: the number of layers to end up with. Will silently return
            fewer layers if all colors are exhausted.
        :effect: update state.layers
        """
        if len(self.layers) >= num_layers:
            return
        try:
            self._add_one_layer()
        except ColorsExhaustedError:
            return
        self.fill_layers(num_layers)

    def two_pass_fill_layers(self, num_layers: int) -> None:
        """Fill layers to create masks then fill masks to create layers.

        :param num_layers: the number of layers to end up with. Will silently return
            fewer layers if all colors are exhausted.
        :effect: update state.layers

        The first pass is only to create masks. The colors selected in the first pass
        will try to approximate a larger area than the mask will cover. This creates
        the Japanese flag effect described in the module docstring.

        The second pass selects colors that only try to approximate the mask area.
        """
        if len(self.layers) >= num_layers:
            return
        try:
            self._add_one_layer()
        except ColorsExhaustedError:
            return
        if len(self.layers) >= 2:
            layer_masks = self.get_layer_masks()
            self.layers.resize((0, self.layers.shape[1]), refcheck=False)
            for mask in layer_masks:
                self._add_one_layer(mask=mask)
        self.two_pass_fill_layers(num_layers)

    # ===============================================================================
    #   Define and select new candidate layers
    # ===============================================================================

    def _new_candidate_layer(self, palette_index: int) -> _IntA:
        """Create a new candidate layer.

        :param palette_index: the index of the color to use in the new layer
        :return: a new candidate layer with the same shape as the quantized image.
            Pixels where palette_index would improve the approximation are set to
            palette_index. Pixels where palette_index would not improve the
            approximation are set to -1.

        If there are no layers, the candidate will be a solid color.
        """
        solid = np.full(self.target.pmatrix.shape[0], palette_index)
        if len(self.layers) == 0:
            return solid

        solid_cost_matrix = self.target.get_cost_matrix(solid)
        state_cost_matrix = self.target.get_cost_matrix(*self.layers)
        return np.where(state_cost_matrix > solid_cost_matrix, palette_index, -1)

    def _sum_masked_weight(self, layer: _IntA, mask: None | _IntA) -> np.float64:
        """Mask a layer then sum the weights.

        This is a subroutine for get_best_candidate_layer.
        """
        masked = apply_mask(layer, mask)
        weights = self.target.weights[np.where(masked != -1)]
        return np.sum(weights)

    def get_best_candidate_layer(self, mask: _IntA | None = None) -> _IntA:
        """Get the best candidate layer to add to layers.

        :param state_layers: the current state or a presumed state
        :return: the candidate layer with the lowest cost
        """
        state = merge_layers(*self.layers)
        state_cost = self.target.get_cost(state, mask=mask)
        available_colors = self.get_available_colors()

        if not available_colors:
            raise ColorsExhaustedError

        candidates = [self._new_candidate_layer(x) for x in available_colors]
        scores = [self.target.get_cost(state, x, mask=mask) for x in candidates]

        if state_cost == np.inf:
            state_cost = max(scores)

        layer_savings = _set_max_val_to_one([state_cost - x for x in scores])
        layer_weights = _set_max_val_to_one(
            [self._sum_masked_weight(x, mask) for x in candidates]
        )
        layer_averages = [
            float(s / p) if p > 0 else 0.0
            for s, p in zip(layer_savings, layer_weights, strict=True)
        ]

        more_is_better = [
            s * self.savings_weight + a * (1 - self.savings_weight)
            for s, a in zip(layer_savings, layer_averages)
        ]

        best_idx = np.argmax(np.array(more_is_better))
        return candidates[best_idx]

def _percentage_infix(float_: float) -> str:
    """Get a string to use in the filename for a percentage."""
    return f"{int(float_*100):02}"


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
    source_image: Path,
    state: ImageApproximation,
    num_cols: int | None = None,
    stem: str = "",
) -> None:
    """Infer a name from the state and draw the approximation.

    This is for debugging how well image is visually represented and what colors
    might be "eating" others in the image.
    """
    big_layers = _expand_layers(state.target.indices, state.layers)
    draw_posterized_image(state.target.palette, big_layers[:num_cols], stem)


def stemize(*args: Path | float | int | str | None) -> Iterator[str]:
    """Convert args to strings and filter out empty strings."""
    if not args:
        return
    arg, *tail = args
    if arg is None:
        pass
    elif isinstance(arg, str):
        yield arg
    elif isinstance(arg, Path):
        yield arg.stem
    elif isinstance(arg, float):
        assert 0 <= arg <= 1
        yield _percentage_infix(arg)
    else:
        assert isinstance(arg, int)
        yield f"{arg:03d}"
    yield from stemize(*tail)


def posterize(
    image_path: Path,
    num_cols: int,
    *,
    savings_weight: None | float = None,
    vibrant_weight: None | float = None,
) -> ImageApproximation:
    """Posterize an image.

    :param image_path: path to the image
    :param num_cols: the number of colors in the posterization image
    :param savings_weight: weight for the savings metric vs average savings
    :param vibrant_weight: weight for the vibrance metric vs savings metric
    :return: posterized image
    """
    target = new_target_image(image_path)
    state = ImageApproximation(
        target, savings_weight=savings_weight, vibrant_weight=vibrant_weight
    )
    state.two_pass_fill_layers(num_cols)
    return state
